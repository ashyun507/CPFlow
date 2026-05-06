import torch  # PyTorch 张量库；本文件所有张量运算都基于它。
from torch import nn  # 神经网络层与 Module 基类。

from pepflow.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles  # 几何工具：构局部坐标系、全局转局部、算主链二面角。
from pepflow.modules.common.layers import AngularEncoding  # 角度编码器：把角度值映射到 sin/cos 风格高维特征。
from pepflow.modules.protein.constants import BBHeavyAtom, AA  # 蛋白常量：主链重原子索引、氨基酸类型枚举。
from models_con.utils import get_chain_local_index, get_index_embedding  # 固定绝对位置编码：先构造每条链内部的 0-based index，再做 sin/cos 编码。


class NodeEmbedder(nn.Module):  # 残基级节点编码器：把每个 residue 编成一个 node feature，供后续 trunk/GAEncoder 使用。

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):  # feat_dim=节点输出维度 F；max_num_atoms=每个残基最多保留多少个重原子 A；max_aa_types=残基类别数。
        super().__init__()  # 初始化 nn.Module 基类。
        if feat_dim % 2 != 0:  # 这里把小分支宽度固定成 feat_dim 的一半，所以要求最终节点宽度是偶数。
            raise ValueError(f"NodeEmbedder feat_dim must be even, got {feat_dim}.")
        self.max_num_atoms = max_num_atoms  # 保存原子上限 A，后续会把输入原子截到这个长度。
        self.max_aa_types = max_aa_types  # 保存氨基酸类别数 K，默认 22（20 标准 aa + 特殊类别）。
        self.feat_dim = feat_dim  # 保存节点特征维度 F。
        self.token_feat_dim = feat_dim // 2  # 小型离散/位置分支统一走 F/2；当前 256 -> 128。
        self.aatype_embed = nn.Embedding(self.max_aa_types, self.token_feat_dim)  # aa 离散标签 -> [F/2] 向量；输入 [B, L] 输出 [B, L, F/2]。
        self.dihed_embed = AngularEncoding()  # 主链二面角编码器；输入角度值，输出高维角特征。
        self.crd_feat_dim = self.max_aa_types * max_num_atoms * 3  # 局部坐标这一路保持原始展开维度；当前 22*15*3=990。
        self.dihed_feat_dim = self.dihed_embed.get_out_dim(3)  # 主链角这一路保持原始角编码维度；当前 39。

        infeat_dim = self.token_feat_dim + self.token_feat_dim + self.crd_feat_dim + self.dihed_feat_dim  # aa/idx 小分支各用 F/2，crd/dihed 维度保持原始值，最后统一融合到 F。
        self.mlp = nn.Sequential(  # 节点特征投影 MLP：把拼接后的高维手工特征压到统一的 feat_dim。
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),  # 第 1 层：[B, L, Cin] -> [B, L, 2F]。
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),  # 第 2 层：[B, L, 2F] -> [B, L, F]。
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),  # 第 3 层：[B, L, F] -> [B, L, F]。
            nn.Linear(feat_dim, feat_dim)  # 第 4 层：输出最终节点向量 [B, L, F]。
        )
    
    # def embed_t(self, timesteps, mask):  # 旧的 timestep embedding 接口，当前未使用；保留作参考。
    #     timestep_emb = get_time_embedding(  # 把扩散时间步编码成位置向量。
    #         timesteps[:, 0],  # 输入 [B]。
    #         self.feat_dim,  # 输出维度 F。
    #         max_positions=2056  # 时间 embedding 的最大位置尺度。
    #     )[:, None, :].repeat(1, mask.shape[1], 1)  # 扩成 [B, L, F]，让每个残基都收到同样的时间特征。
    #     return timestep_emb  # 返回 [B, L, F]。

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):  # 主前向：把 residue 的序列+局部几何编码成节点特征。
        """
        Args:
            aa:         (B, L).  # 残基类型 ID；B=batch 大小，L=残基总数（受体+pocket+peptide 拼在一起）。
            res_nb:     (B, L).  # 链内残基编号；当前 node 特征里不直接 embedding，只用于二面角拓扑判断。
            chain_nb:   (B, L).  # 链编号；用于判断相邻残基是否真在同一条链上。
            pos_atoms:  (B, L, A_all, 3).  # 每个残基的重原子坐标；A_all 是输入保留的原子槽位数。
            mask_atoms: (B, L, A_all).  # 原子存在性 mask；True/1 表示该槽位有真实原子。
            structure_mask: (B, L), mask out unknown structures to generate.  # 结构内容可见 mask；False 的位置避免泄漏 GT 几何。
            sequence_mask:  (B, L), mask out unknown amino acids to generate.  # 序列内容可见 mask；False 的位置避免泄漏 GT aa。
        """
        N, L = aa.size()  # N 这里其实是 batch 维 B；L 是残基数。后续代码沿用原变量名 N。
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # [B, L]；用 CA 是否存在作为“这个 residue 是否有效”的残基级 mask。

        # Remove other atoms  # 只保留前 max_num_atoms 个重原子槽位，和 embedder 预设维度对齐。
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]  # [B, L, A, 3]；A=max_num_atoms。
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]  # [B, L, A]；与 pos_atoms 对齐的原子 mask。

        # Amino acid identity features  # 第 1 路特征：残基类型。
        if sequence_mask is not None:  # 如果当前 pipeline 指定某些位置的序列不可见，就在 embed 前先抹掉。
            # Avoid data leakage at training time  # 训练时避免把待生成位置的真实 aa 直接喂给模型。
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))  # [B, L]；不可见位置替换成 UNK。
        aa_feat = self.aatype_embed(aa)  # [B, L, F]；离散 aa ID -> learned embedding。

        # Absolute chain-local index features  # 第 2 路特征：每条 chain 内部从 0 开始的绝对位置编码；线性链和环肽都统一使用。
        chain_local_idx = get_chain_local_index(chain_nb, mask_residue)  # [B, L]；每条链自己的 residue 顺序号，跨链会重新从 0 开始。
        idx_feat = get_index_embedding(chain_local_idx, self.token_feat_dim)  # [B, L, F/2]；固定 sin/cos 绝对位置编码，不引入新的 learnable lookup table。
        idx_feat = idx_feat * mask_residue[:, :, None]  # [B, L, F/2]；无效 residue 的位置编码清零。

        # Coordinate features  # 第 3 路特征：把每个残基的局部原子排布编码成固定长度向量。
        R = construct_3d_basis(  # 利用 backbone 的 CA/C/N 三点构局部坐标系。
            pos_atoms[:, :, BBHeavyAtom.CA],  # [B, L, 3]；局部坐标系原点附近的 CA。
            pos_atoms[:, :, BBHeavyAtom.C],  # [B, L, 3]；定义一个轴方向。
            pos_atoms[:, :, BBHeavyAtom.N]  # [B, L, 3]；定义另一个轴方向。
        )  # 输出 R 形状 [B, L, 3, 3]；每个残基一个局部正交基。
        t = pos_atoms[:, :, BBHeavyAtom.CA]  # [B, L, 3]；把 CA 坐标作为局部坐标系的平移中心。
        crd = global_to_local(R, t, pos_atoms)  # [B, L, A, 3]；把所有原子从全局坐标转到 residue 局部坐标。
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)  # [B, L, A, 3]；把原子 mask 扩到 xyz 三个坐标分量上。
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))  # [B, L, A, 3]；不存在的原子坐标清零，避免噪声。

        aa_expand = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)  # [B, L, K, A, 3]；把 aa 标签扩展到 aa-type 维与原子维。
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)  # [B, L, K, A, 3]；每个 aa-type 槽位对应自己的类别索引。
        place_mask = (aa_expand == rng_expand)  # [B, L, K, A, 3]；只在真实 aa 对应的类别槽位上为 True。
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)  # [B, L, K, A, 3]；把局部坐标复制到每个 aa-type 槽位。
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))  # [B, L, K, A, 3]；只保留真实 aa 对应槽位的局部原子坐标，其余类型槽位清零。
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types * self.max_num_atoms * 3)  # [B, L, K*A*3]；把“按 aa-type 放置”的局部坐标展平成一个大向量。
        if structure_mask is not None:  # 如果当前 pipeline 指定某些位置的结构不可见，就在这里清零。
            # Avoid data leakage at training time  # 防止把待生成位置的 GT 局部几何直接喂给模型。
            crd_feat = crd_feat * structure_mask[:, :, None]  # [B, L, K*A*3]；不可见 residue 的局部坐标特征整体清零。
        crd_feat = crd_feat * mask_residue[:, :, None]  # [B, L, 990]；无效 residue 清零。

        # Backbone dihedral features  # 第 4 路特征：主链二面角。
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)  # bb_dihedral:[B,L,3]；mask_bb_dihed:[B,L,3]，3 个角通常对应 omega/phi/psi 可用性。
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # [B, L, 3, C_angle]；每个角各自编码，不可用角清零。
        dihed_feat = dihed_feat.reshape(N, L, -1)  # [B, L, 3*C_angle]；把 3 个角的编码拼平。
        if structure_mask is not None:  # 若结构不可见，还需要进一步防止 anchor residue 的角度泄漏。
            # Avoid data leakage at training time  # 因为某个 residue 的二面角会用到前后相邻残基，直接乘 structure_mask 仍可能泄漏边界信息。
            dihed_mask = torch.logical_and(  # 构造更严格的“角度可见”掩码。
                structure_mask,  # [B, L]；当前位置本身必须结构可见。
                torch.logical_and(  # 还要求左右相邻位置也结构可见。
                    torch.roll(structure_mask, shifts=+1, dims=1),  # [B, L]；左邻居可见。
                    torch.roll(structure_mask, shifts=-1, dims=1)  # [B, L]；右邻居可见。
                ),
            )  # [B, L]；仅在当前位置与两侧邻居都可见时为 True。
            dihed_feat = dihed_feat * dihed_mask[:, :, None]  # [B, L, 3*C_angle]；边界位置角度特征清零。
        dihed_feat = dihed_feat * mask_residue[:, :, None]  # [B, L, 39]；无效 residue 清零。
        
        # # timestep  # 如需把扩散时间并入节点特征，可在这里恢复旧逻辑；当前 pipeline 不使用。
        # timestep_emb = self.embed_t(timesteps, mask_residue)  # 预期形状 [B, L, F]。

        out_feat = self.mlp(torch.cat([aa_feat, idx_feat, crd_feat, dihed_feat], dim=-1))  # [B, L, F]；把 aa[F/2] / 绝对位置[F/2] / 局部坐标[990] / 主链角[39] 拼接后统一投影到最终 node 宽度。
        out_feat = out_feat * mask_residue[:, :, None]  # [B, L, F]；对无效 residue（没有 CA）的节点特征整体清零。

        # print(f'aa_seq:{aa},aa:{aa_feat},crd:{crd_feat},dihed:{dihed_feat},time:{timestep_emb}')  # 旧调试打印：查看各路节点特征的数值。

        # print(f'weight:{self.aatype_embed.weight}') # nan, why?  # 旧调试打印：查看 aa embedding 参数是否出现数值异常。

        return out_feat  # 返回节点特征 [B, L, F]；后续进入 trunk/GAEncoder 作为 per-residue 输入。
