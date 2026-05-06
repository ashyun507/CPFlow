import torch  # PyTorch 张量库；本文件所有残基对特征都用它构造。
import torch.nn as nn  # 神经网络层定义。
import torch.nn.functional as F  # softplus 等逐元素函数。


from pepflow.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals  # 距离单位换算、残基对方向角计算。
from pepflow.modules.common.layers import AngularEncoding  # 角度编码器：把 2 个 pairwise dihedral 变成高维特征。
from pepflow.modules.protein.constants import BBHeavyAtom, AA  # 蛋白常量：主链原子索引、UNK 残基类型等。
from models_con.cyclic_edge_priors import CYCLIC_ADJACENT_PRIORS, CYCLIC_TERMINAL_PRIORS  # 按成环类型预统计好的 128 维 Cβ 距离先验。
from models_con.utils import get_chain_local_index, get_index_embedding  # 固定位置编码：每条链内部绝对编号 + pair 上的编号差编码。


class EdgeEmbedder(nn.Module):  # 残基对边编码器：把 (i, j) 这条边编码成 pair feature，供 trunk/GAEncoder 使用。

    def __init__(  # 初始化边特征编码器。
        self,  # Python 实例本身。
        feat_dim,  # 输出边特征维度 F_z。
        max_num_atoms,  # 每个残基保留的最大重原子数 A。
        max_aa_types=22,  # 残基类别数 K。
        max_relpos=32,  # 相对位置裁剪窗口；超过后会 clamp。
        num_bins=16,  # 旧 distogram 相关参数；当前实现里未直接用于主路径。
        position_cfg=None,  # 位置编码配置；可选 linear / cyclic。
    ):
        super().__init__()  # 初始化 nn.Module 基类。
        if feat_dim % 2 != 0:  # 这里把小型 pair 分支宽度固定成 feat_dim 的一半，所以要求最终 edge 宽度是偶数。
            raise ValueError(f"EdgeEmbedder feat_dim must be even, got {feat_dim}.")
        self.max_num_atoms = max_num_atoms  # 保存原子上限 A。
        self.max_aa_types = max_aa_types  # 保存残基类别数 K。
        self.max_relpos = max_relpos  # 保存相对位置裁剪上限。
        self.num_bins = num_bins  # 保存 distogram bin 数；当前主要是兼容旧接口。
        self.feat_dim = feat_dim  # 保存输出边特征维度 F，固定 idx 编码会直接用这个维度输出。
        self.pair_feat_dim = feat_dim // 2  # 小型 pair 分支统一走 F/2；当前 128 -> 64。
        self.position_cfg = position_cfg  # 原始位置配置对象，便于后续读取更多参数。
        self.position_type = getattr(position_cfg, "type", "linear") if position_cfg is not None else "linear"  # 位置模式保留给配置兼容使用；当前 linear/cyclic 都统一走固定 idx 编码。
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, self.pair_feat_dim)  # aa-pair embedding：输入 [B,L,L] 的 pair ID，输出 [B,L,L,F/2]。
        if self.position_type not in {"linear", "cyclic"}:  # 当前只接受这两种配置值；功能上二者现在共享同一套 idx 编码逻辑。
            raise ValueError(  # 显式失败，避免静默走错编码逻辑。
                f"Unsupported encoder.position.type: {self.position_type}"  # 错误信息里带出实际配置值。
            )

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)  # 对每种 aa-pair 学一组距离核宽度参数；输出 [A*A]。
        nn.init.zeros_(self.aapair_to_distcoef.weight)  # 初始化为 0；经 softplus 后约为常数，训练中再学习不同 aa-pair 的距离响应。
        self.distance_embed = nn.Sequential(  # 原子两两距离特征 -> 边特征的 MLP。
            nn.Linear(max_num_atoms * max_num_atoms, self.pair_feat_dim), nn.ReLU(),  # [B,L,L,A*A] -> [B,L,L,F/2]。
            nn.Linear(self.pair_feat_dim, self.pair_feat_dim), nn.ReLU(),  # [B,L,L,F/2] -> [B,L,L,F/2]。
        )

        self.dihedral_embed = AngularEncoding()  # pairwise dihedral 编码器。
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # 2 个 pairwise dihedral 角编码后的总维度。
        self.cyclic_prior_dim = 128 if self.position_type == "cyclic" else 0  # 只有 cyclic 模式才追加 128 维的环肽链距离先验。
        if self.position_type == "cyclic":  # 把统计得到的 128-bin 先验写成 buffer，训练时不更新。
            self.register_buffer("cyclic_adjacent_priors", CYCLIC_ADJACENT_PRIORS.clone())  # [3,128]；按类型给相邻边的固定向量。
            self.register_buffer("cyclic_terminal_priors", CYCLIC_TERMINAL_PRIORS.clone())  # [3,128]；按类型给首尾边的固定向量。

        infeat_dim = self.pair_feat_dim + self.pair_feat_dim + self.pair_feat_dim + feat_dihed_dim + self.cyclic_prior_dim  # aa-pair / idx-diff / atom-dist 走 F/2，小几何分支保持原始维度；cyclic 模式额外追加原始 128 维先验。
        self.out_mlp = nn.Sequential(  # 最终边特征融合 MLP。
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),  # [B,L,L,Cin] -> [B,L,L,F]。
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),  # [B,L,L,F] -> [B,L,L,F]。
            nn.Linear(feat_dim, feat_dim),  # 输出统一边特征 [B,L,L,F]。
        )

    def _chain_local_idx_features(self, chain_nb, mask_residue, same_chain):  # 用每条链内部的 0-based 顺序号构造固定位置编码；node 用绝对 idx，edge 用 idx_i-idx_j。
        chain_local_idx = get_chain_local_index(chain_nb, mask_residue)  # [B,L]；每条链自己从 0 开始编号，跨链重新从 0 开始。
        rel_idx = chain_local_idx[:, :, None] - chain_local_idx[:, None, :]  # [B,L,L]；pair 上使用 idx_i - idx_j，符合用户要求的 edge 位置编码方式。
        rel_idx = torch.clamp(rel_idx, min=-self.max_relpos, max=self.max_relpos)  # [B,L,L]；对过大的链长差做裁剪，控制 sin/cos 编码数值尺度。
        feat_relpos = get_index_embedding(rel_idx, self.pair_feat_dim)  # [B,L,L,F/2]；固定 sin/cos 编码直接输出小型 pair 特征维度，替换原先 learnable relpos embedding。
        return feat_relpos * same_chain[:, :, :, None]  # [B,L,L,F/2]；只在同链 pair 上保留 idx 差编码，不同链 pair 清零。

    def _cyclic_prior_features(self, chain_nb, mask_residue, cyclic_mask, cyclic_type_id, dtype):  # 只在 cyclic 模式下，把“相邻边 / 首尾边”的固定 128 维先验写进 edge feature。
        B, L = chain_nb.shape  # B=batch，L=残基数。
        feat_cyclic_prior = torch.zeros(  # 默认所有边这 128 维都是 0；受体边、受体-肽边、非相邻肽边都会保持 0。
            B,
            L,
            L,
            self.cyclic_prior_dim,
            device=chain_nb.device,
            dtype=dtype,
        )
        if self.position_type != "cyclic" or cyclic_mask is None or cyclic_type_id is None:  # 非 cyclic 模式或拿不到类型标签时，直接返回全 0。
            return feat_cyclic_prior

        peptide_mask = cyclic_mask.bool() & mask_residue.bool()  # [B,L]；当前项目里 cyclic_mask 实际传的是 peptide/generate_mask，所以这里锁定肽链残基。
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])  # [B,L,L]；只允许同链 pair 激活这条先验分支。
        peptide_pair = peptide_mask[:, :, None] & peptide_mask[:, None, :] & same_chain  # [B,L,L]；只考虑同一条肽链内部的边。
        peptide_local_idx = get_chain_local_index(chain_nb, peptide_mask)  # [B,L]；在肽链内部重新从 0 开始编号，用来判断相邻边和首尾边。
        abs_sep = (peptide_local_idx[:, :, None] - peptide_local_idx[:, None, :]).abs()  # [B,L,L]；|idx_i-idx_j|。
        adjacent_pair = peptide_pair & (abs_sep == 1)  # [B,L,L]；仅命中 (i,i+1) / (i+1,i)。
        peptide_len = peptide_mask.long().sum(dim=-1)  # [B]；每个样本肽链长度。
        first_mask = peptide_mask & (peptide_local_idx == 0)  # [B,L]；链首残基。
        last_mask = peptide_mask & (peptide_local_idx == (peptide_len[:, None] - 1))  # [B,L]；链尾残基。
        terminal_pair = peptide_pair & (  # [B,L,L]；仅命中 (1,N) / (N,1)。
            (first_mask[:, :, None] & last_mask[:, None, :])
            | (last_mask[:, :, None] & first_mask[:, None, :])
        )

        for b in range(B):  # 按样本选对应类型的固定向量；这里只做常量写入，不引入新的可学习参数。
            type_id = int(cyclic_type_id[b].item())  # 当前样本的环类型 ID；0=headtail,1=isopeptide,2=disulfide，-1 表示未知。
            if type_id < 0 or type_id >= self.cyclic_adjacent_priors.shape[0]:  # 未知类型保持 0，不给错误先验。
                continue
            adj_index = adjacent_pair[b].nonzero(as_tuple=False)  # [M_adj,2]；所有相邻边坐标。
            if adj_index.numel() > 0:
                adj_prior = self.cyclic_adjacent_priors[type_id].to(dtype=dtype)  # [128]；当前类型的相邻边固定分布向量。
                feat_cyclic_prior[b, adj_index[:, 0], adj_index[:, 1]] = adj_prior[None, :].expand(adj_index.size(0), -1)  # 给所有相邻边写入同一向量。
            terminal_index = terminal_pair[b].nonzero(as_tuple=False)  # [M_term,2]；所有首尾边坐标。
            if terminal_index.numel() > 0:
                terminal_prior = self.cyclic_terminal_priors[type_id].to(dtype=dtype)  # [128]；当前类型的首尾边固定分布向量。
                feat_cyclic_prior[b, terminal_index[:, 0], terminal_index[:, 1]] = terminal_prior[None, :].expand(terminal_index.size(0), -1)  # 首尾边若与相邻边重叠，后写入的 terminal prior 会覆盖。
        return feat_cyclic_prior  # [B,L,L,128]；只有 cyclic 模式下的特殊肽边非 0。

    def forward(  # 主前向：构造整张残基对图的边特征。
        self,  # Python 实例本身。
        aa,  # [B,L]；残基类型 ID。
        res_nb,  # [B,L]；链内编号；当前 edge 位置编码已经改成 chain-local idx，不再直接使用 res_nb。
        chain_nb,  # [B,L]；链 ID。
        pos_atoms,  # [B,L,A_all,3]；重原子坐标。
        mask_atoms,  # [B,L,A_all]；原子存在性 mask。
        structure_mask=None,  # [B,L]；结构内容是否可见，训练/采样时防止几何泄漏。
        sequence_mask=None,  # [B,L]；序列内容是否可见，训练/采样时防止 aa 泄漏。
        cyclic_mask=None,  # [B,L]；标出哪些 residue 属于“环肽链”；当前实现里通常传 generate_mask。
        cyclic_type_id=None,  # [B]；每个样本的成环类型 ID，只有 cyclic 模式下会读取它来选择 128 维固定先验向量。
    ):
        """
        Args:
            aa: (B, L).  # 残基类型离散 ID。
            res_nb: (B, L).  # 链内残基编号；用于 relpos 与拓扑。
            chain_nb: (B, L).  # 链编号；用于 same_chain 判断。
            pos_atoms:  (B, L, A, 3)  # 每个残基的重原子坐标。
            mask_atoms: (B, L, A)  # 原子存在性 mask。
            trans, sc_trans: (B,L,3)  # 旧接口注释遗留；当前实现未直接使用这两个变量。
            structure_mask: (B, L)  # 结构可见 mask。
            sequence_mask:  (B, L), mask out unknown amino acids to generate.  # 序列可见 mask。
            cyclic_mask:    (B, L), marks residues that belong to the cyclic peptide chain.  # 环肽残基 mask。

        Returns:
            (B, L, L, feat_dim)  # 整张残基对图的 edge embedding；后续送入 trunk/GAEncoder。
        """
        N, L = aa.size()  # N 这里其实是 batch 维 B；L 是残基数。后续沿用原变量名 N。

        # Remove other atoms  # 与 NodeEmbedder 一样，只保留前 max_num_atoms 个重原子槽位。
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]  # [B,L,A,3]；A=max_num_atoms。
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]  # [B,L,A]；对应原子 mask。

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # [B,L]；用 CA 是否存在作为 residue 有效性标志。
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]  # [B,L,L]；只有两个 residue 都有效时，该 pair 才有效。
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None  # [B,L,L] 或 None；只有两端结构都可见时，pair 几何特征才可见。

        # Pair identities  # 第 1 路边特征：aa-pair identity。
        if sequence_mask is not None:  # 若当前 pipeline 把某些位置序列设为不可见，则先替换成 UNK。
            # Avoid data leakage at training time  # 防止把待生成位置真实 aa 直接喂给 pair feature。
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))  # [B,L]；不可见位置替换为 UNK。
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # [B,L,L]；把 (aa_i, aa_j) 压成单个 pair ID。
        feat_aapair = self.aa_pair_embed(aa_pair)  # [B,L,L,F]；aa-pair ID -> learned pair embedding。
    
        # Relative / positional features  # 第 2 路边特征：不再区分 linear/cyclic；统一改成“每条链内部 idx_i-idx_j 的固定 sin/cos 编码”。
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])  # [B,L,L]；只有同一条链上的 residue pair 才共享同一套链内位置坐标系。
        feat_relpos = self._chain_local_idx_features(chain_nb, mask_residue, same_chain)  # [B,L,L,F]；固定位置编码替换掉原来的 learnable linear/cyclic relpos embedding。

        # Distances  # 第 3 路边特征：两残基所有原子两两距离，经 aa-pair 自适应高斯核后再投影。
        d = angstrom_to_nm(torch.linalg.norm(  # 先算欧氏距离，再从 Angstrom 转成 nm。
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],  # [B,L,L,A,A,3]；residue i 的每个原子与 residue j 的每个原子两两相减。
            dim=-1, ord=2,  # 对 xyz 维求 L2 范数，得到距离。
        )).reshape(N, L, L, -1)  # [B,L,L,A*A]；把原子对矩阵展平为一维通道。
        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # [B,L,L,A*A]；针对不同 aa-pair 学到的距离核宽度参数，softplus 保证为正。
        d_gauss = torch.exp(-1 * c * d**2)  # [B,L,L,A*A]；每个原子对距离经过高斯核变换，形成平滑距离基函数。
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)  # [B,L,L,A*A]；只有两个原子都存在时，该原子对距离才有效。
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)  # [B,L,L,F]；把展平的原子对距离特征投影到边特征空间。
        if pair_structure_mask is not None:  # 若某些 residue 结构不可见，则对应 pair 的几何距离特征要整体清零。
            # Avoid data leakage at training time  # 防止待生成区域直接看到 GT 原子距离。
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]  # [B,L,L,F]；不可见 pair 清零。

        # Orientations  # 第 4 路边特征：残基对方向/二面角。
        dihed = pairwise_dihedrals(pos_atoms)  # [B,L,L,2]；每个 residue pair 提取 2 个方向相关二面角。
        feat_dihed = self.dihedral_embed(dihed)  # [B,L,L,C_dihed]；角度编码成高维方向特征。
        if pair_structure_mask is not None:  # 若结构不可见，方向特征同样要屏蔽。
            # Avoid data leakage at training time  # 避免把 GT pair orientation 泄漏给模型。
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]  # [B,L,L,C_dihed]；不可见 pair 清零。

        # # trans embed  # 旧的 distogram/trans 特征尝试；当前实现未启用，保留作参考。
        # dist_feats = calc_distogram(  # 若启用，会把 CA 平移距离离散成 bin 特征。
        #     trans, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)  # 预期输出 [B,L,L,num_bins]。
        # if sc_trans == None:  # 若 side-chain translation 缺失，则用 0 填。
        #     sc_trans = torch.zeros_like(trans)  # 与 trans 形状相同。
        # sc_feats = calc_distogram(  # side-chain 平移的 distogram。
        #     sc_trans, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)  # 预期输出 [B,L,L,num_bins]。

        feat_list = [feat_aapair, feat_relpos, feat_dist, feat_dihed]  # 前 4 路是所有模式都存在的边特征。
        if self.position_type == "cyclic":  # 只有 cyclic 模式才附加环肽链 Cβ 距离先验，linear 模式保持原样。
            feat_cyclic_prior = self._cyclic_prior_features(  # [B,L,L,128]；只在肽链相邻边/首尾边非 0。
                chain_nb=chain_nb,
                mask_residue=mask_residue,
                cyclic_mask=cyclic_mask,
                cyclic_type_id=cyclic_type_id,
                dtype=feat_aapair.dtype,
            )
            feat_list.append(feat_cyclic_prior)  # 把固定先验原始 128 维直接拼进去；只有 special edge 非 0。

        # All  # 把所有边特征拼成统一输入，再压成最终 edge embedding。
        feat_all = torch.cat(feat_list, dim=-1)  # [B,L,L,Cin]；按最后一维拼接 aa-pair[64] / idx-diff[64] / distance[64] / orientation[26] / optional cyclic_prior[128]。
        feat_all = self.out_mlp(feat_all)  # [B,L,L,F]；融合后得到最终边特征。
        feat_all = feat_all * mask_pair[:, :, :, None]  # [B,L,L,F]；无效 residue pair（任一侧没有 CA）整体清零。

        return feat_all  # 返回边特征 [B,L,L,F]；后续作为 trunk/GAEncoder 的 pair 输入。
