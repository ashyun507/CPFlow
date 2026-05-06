import torch  # PyTorch 张量库；本文件里的基础层和编码函数都基于它实现。
import torch.nn as nn  # 神经网络模块定义。
import torch.nn.functional as F  # one_hot、激活函数等逐元素函数接口。


def mask_zero(mask, value):  # 通用掩码函数：mask=True 的位置保留原值，mask=False 的位置清零。
    return torch.where(mask, value, torch.zeros_like(value))  # 输出形状与 value 相同；常用于把无效位置的特征/损失抹为 0。


def clampped_one_hot(x, num_classes):  # 把整数标签张量安全地转成 one-hot；越界标签会先被 clamp，再用原始合法性 mask 清零。
    mask = (x >= 0) & (x < num_classes)  # 与 x 同形状，例如 [B, L]；True 表示该位置标签在合法类别范围内。
    x = x.clamp(min=0, max=num_classes - 1)  # 与 x 同形状；把越界值硬截到 [0, C-1]，避免 one_hot 报错。
    y = F.one_hot(x, num_classes) * mask[..., None]  # [..., C]；先做 one-hot，再把非法位置整条 one-hot 向量清零。
    return y  # 返回 one-hot 编码；典型形状如 [B, L, C]。


def sample_from(c):  # 从离散分布 c 中逐位置采样类别 ID。
    """sample from c"""  # c 通常是类别概率或非负权重张量。
    N, L, K = c.size()  # N=batch 大小，L=序列长度/残基数，K=类别数。
    c = c.view(N * L, K) + 1e-8  # [N*L, K]；把 batch 和长度维展平，且加极小值避免某行全 0 导致 multinomial 报错。
    x = torch.multinomial(c, 1).view(N, L)  # [N, L]；每个位置采 1 个类别，返回离散类别索引。
    return x  # 输出类别 ID 张量；后续常用于从 simplex/概率分布采样 aa token。


class DistanceToBins(nn.Module):  # 距离离散化/软编码层：把标量距离映射成 num_bins 维离散或软分箱表示。

    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot=False):  # dist_min/dist_max 为距离范围；num_bins 为输出通道数；use_onehot 控制硬分箱还是高斯软分箱。
        super().__init__()  # 初始化 nn.Module。
        self.dist_min = dist_min  # 距离下界。
        self.dist_max = dist_max  # 距离上界；超过它的距离会落到 overflow bin。
        self.num_bins = num_bins  # 输出 bin 数。
        self.use_onehot = use_onehot  # True=最近 bin 的 one-hot；False=高斯软编码 + overflow 符号位。

        if use_onehot:  # 硬分箱模式。
            offset = torch.linspace(dist_min, dist_max, self.num_bins)  # [num_bins]；每个 bin 的中心/参考位置。
        else:  # 软分箱模式。
            offset = torch.linspace(dist_min, dist_max, self.num_bins - 1)  # [num_bins-1]；前 num_bins-1 个位置用于高斯响应，最后 1 个通道留给 overflow。
            self.coeff = -0.5 / ((offset[1] - offset[0]) * 0.2).item() ** 2  # 标量系数；控制高斯核宽度，`*0.2` 让响应更尖一些、避免过度模糊。
        self.register_buffer('offset', offset)  # 把 offset 注册成 buffer；随模型搬设备/保存，但不是可学习参数。

    @property  # 只读属性：外部可通过 layer.out_channels 查询输出通道数。
    def out_channels(self):
        return self.num_bins  # 距离编码后的最后一维通道数。

    def forward(self, dist, dim, normalize=True):  # 输入距离张量，在指定维度 dim 上把单通道距离扩成 num_bins 维编码。
        """
        Args:
            dist:   (N, *, 1, *)  # dist 在 dim 对应的位置必须是长度为 1 的“距离通道”；其余维可以是任意 batch/空间维。
        Returns:
            (N, *, num_bins, *)  # 把 dim 位置从 1 扩成 num_bins 后得到的编码张量。
        """
        assert dist.size()[dim] == 1  # 防御性检查：要求输入在指定 dim 上确实只有 1 个距离标量。
        offset_shape = [1] * len(dist.size())  # 先构造一个全 1 的 shape 模板，长度等于 dist 的维数。
        offset_shape[dim] = -1  # 在目标维度放 -1，使 offset reshape 后正好沿 dim 展开成 num_bins 或 num_bins-1。

        if self.use_onehot:  # 硬分箱模式。
            diff = torch.abs(dist - self.offset.view(*offset_shape))  # 与输出同形状 [N, *, num_bins, *]；每个距离到各 bin 参考位置的绝对差。
            bin_idx = torch.argmin(diff, dim=dim, keepdim=True)  # [N, *, 1, *]；找到最近的 bin 索引。
            y = torch.zeros_like(diff).scatter_(dim=dim, index=bin_idx, value=1.0)  # [N, *, num_bins, *]；把最近 bin 置 1，其余为 0。
        else:  # 软分箱模式。
            overflow_symb = (dist >= self.dist_max).float()  # [N, *, 1, *]；超出 dist_max 的距离会激活 overflow 通道。
            y = dist - self.offset.view(*offset_shape)  # [N, *, num_bins-1, *]；距离与各个 bin 参考位置的偏差。
            y = torch.exp(self.coeff * torch.pow(y, 2))  # [N, *, num_bins-1, *]；高斯响应值，越接近某个 offset 响应越高。
            y = torch.cat([y, overflow_symb], dim=dim)  # [N, *, num_bins, *]；把 overflow 符号位拼到最后一个 bin。
            if normalize:  # 是否在 bin 维归一化成“近似分布”。
                y = y / y.sum(dim=dim, keepdim=True)  # [N, *, num_bins, *]；保证沿 dim 的和为 1。

        return y  # 返回距离编码；常用于把几何距离变成更容易被 MLP/注意力使用的多通道特征。


class PositionalEncoding(nn.Module):  # 通用实值位置编码层：把标量/向量 x 做多频率 sin/cos 展开。
    
    def __init__(self, num_funcs=6):  # num_funcs=频率数量 f；输出维度会扩成原始维度的 (2f+1) 倍。
        super().__init__()  # 初始化 nn.Module。
        self.num_funcs = num_funcs  # 保存频率个数。
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs - 1, num_funcs))  # [f]；频率带，按 2^k 递增。
    
    def get_out_dim(self, in_dim):  # 给定输入最后一维 in_dim，返回编码后的输出维度。
        return in_dim * (2 * self.num_funcs + 1)  # 每个输入维展开成 [原值, sin(f*x), cos(f*x)] 共 2f+1 个通道。

    def forward(self, x):  # 对最后一维做位置编码。
        """
        Args:
            x:  (..., d).  # 输入张量；最后一维 d 是需要编码的实值维度，其余维保持不变。
        """
        shape = list(x.shape[:-1]) + [-1]  # 目标输出 shape；最后一维展平为 d*(2f+1)。
        x = x.unsqueeze(-1)  # (..., d, 1)；在最后加一个频率维，便于和 freq_bands 广播相乘。
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)  # (..., d, 2f+1)；每个输入维拼上多频率 sin/cos。
        code = code.reshape(shape)  # (..., d*(2f+1))；把倒数两维展平，供后续 MLP/注意力直接使用。
        return code  # 返回编码后的张量；常用于标量几何量或位置量的多频表示。


class AngularEncoding(nn.Module):  # 角度专用编码层：同时用整数频率和倒数频率展开角度，更适合周期变量。

    def __init__(self, num_funcs=3):  # num_funcs=f；输出每个输入维会扩成 1 + 4f 个通道。
        super().__init__()  # 初始化 nn.Module。
        self.num_funcs = num_funcs  # 保存频率数量。
        self.register_buffer('freq_bands', torch.FloatTensor(  # [2f]；前半部分是 1,2,...,f，后半部分是 1,1/2,...,1/f。
            [i + 1 for i in range(num_funcs)] + [1. / (i + 1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):  # 查询角度编码后的输出维度。
        return in_dim * (1 + 2 * 2 * self.num_funcs)  # 每个输入维 -> 原值 1 个 + sin/cos 各 2f 个，所以总共 1+4f。

    def forward(self, x):  # 对角度张量的最后一维做编码。
        """
        Args:
            x:  (..., d).  # 输入角度张量；最后一维 d 是角度数，其余维可为 batch/残基/pair 等。
        """
        shape = list(x.shape[:-1]) + [-1]  # 目标输出 shape；最后一维展平为 d*(1+4f)。
        x = x.unsqueeze(-1)  # (..., d, 1)；加频率维准备与 freq_bands 广播相乘。
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)  # (..., d, 1+4f)；把原角度值与多频率 sin/cos 拼起来。
        code = code.reshape(shape)  # (..., d*(1+4f))；展平最后两维，输出给下游层。
        return code  # 返回角度编码；当前项目里常用于 backbone dihedral / pairwise dihedral。


class LayerNorm(nn.Module):  # 手写 LayerNorm：对最后一维做归一化，可选 learnable gamma/beta。

    def __init__(self,  # 初始化 LayerNorm。
                 normal_shape,  # 归一化的目标 shape；通常传最后一维大小或一个 shape tuple。
                 gamma=True,  # 是否学习缩放参数 gamma。
                 beta=True,  # 是否学习平移参数 beta。
                 epsilon=1e-10):  # 防止除 0 的数值稳定项。
        """Layer normalization layer  # 模块说明：对最后一维做均值方差归一化。
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)  # 参考论文。
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.  # 归一化维度。
        :param gamma: Add a scale parameter if it is True.  # 是否加可学习缩放。
        :param beta: Add an offset parameter if it is True.  # 是否加可学习偏置。
        :param epsilon: Epsilon for calculating variance.  # 方差项稳定常数。
        """
        super().__init__()  # 初始化 nn.Module。
        if isinstance(normal_shape, int):  # 如果传的是单个整数。
            normal_shape = (normal_shape,)  # 统一转成 tuple，表示只对最后一维做归一化。
        else:  # 如果传的是一个 shape。
            normal_shape = (normal_shape[-1],)  # 只取最后一维，因为当前实现只支持最后一维归一化。
        self.normal_shape = torch.Size(normal_shape)  # 保存标准化 shape。
        self.epsilon = epsilon  # 保存数值稳定项。
        if gamma:  # 若启用缩放参数。
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))  # [C]；每个通道一个 learnable scale。
        else:  # 否则不创建 gamma。
            self.register_parameter('gamma', None)  # 显式注册空参数。
        if beta:  # 若启用偏移参数。
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))  # [C]；每个通道一个 learnable bias。
        else:  # 否则不创建 beta。
            self.register_parameter('beta', None)  # 显式注册空参数。
        self.reset_parameters()  # 初始化 gamma/beta。

    def reset_parameters(self):  # 参数初始化：gamma=1，beta=0。
        if self.gamma is not None:  # 若 gamma 存在。
            self.gamma.data.fill_(1)  # 让初始缩放为恒等。
        if self.beta is not None:  # 若 beta 存在。
            self.beta.data.zero_()  # 让初始偏置为 0。

    def forward(self, x):  # 对输入最后一维做 LayerNorm。
        mean = x.mean(dim=-1, keepdim=True)  # 与 x 同形状但最后一维为 1；每个样本/位置的通道均值。
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # 与 mean 同形状；每个样本/位置的通道方差。
        std = (var + self.epsilon).sqrt()  # 与 mean 同形状；标准差。
        y = (x - mean) / std  # 与 x 同形状；归一化后的结果，最后一维均值约为 0、方差约为 1。
        if self.gamma is not None:  # 若启用缩放参数。
            y *= self.gamma  # 广播乘；每个通道各自缩放。
        if self.beta is not None:  # 若启用偏移参数。
            y += self.beta  # 广播加；每个通道各自平移。
        return y  # 返回归一化后的张量；形状与 x 完全相同。

    def extra_repr(self):  # 控制 print(module) 时的附加字符串。
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(  # 以可读文本展示 LayerNorm 的关键超参数。
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,  # 分别显示归一化维度、是否有 gamma/beta、epsilon。
        )
