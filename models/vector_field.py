# 神经网络天生只会做平直空间的加减乘除（Linear + SiLU），
# 它根本不会算弯曲流形上的指数和对数。
# 所以我们把第一步（lie_algebra.py）和第二步（manifold.py）放在神经网络的“前面”和“后面”，
# 充当弯曲空间与平直空间之间的“翻译”。

# 模型只管输出角速度和线速度。剩下的交给前两步兜底
import math
import torch
import torch.nn as nn
from typing import Tuple

# 两个类干的事情非常纯粹：把“现在几点了（时间 t）”和“夹爪眼里看到的水杯长什么样（点云）”翻译成 6 个速度数字

class SinusoidalTimeEmbedding(nn.Module):
    """
    将标量时间 t in [0, 1] 编码为高维正余弦频率特征（标准 Flow Matching / Diffusion 时间编码）
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim   # 时间编码的维度

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, 1) 或 (B,) 时间标量
        Returns:
            (B, dim) 时间特征向量
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2  # 一半给sin，一半给cos
        # arange生成等差数列(从大到小)，生成的。一组从高频到低频
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t * freqs.unsqueeze(0) # unsqueeze在最前面拓展维度，方便与t相乘
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # 在最后一维上将sin与cos拼接起来


class SE3EquivariantVectorField(nn.Module):
    """
    SE(3) 等变流匹配速度场网络：
    利用夹爪局部坐标系规范化，保证对任意三维刚体变换严格等变。
    """
    def __init__(
        self,
        point_dim: int = 3,  # 点云的三维的 x y z
        hidden_dim: int = 256,
        time_dim: int = 64,   # 时间编码的维度
    ):
        super().__init__()
        # 时间理解区
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )

        # 视觉看图区
        # 局部点云特征提取器 (PointNet 骨干：对每个点独立提特征再做全局池化)
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 融合几何特征与时间特征，预测局部 6 维速度 (v_local, omega_local)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 前 3 维局部线速度，后 3 维局部角速度
        )

    def forward(
        self,
        R_t: torch.Tensor,
        p_t: torch.Tensor,
        points_world: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            R_t:          (B, 3, 3) 当前夹爪旋转矩阵
            p_t:          (B, 3)    当前夹爪三维坐标
            points_world: (B, N, 3) 世界坐标系下的物体表面点云  B批次 N每个场景几个点
            t:            (B, 1)    当前流匹配时间步 [0, 1]
        Returns:
            v_world:      (B, 3) 世界坐标系下的预测平移线速度（满足 SO(3) 等变性）
            omega_body:   (B, 3) 夹爪局部坐标系下的预测旋转角速度（满足 SE(3) 不变性）
        """
        # 1. 核心等变变换：把世界点云投影到当前夹爪局部坐标系 P_local = (P_world - p_t) @ R_t
        # p_t.unsqueeze(1) 形状为 (B, 1, 3)，自动广播减去夹爪中心坐标
        points_local = torch.matmul(points_world - p_t.unsqueeze(1), R_t)  # (B, N, 3)

        # 2. 提取局部几何特征并做最大池化 + 平均池化聚合全局物体形状
        feat_pts = self.point_encoder(points_local)  # (B, N, hidden_dim)
        feat_global = feat_pts.max(dim=1)[0] + feat_pts.mean(dim=1)  # (B, hidden_dim) 沿着N（一个一个点），对各个特征拿最大值和平均值 把N个点压缩成一个点

        # 3. 编码时间步并拼接特征 把“形状”和“时间”拼在一起
        feat_time = self.time_embed(t)  # (B, hidden_dim)
        feat_cat = torch.cat([feat_global, feat_time], dim=-1)  # (B, hidden_dim * 2)

        # 4. 预测局部速度旋量 [v_local, omega_local]
        twist_local = self.velocity_head(feat_cat)  # (B, 6)
        v_local = twist_local[:, :3]      # (B, 3)
        omega_body = twist_local[:, 3:]   # (B, 3)

        # 5. 将局部线速度旋转回世界坐标系：v_world = R_t @ v_local
        # RT 是 3,3 ，vlocal 是 3。先把后面的变成 3,1 ，然后两式相乘得 3,1 ，再把最后的1去掉
        v_world = torch.matmul(R_t, v_local.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        return v_world, omega_body