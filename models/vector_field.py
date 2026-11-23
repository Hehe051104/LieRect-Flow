import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """
    将标量时间 t in [0, 1] 编码为高维正余弦频率特征（标准 Flow Matching 时间编码）
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResMLPBlock(nn.Module):
    """
    带残差连接的隐藏层模块：让梯度能够无损穿透深层网络，提升亚毫米/亚度级回归精度
    """
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SE3EquivariantVectorField(nn.Module):
    """
    SE(3) 等变流匹配速度场网络（高精度注意力版）：
    1. 通过夹爪局部坐标系规范化，保证对任意三维刚体变换严格等变；
    2. 利用注意力池化自动聚焦杯柄等关键非对称几何特征，消除对称杯身的干扰；
    3. 纯直接输出世界线速度 v_world 与自身角速度 omega_body。
    """
    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ):
        super().__init__()
        # 1. 时间编码网络
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )

        # 2. 逐点局部几何特征提取器
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

        # 3. 几何注意力打分器：自动识别哪些点属于关键特征（如杯柄），赋予高权重
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),  # 为每个点输出一个标量重要性得分
        )

        # 4. 融合全局几何特征与时间特征，通过残差网络直接预测 6 维物理速度
        self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.res_blocks = nn.Sequential(
            ResMLPBlock(hidden_dim),
            ResMLPBlock(hidden_dim),
        )
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 前 3 维局部线速度，后 3 维自身角速度
        )

    def forward(
        self,
        R_t: torch.Tensor,
        p_t: torch.Tensor,
        points_world: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 第一步：核心等变变换，把世界点云投影到夹爪局部坐标系并缩放至分米量级 (* 10.0)
        # points_local 严格满足 SE(3) 不变性
        points_local = torch.matmul(points_world - p_t.unsqueeze(1), R_t) * 10.0  # (B, N, 3)

        # 第二步：提取逐点特征，并结合“注意力加权聚合”与“最大池化”提取高分辨率几何表征
        feat_pts = self.point_encoder(points_local)  # (B, N, hidden_dim)

        # 计算每个点的重要性权重 (Softmax 归一化)
        weights = torch.softmax(self.attn_score(feat_pts), dim=1)          # (B, N, 1)
        feat_attn = torch.sum(feat_pts * weights, dim=1)                   # (B, hidden_dim)
        feat_max = feat_pts.max(dim=1)[0]                                  # (B, hidden_dim)

        # 第三步：拼接几何特征与时间特征，送入残差速度预测头
        feat_time = self.time_embed(t)                                     # (B, hidden_dim)
        feat_cat = torch.cat([feat_attn, feat_max, feat_time], dim=-1)     # (B, hidden_dim * 3)

        h = self.input_proj(feat_cat)
        h = self.res_blocks(h)
        twist_local = self.out_head(h)                                     # (B, 6)

        v_local = twist_local[:, :3]      # (B, 3) 夹爪局部坐标系下的线速度
        omega_body = twist_local[:, 3:]   # (B, 3) 夹爪自身坐标系下的角速度

        # 第四步：将局部线速度旋转回世界坐标系（保证 SO(3) 严格等变性）
        v_world = torch.matmul(R_t, v_local.unsqueeze(-1)).squeeze(-1)     # (B, 3)

        return v_world, omega_body