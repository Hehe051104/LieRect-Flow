import torch
from typing import Tuple
from core.lie_algebra import so3_exp, so3_log


# 在三维旋转空间里“绝对均匀”地随机撒点 R
def sample_so3_uniform(
    batch_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    在 SO(3) 流形上均匀采样随机旋转矩阵 (利用正态分布四元数归一化，保证空间各向同性)
    Args:
        batch_shape: 前导批次维度，例如 (B,) 或 (B, T)
        device: 张量所在设备 (cpu / cuda)
        dtype: 浮点精度
    Returns:
        (*batch_shape, 3, 3) 合法的正交旋转矩阵
    """
    # 采样 4 维标准正态分布并归一化，等价于在 4 维超球面 S^3 上均匀采样单位四元数
    q = torch.randn(*batch_shape, 4, device=device, dtype=dtype)
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # 用四元数生成随机旋转
    # w 负责记录到底转了多少度(θ)      角度
    # x, y ,z 负责指明转轴 n 朝向哪里    方向

    # 四元数转 3x3 旋转矩阵
    R = torch.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
    ], dim=-1).reshape(*batch_shape, 3, 3)

    return R


# 打包生成 t=0 的初始噪声位姿 pose = (R , p)
def sample_se3_noise(
    batch_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    trans_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    采样流匹配 t=0 时刻的初始噪声位姿 (R_0, p_0)
    Args:
        batch_shape: 前导批次维度
        device: 设备
        dtype: 数据类型
        trans_scale: 平移噪声的标准差缩放系数（单位：米）
    Returns:
        R_0: (*batch_shape, 3, 3) 随机均匀旋转
        p_0: (*batch_shape, 3) 高斯随机位置
    """
    R_0 = sample_so3_uniform(batch_shape, device=device, dtype=dtype)
    p_0 = torch.randn(*batch_shape, 3, device=device, dtype=dtype) * trans_scale
    return R_0, p_0


# 训练时的“出题器”   算出起点与终点之间任意时刻t的信息 p
def se3_geodesic_interpolate(
    R_0: torch.Tensor,
    p_0: torch.Tensor,
    R_1: torch.Tensor,
    p_1: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SE(3) 解耦测地线插值（训练阶段出题器）：
    给定起点位姿 (R_0, p_0)、终点位姿 (R_1, p_1) 和时间 t，
    计算中间状态 (R_t, p_t) 以及恒定目标速度标签 (v_target, omega_target)。
    Args:
        R_0, R_1: (..., 3, 3) 起点与终点旋转矩阵
        p_0, p_1: (..., 3)    起点与终点三维坐标
        t:        (..., 1)    时间比例标量，范围 [0, 1]
    Returns:
        R_t:          (..., 3, 3) t 时刻的插值旋转矩阵
        p_t:          (..., 3)    t 时刻的插值三维位置
        v_target:     (..., 3)    标准平移线速度标签 (p_1 - p_0)
        omega_target: (..., 3)    标准自身坐标系角速度标签 log(R_0^T R_1)
    """
    # 确保 t 的维度至少能与 (..., 3) 广播对齐
    if t.ndim < p_0.ndim:
        t = t.unsqueeze(-1)  # (..., 1)

    # 1. 平移部分：欧氏空间线性插值
    v_target = p_1 - p_0
    p_t = p_0 + t * v_target

    # 2. 旋转部分：SO(3) 测地线插值
    # 计算相对旋转差量 delta_R = R_0^T * R_1
    delta_R = torch.matmul(R_0.transpose(-1, -2), R_1)
    # 映射到平坦切空间，得到恒定角速度标签
    omega_target = so3_log(delta_R)
    # 按时间比例 t 缩放角速度并映射回旋转矩阵，右乘叠加到起点 R_0
    R_t = torch.matmul(R_0, so3_exp(t * omega_target))

    return R_t, p_t, v_target, omega_target


# 推理时的“步进执行器”  执行网络输出的动作
def se3_euler_step(
    R_t: torch.Tensor,
    p_t: torch.Tensor,
    v_pred: torch.Tensor,
    omega_pred: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    流形欧拉积分单步推进（推理阶段执行器）：
    沿网络预测的速度场 (v_pred, omega_pred) 前进一小步 dt。
    Args:
        R_t, p_t:           当前时刻的旋转矩阵 (..., 3, 3) 与位置 (..., 3)
        v_pred, omega_pred: 网络预测的线速度 (..., 3) 与角速度 (..., 3)
        dt:                 积分时间步长 (例如 1.0 / num_steps)  分几步走完
    Returns:
        R_next, p_next:     推进后的新旋转矩阵与新位置
    """
    p_next = p_t + dt * v_pred
    R_next = torch.matmul(R_t, so3_exp(dt * omega_pred))
    return R_next, p_next



def sample_local_se3_noise(
    R_ref: torch.Tensor,
    p_ref: torch.Tensor,
    trans_scale: float = 0.25,
    rot_scale: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """围绕参考位姿 (R_ref, p_ref) 添加局部扰动，生成安全的起点位姿 (R_0, p_0)"""
    omega_noise = torch.randn_like(p_ref) * rot_scale
    R_0 = torch.matmul(R_ref, so3_exp(omega_noise))
    p_0 = p_ref + torch.randn_like(p_ref) * trans_scale
    return R_0, p_0