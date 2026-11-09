import torch

# 考虑torch的批处理，所有变量都可能是在batch中的，所以要用...

def hat(omega: torch.Tensor) -> torch.Tensor:   # 加帽子->反对称矩阵
    """
    将 3 维旋转向量映射为反对称矩阵 (Hat Operator)
    Args:
        omega: (..., 3) 形状的张量 ...意味有batch
    Returns:
        (..., 3, 3) 形状的反对称矩阵
        0,    -w3,     w2,
        w3,    0,     -w1,
        -w2,   w1,     0
    """
    w1, w2, w3 = omega[..., 0], omega[..., 1], omega[..., 2]  # (...,  )
    zeros = torch.zeros_like(w1)
    
    # 按照三阶外积行列式展开严格排布
    hat_matrix = torch.stack([
        zeros, -w3,   w2,
        w3,    zeros, -w1,
        -w2,   w1,    zeros
    ], dim=-1).reshape(omega.shape[:-1] + (3, 3)) #dim=-1 让9个数字再最后一个维度堆叠起来，再reshape成将最后一个维度3变成3✖️3维度的矩阵
    
    return hat_matrix


def vee(omega_hat: torch.Tensor) -> torch.Tensor:  # 去帽子
    """
    从反对称矩阵中提取 3 维旋转向量 (Vee Operator)
    Args:
        omega_hat: (..., 3, 3) 形状的反对称矩阵
    Returns:
        (..., 3) 形状的向量
    """
    w1 = omega_hat[..., 2, 1]
    w2 = omega_hat[..., 0, 2]
    w3 = omega_hat[..., 1, 0]
    return torch.stack([w1, w2, w3], dim=-1)


def so3_exp(omega: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # 刚开始，我们只知道Omega，即角速度
    """
    罗德里格斯公式：李代数 so(3) -> 李群 SO(3)
    Args:
        omega: (..., 3) 形状的李代数向量 (转轴 * 角度)
        eps: 小角度截断阈值，避免除以零
    Returns:
        (..., 3, 3) 合法的旋转矩阵 = R 旋转矩阵
    """
    # 要将罗德里格斯公式里的n hat 替换成 omega_hat / θ
    theta = torch.norm(omega, dim=-1, keepdim=True)  # (..., 1) 总角度
    omega_hat = hat(omega)                           # (..., 3, 3)
    omega_hat_sq = torch.matmul(omega_hat, omega_hat) # (..., 3, 3)
    
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device) # 创建单位矩阵
    
    # 针对 theta -> 0 区域采用泰勒级数展开做数值保护：
    # sin(theta)/theta ≈ 1 - theta^2 / 6
    # (1 - cos(theta))/theta^2 ≈ 0.5 - theta^2 / 24
    theta_sq = theta ** 2
    small_angle = theta < eps
    
    c1 = torch.where(small_angle, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
    c2 = torch.where(small_angle, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + eps))
    
    c1 = c1.unsqueeze(-1)  # (..., 1, 1)  转到与omega_hat相同维数上才能相乘
    c2 = c2.unsqueeze(-1)  # (..., 1, 1)
    
    R = eye + c1 * omega_hat + c2 * omega_hat_sq
    return R


def so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    对数映射：李群 SO(3) -> 李代数 so(3) 从R反推数据
    Args:
        R: (..., 3, 3) 形状的合法旋转矩阵
    Returns:
        (..., 3) 形状的李代数向量 (转轴 * 角度)   omega(角速度) = θ * n
    """
    # 1. 求迹 tr(R) 并限制范围在 [-1, 3]，防止浮点误差导致 acos 越界返回 nan
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]    # (..., )
    cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)  # (..., )
    theta = torch.acos(cos_theta).unsqueeze(-1)  # (..., 1)
    
    # 2. 矩阵减转置提取非对称分量：R - R^T = 2sin(θ)* n hat
    diff = R - R.transpose(-1, -2)
    omega_hat_unscaled = diff / 2.0   # (..., 3 , 3)    sin(θ)* n hat
    omega_unscaled = vee(omega_hat_unscaled)     # (..., 3) sin(θ)* n
    
    # 3. 针对 theta -> 0 区域的数值保护：theta / sin(theta) ≈ 1 + theta^2 / 6
    sin_theta = torch.sin(theta)
    small_angle = theta < eps

    # scale = θ/sin(θ)
    scale = torch.where(small_angle, 1.0 + (theta ** 2) / 6.0, theta / (sin_theta + eps))
    return omega_unscaled * scale  # sin(θ)* n * θ/sin(θ) = θ * n = omega