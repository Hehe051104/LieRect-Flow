import torch
from typing import Tuple
from core.manifold import sample_se3_noise


def generate_asymmetric_mug_batch(
    batch_size: int,
    num_points: int = 512,
    device: torch.device = torch.device("cpu"),
    random_pose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成非对称带柄水杯的确定性表面点云及唯一对应的标准抓取位姿 (R_1, p_1)。
    去掉随机逐帧抖动，避免在 t -> 1 时被 1/(1-t) 放大成虚假角速度标签。
    Args:
        batch_size:  一次生成的场景数量 B
        num_points:  固定为 512（384 个杯身点 + 128 个杯柄点）
        device:      张量计算设备 (cpu / cuda)
        random_pose: False 时水杯固定在世界原点（用于训练）；
                     True 时水杯在三维空间随机旋转平移（用于测试等变泛化）
    Returns:
        points_world: (B, 512, 3) 世界坐标系下的水杯表面点云
        R_1:          (B, 3, 3)   专家标准抓取旋转矩阵
        p_1:          (B, 3)      专家标准抓取三维位置
    """
    # 1. 构建水杯杯身圆柱表面规则网格 (24 个圆周角度 × 16 层高度 = 384 个点)
    # 半径 0.04m（直径 8cm），高度 0.10m（-0.05m 到 +0.05m）
    theta = torch.linspace(0.0, 2.0 * torch.pi, 25, device=device)[:-1]  # (24,)
    z_body = torch.linspace(-0.05, 0.05, 16, device=device)             # (16,)
    grid_theta, grid_z = torch.meshgrid(theta, z_body, indexing="ij")
    pts_body = torch.stack([
        0.04 * torch.cos(grid_theta.reshape(-1)),
        0.04 * torch.sin(grid_theta.reshape(-1)),
        grid_z.reshape(-1),
    ], dim=-1)  # (384, 3)

    # 2. 构建位于 +X 侧的杯柄表面规则网格 (8 × 2 × 8 = 128 个点，总计 512 点)
    # 杯柄只在 +X 侧伸出，打破圆柱体的旋转对称性，保证抓取朝向唯一
    hx = torch.linspace(0.04, 0.07, 8, device=device)   # 从杯壁 4cm 伸出到 7cm
    hy = torch.tensor([-0.008, 0.008], device=device)   # 杯柄厚度 1.6cm
    hz = torch.linspace(-0.03, 0.03, 8, device=device)  # 杯柄高度 6cm
    gx, gy, gz = torch.meshgrid(hx, hy, hz, indexing="ij")
    pts_handle = torch.stack([
        gx.reshape(-1),
        gy.reshape(-1),
        gz.reshape(-1),
    ], dim=-1)  # (128, 3)

    # 拼接完整水杯表面点云并复制到整个 Batch（干净无抖动的刚体几何）
    base_points = torch.cat([pts_body, pts_handle], dim=0)  # (512, 3)
    points_obj = base_points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  # (B, 512, 3)

    # 3. 定义物体局部坐标系下的专家标准抓取姿态 (R_grasp_obj, p_grasp_obj)
    # 抓取中心位于杯柄外侧 (+0.065m, 0, 0)，夹爪局部 X 轴朝向 -X（握向杯柄），Z 轴竖直朝上
    p_grasp_obj = torch.tensor([0.065, 0.0, 0.0], device=device)
    R_grasp_obj = torch.tensor([
        [-1.0,  0.0, 0.0],
        [ 0.0, -1.0, 0.0],
        [ 0.0,  0.0, 1.0],
    ], device=device)

    # 4. 变换到世界坐标系
    if random_pose:
        # 推理测试阶段：对整个水杯施加全空间任意 3D 旋转和平移，检验零样本空间泛化能力
        R_obj, p_obj = sample_se3_noise((batch_size,), device=device, trans_scale=0.5)
        points_world = torch.matmul(points_obj, R_obj.transpose(-1, -2)) + p_obj.unsqueeze(1)
        R_1 = torch.matmul(R_obj, R_grasp_obj)
        p_1 = torch.matmul(R_obj, p_grasp_obj) + p_obj
    else:
        # 训练阶段：水杯固定在世界原点不动
        points_world = points_obj
        R_1 = R_grasp_obj.expand(batch_size, 3, 3)
        p_1 = p_grasp_obj.expand(batch_size, 3)

    return points_world, R_1, p_1