import torch
from core.manifold import sample_se3_noise
from models.vector_field import SE3EquivariantVectorField

# 验证SE(3)网络是否是等变的
# 线速度严格跟随旋转 v世界 = R * v局部
# 自身角速度严格保持不变 omega世界 = omega局部

def test_se3_equivariance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SE(3) equivariance on: {device}")

    torch.manual_seed(42)
    batch_size = 64
    num_points = 128

    model = SE3EquivariantVectorField(hidden_dim=256).to(device)
    model.eval()

    # 1. 构造原始场景：随机夹爪位姿 (R_t, p_t)、随机物体点云 points、随机时间 t
    R_t, p_t = sample_se3_noise((batch_size,), device=device)
    points = torch.randn(batch_size, num_points, 3, device=device)
    t = torch.rand(batch_size, 1, device=device)

    with torch.no_grad():
        v_orig, omega_orig = model(R_t, p_t, points, t)

    # 2. 对整个三维世界施加一个任意的 SE(3) 刚体变换 g = (R_g, p_g)
    R_g, p_g = sample_se3_noise((batch_size,), device=device, trans_scale=5.0)

    # 变换夹爪位姿：R'_t = R_g @ R_t,  p'_t = R_g @ p_t + p_g
    R_t_trans = torch.matmul(R_g, R_t)
    p_t_trans = torch.matmul(R_g, p_t.unsqueeze(-1)).squeeze(-1) + p_g

    # 变换物体点云：P' = P @ R_g^T + p_g
    points_trans = torch.matmul(points, R_g.transpose(-1, -2)) + p_g.unsqueeze(1)

    with torch.no_grad():
        v_trans, omega_trans = model(R_t_trans, p_t_trans, points_trans, t)

    # 3. 验证等变性与不变性误差
    v_expected = torch.matmul(R_g, v_orig.unsqueeze(-1)).squeeze(-1)
    v_err = (v_trans - v_expected).abs().max().item()
    omega_err = (omega_trans - omega_orig).abs().max().item()

    print(f"Linear velocity equivariance max error:  {v_err:.2e}")
    print(f"Angular velocity invariance max error:   {omega_err:.2e}")

    assert v_err < 1e-5, f"线速度不满足 SO(3) 等变性: {v_err}"
    assert omega_err < 1e-5, f"角速度不满足 SE(3) 不变性: {omega_err}"

    print("✅ SE(3) 严格等变性测试全部通过（误差达到浮点精度极限）！")


if __name__ == "__main__":
    test_se3_equivariance()