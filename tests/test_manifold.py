import torch
from core.manifold import (
    sample_se3_noise,
    se3_geodesic_interpolate,
    se3_euler_step,
)


def test_se3_manifold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SE(3) manifold on: {device}")

    batch_size = 512
    # 随机生成 512 对起点噪声位姿和终点目标位姿
    R_0, p_0 = sample_se3_noise((batch_size,), device=device)
    R_1, p_1 = sample_se3_noise((batch_size,), device=device)

    # 1. 验证 t=0 起点对齐
    t0 = torch.zeros(batch_size, 1, device=device)
    R_t0, p_t0, _, _ = se3_geodesic_interpolate(R_0, p_0, R_1, p_1, t0)
    assert torch.allclose(R_t0, R_0, atol=1e-5), "t=0 旋转未对齐起点！"
    assert torch.allclose(p_t0, p_0, atol=1e-5), "t=0 位置未对齐起点！"

    # 2. 验证 t=1 终点对齐（Float32 单精度 acos 理论精度上限约 5e-4）
    t1 = torch.ones(batch_size, 1, device=device)
    R_t1, p_t1, _, _ = se3_geodesic_interpolate(R_0, p_0, R_1, p_1, t1)
    t1_rot_err = (R_t1 - R_1).abs().max().item()
    print(f"t=1 endpoint rotation max error: {t1_rot_err:.2e}")
    assert t1_rot_err < 1e-3, f"t=1 旋转偏差过大: {t1_rot_err}"
    assert torch.allclose(p_t1, p_1, atol=1e-5), "t=1 位置未对齐终点！"

    # 3. 验证半路 t=0.5 时的正交性
    t_mid = torch.full((batch_size, 1), 0.5, device=device)
    R_mid, _, v_target, omega_target = se3_geodesic_interpolate(R_0, p_0, R_1, p_1, t_mid)
    eye_recon = torch.matmul(R_mid.transpose(-1, -2), R_mid)
    eye_target = torch.eye(3, device=device).expand_as(eye_recon)
    ortho_err = (eye_recon - eye_target).abs().max().item()
    print(f"Midpoint orthogonality error: {ortho_err:.2e}")
    assert ortho_err < 1e-5, "半路旋转矩阵破坏正交性！"

    # 4. 验证 ODE 欧拉积分推演（模拟推理阶段：从起点走 10 步看能否抵达终点）
    R_curr, p_curr = R_0.clone(), p_0.clone()
    num_steps = 10
    dt = 1.0 / num_steps
    for _ in range(num_steps):
        R_curr, p_curr = se3_euler_step(R_curr, p_curr, v_target, omega_target, dt)

    rot_err = (R_curr - R_1).abs().max().item()
    pos_err = (p_curr - p_1).abs().max().item()
    print(f"10-step ODE integration error -> Rotation: {rot_err:.2e}, Position: {pos_err:.2e}")
    assert rot_err < 1e-3 and pos_err < 1e-5, "ODE 多步积分未抵达终点！"

    print("? SE(3) 流形插值与积分测试全部通过！")


if __name__ == "__main__":
    test_se3_manifold()