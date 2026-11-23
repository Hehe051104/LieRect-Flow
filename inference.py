import torch
from core.lie_algebra import so3_log
from core.manifold import sample_local_se3_noise, se3_euler_step
from models.vector_field import SE3EquivariantVectorField
from datasets.synthetic_grasp import generate_asymmetric_mug_batch


@torch.no_grad()
def evaluate_ode_steps(
    model: SE3EquivariantVectorField,
    points_world: torch.Tensor,
    R_1: torch.Tensor,
    p_1: torch.Tensor,
    num_steps: int,
    use_midpoint: bool = True,
):
    """
    沿预测速度场执行 num_steps 步流形积分，计算最终抓取位置误差与姿态误差。
    use_midpoint=True 时采用流形二阶中点法（RK2），消除一阶欧拉法在旋转时的滞后漂移。
    """
    batch_size = points_world.shape[0]
    device = points_world.device

    # 从目标前方局部扇形区采样随机初始位姿
    R_curr, p_curr = sample_local_se3_noise(R_1, p_1, trans_scale=0.15, rot_scale=0.5)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((batch_size, 1), t_val, device=device)
        v_1, omega_1 = model(R_curr, p_curr, points_world, t)

        if use_midpoint and num_steps > 1:
            # 流形二阶中点法 (Manifold Midpoint / RK2)：
            # 1) 先沿当前速度探测半步 dt/2 处的中间位姿
            R_mid, p_mid = se3_euler_step(R_curr, p_curr, v_1, omega_1, 0.5 * dt)
            t_mid = torch.full((batch_size, 1), t_val + 0.5 * dt, device=device)
            # 2) 用中点处的速度场作为整步平均速度推进完整步长 dt
            v_mid, omega_mid = model(R_mid, p_mid, points_world, t_mid)
            R_curr, p_curr = se3_euler_step(R_curr, p_curr, v_mid, omega_mid, dt)
        else:
            # 标准一阶欧拉步进
            R_curr, p_curr = se3_euler_step(R_curr, p_curr, v_1, omega_1, dt)

    pos_err_mm = torch.norm(p_curr - p_1, dim=-1).mean().item() * 1000.0
    delta_R = torch.matmul(R_curr.transpose(-1, -2), R_1)
    rot_err_rad = torch.norm(so3_log(delta_R), dim=-1)
    rot_err_deg = torch.rad2deg(rot_err_rad).mean().item()

    return pos_err_mm, rot_err_deg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SE3EquivariantVectorField(hidden_dim=256).to(device)
    model.load_state_dict(torch.load("checkpoints/lierect_flow.pt", map_location=device))
    model.eval()

    # 测试集开启 random_pose=True：水杯在全空间随机 3D 旋转和平移
    points_world, R_1, p_1 = generate_asymmetric_mug_batch(
        batch_size=512,
        num_points=512,
        device=device,
        random_pose=True,
    )

    print("Evaluating Upgraded Pure v-Prediction LieRect-Flow (Attention + RK2):")
    print("-" * 68)
    for steps in [1, 2, 5, 10]:
        pos_mm, rot_deg = evaluate_ode_steps(
            model, points_world, R_1, p_1, num_steps=steps, use_midpoint=True
        )
        print(
            f"ODE Steps: {steps:2d} | "
            f"Position Error: {pos_mm:6.2f} mm | "
            f"Rotation Error: {rot_deg:6.2f} deg"
        )
    print("-" * 68)


if __name__ == "__main__":
    main()