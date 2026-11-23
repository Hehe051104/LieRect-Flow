import os
import time
import torch
import torch.nn.functional as F
from core.manifold import sample_local_se3_noise, se3_geodesic_interpolate
from models.vector_field import SE3EquivariantVectorField
from datasets.synthetic_grasp import generate_asymmetric_mug_batch


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting LieRect-Flow training on: {device}")

    num_steps = 8000     # 充分迭代以打磨亚度级对齐
    batch_size = 512     # 大 Batch 稳定梯度
    num_points = 512     # 水杯表面固定网格点数
    lr = 1e-3            # 初始学习率

    model = SE3EquivariantVectorField(hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=1e-5
    )

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

    model.train()
    for step in range(1, num_steps + 1):
        # 1. 获取目标抓取数据（训练时水杯固定在原点，测试零样本空间泛化）
        points_world, R_1, p_1 = generate_asymmetric_mug_batch(
            batch_size=batch_size,
            num_points=num_points,
            device=device,
            random_pose=False,
        )

        # 2. 在目标前方局部扇形区采样初始位姿 (R_0, p_0)
        R_0, p_0 = sample_local_se3_noise(R_1, p_1, trans_scale=0.15, rot_scale=0.5)

        # 3. 采样时间进度 t ∈ [0, 0.98]
        # 使用 u^0.75 略微增加后半段（靠近终点精细对齐区）的采样比例，同时保留 0.98 上限保护
        u = torch.rand(batch_size, 1, device=device)
        t = (u ** 0.75) * 0.98

        # 4. 流形测地线插值得到半路位姿与恒定标准速度
        R_t, p_t, v_target, omega_target = se3_geodesic_interpolate(R_0, p_0, R_1, p_1, t)

        # 5. 网络直接预测物理速度
        v_pred, omega_pred = model(R_t, p_t, points_world, t)

        # 6. 混合损失函数（MSE + 0.2 * L1）：
        # MSE 负责大误差快速收敛；L1 在误差小于 3° 时仍提供恒定梯度，彻底消除最后几度的残余偏差
        loss_trans = F.mse_loss(v_pred, v_target) + 0.2 * F.l1_loss(v_pred, v_target)
        loss_rot = F.mse_loss(omega_pred, omega_target) + 0.2 * F.l1_loss(omega_pred, omega_target)
        loss = loss_trans + loss_rot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step [{step:4d}/{num_steps}] | Total: {loss.item():.5f} | "
                f"Trans: {loss_trans.item():.5f} | Rot: {loss_rot.item():.5f} | "
                f"LR: {current_lr:.6f} | Time: {time.time() - start_time:.1f}s"
            )

    save_path = "checkpoints/lierect_flow.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training finished! Model saved to {save_path}")


if __name__ == "__main__":
    train()