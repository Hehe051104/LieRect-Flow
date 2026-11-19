现在正式进入第二模块：**`core/manifold.py`**。

这个文件是整个 `LieRect-Flow` 的**轨迹生成器与积分步进器**，负责把前面写好的 `so3_exp` 和 `so3_log` 组装成完整的 6 自由度位姿 $T = (\mathbf{R}, \mathbf{p})$ 运算管道。它包含 4 个核心函数：

1. **`sample_so3_uniform`**：在三维旋转空间里均匀随机撒点（用来生成训练/推理起点的乱序旋转 $\mathbf{R}_0$）。
2. **`sample_se3_noise`**：打包生成完整的初始噪声位姿 $(\mathbf{R}_0, \mathbf{p}_0)$。
3. **`se3_geodesic_interpolate`**：**训练核心**——给定起点 $(\mathbf{R}_0, \mathbf{p}_0)$、终点 $(\mathbf{R}_1, \mathbf{p}_1)$ 和随机时间 $t$，同时算出半路位姿 $(\mathbf{R}_t, \mathbf{p}_t)$（网络输入）和标准目标速度 $(\mathbf{v}_{\text{target}}, \boldsymbol{\omega}_{\text{target}})$（Loss 标签）。
4. **`se3_euler_step`**：**推理核心**——拿到神经网络预测的速度 $(\hat{\mathbf{v}}, \hat{\boldsymbol{\omega}})$ 后，按步长 $\Delta t$ 向前推进一步，更新机械臂夹爪的位姿。

---

### 第一步：创建并编写 `core/manifold.py`

在项目根目录的 `core/` 文件夹下新建 `core/manifold.py`，写入以下代码：

```python
import torch
from typing import Tuple
from core.lie_algebra import so3_exp, so3_log


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

    # 四元数转 3x3 旋转矩阵
    R = torch.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y),
    ], dim=-1).reshape(*batch_shape, 3, 3)

    return R


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
        dt:                 积分时间步长 (例如 1.0 / num_steps)
    Returns:
        R_next, p_next:     推进后的新旋转矩阵与新位置
    """
    p_next = p_t + dt * v_pred
    R_next = torch.matmul(R_t, so3_exp(dt * omega_pred))
    return R_next, p_next

```

---

### 第二步：创建测试文件 `tests/test_manifold.py`

这个测试脚本会验证三件事：

1. **端点对齐**：当 $t=0$ 时，插值结果是否精确等于起点 $(\mathbf{R}_0, \mathbf{p}_0)$；当 $t=1$ 时，是否精确等于终点 $(\mathbf{R}_1, \mathbf{p}_1)$。
2. **全程正交保真**：半路 $t \in (0, 1)$ 算出来的 $\mathbf{R}_t$ 是否依然是合法的旋转矩阵（$\mathbf{R}_t^\top \mathbf{R}_t = \mathbf{I}$）。
3. **ODE 积分闭环还原**：把算出来的标准速度 $(\mathbf{v}_{\text{target}}, \boldsymbol{\omega}_{\text{target}})$ 喂给 `se3_euler_step`，连续推进 10 步，看看能不能从起点严丝合缝地走到终点。

在 `tests/` 目录下新建 `tests/test_manifold.py`：

```python
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

    # 2. 验证 t=1 终点对齐
    t1 = torch.ones(batch_size, 1, device=device)
    R_t1, p_t1, _, _ = se3_geodesic_interpolate(R_0, p_0, R_1, p_1, t1)
    assert torch.allclose(R_t1, R_1, atol=1e-4), "t=1 旋转未对齐终点！"
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
    assert rot_err < 1e-4 and pos_err < 1e-5, "ODE 多步积分未抵达终点！"

    print("? SE(3) 流形插值与积分测试全部通过！")


if __name__ == "__main__":
    test_se3_manifold()

```

---

### 第三步：运行单元测试

在项目根目录 `LieRect-Flow` 下执行：

```bash
python -m tests.test_manifold

```

跑完之后看一下控制台打印的误差数据，如果全部通过，我们就已经打通了整个流形数学引擎，接下来就可以直接搭建预测速度的神经网络了。