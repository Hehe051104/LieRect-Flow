import torch
from core.lie_algebra import hat, vee, so3_exp, so3_log

def test_lie_roundtrip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running unit tests on: {device}")
    
    # 1. 批量随机采样（限制旋转角在 [0, pi - 0.1] 内，避开 pi 拓扑奇异面）
    batch_omega = torch.randn(1024, 3, device=device)  # rand是均匀分布，会导致转轴偏向立方体的八个角，而radnn是高斯分布，任意朝向一个方向的概率相等
    angles = torch.norm(batch_omega, dim=-1, keepdim=True)  # 模长  把omega除这个得到n =  也就是表示转轴的单位向量
    # 将角速度模长缩放到 (0, pi - 0.1) 主值区间内   omega = n * θ  将一个任意向量强行重构为标准的李代数“轴角向量”omega：
    batch_omega = (batch_omega / (angles + 1e-8)) * torch.rand_like(angles) * (torch.pi - 0.1)
    
    R = so3_exp(batch_omega)
    
    # 验证正交性：R^T * R 应为单位矩阵
    eye_recon = torch.matmul(R.transpose(-1, -2), R)
    eye_target = torch.eye(3, device=device).expand_as(eye_recon)
    ortho_err = (eye_recon - eye_target).abs().max().item()
    print(f"Max orthogonality error: {ortho_err:.2e}")
    assert ortho_err < 1e-5, f"正交性校验失败！最大误差: {ortho_err}"
    
    # 验证逆映射：log(exp(w)) 应与原矩阵精确对齐
    omega_rec = so3_log(R)
    R_rec = so3_exp(omega_rec)
    recon_err = (R - R_rec).abs().max().item()
    print(f"Max reconstruction error: {recon_err:.2e}")
    assert recon_err < 1e-4, f"对数逆映射重建失败！最大误差: {recon_err}"
    
    # 2. 测试极小角度 (测试泰勒截断防除零保护)
    tiny_omega = torch.zeros(16, 3, device=device) + 1e-9
    R_tiny = so3_exp(tiny_omega)
    assert not torch.isnan(R_tiny).any(), "小角度 exp 产生 NaN！"
    omega_tiny_rec = so3_log(R_tiny)
    assert not torch.isnan(omega_tiny_rec).any(), "小角度 log 产生 NaN！"
    
    print("✅ 全部单元测试通过！小角度数值稳定，重建精度达标。")

if __name__ == "__main__":
    test_lie_roundtrip()