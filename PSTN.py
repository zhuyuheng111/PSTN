# -*- coding: utf-8 -*-
"""
Plant Spatio-Temporal Integration Network (PSTN)
Main Training Script |
------------------------------------------------
Description:
- This script jointly optimizes cell-to-space mapping matrices (M_t)
  across multiple infection stages (0h, 12h, 24h) to reconstruct spatial
  transcriptomic profiles from single-cell data with temporal consistency.
- 脚本通过联合优化多个时间点的映射矩阵，实现单细胞数据到空间转录组的时空一致性重建。
"""

import os
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import scipy.sparse as sp

# ================= Configuration | 参数设置 =================
DATA_ROOT = "E:/code/deeptalk/selected_HVG1000"  # Input .h5ad files directory
SAVE_DIR  = "E:/code/deeptalk"                   # Output directory
TIMEPOINTS = [0, 12, 24]

epochs = 2000
lr = 1e-3
lambda_temp = 0.1     # Temporal consistency weight | 时序一致性权重
weight_decay = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= Utility Functions | 工具函数 =================
def to_dense(x):
    """Convert sparse matrix to dense | 稀疏矩阵转稠密矩阵"""
    return x.toarray() if sp.issparse(x) else np.asarray(x)

def get_counts_matrix(ad):
    """Retrieve true raw counts from AnnData | 提取 AnnData 中的原始 counts"""
    if 'raw_X' in ad.layers:
        X = ad.layers['raw_X']
    elif ad.raw is not None and getattr(ad.raw, "n_vars", ad.n_vars) == ad.n_vars:
        X = ad.raw.X
    else:
        X = ad.X
    return to_dense(X).astype(np.float32)

def normalize_log1p_np(mat, target_sum=1e4):
    """Per-spot normalize_total + log1p (NumPy) | 每个 spot 归一化 + log1p（NumPy 版）"""
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    normed = (mat / sums) * target_sum
    return np.log1p(normed)

def normalize_log1p_torch(mat, target_sum=1e4, eps=1e-8):
    """Torch version of normalize_total + log1p (supports backprop)"""
    sums = torch.clamp(mat.sum(dim=1, keepdim=True), min=eps)
    normed = mat / sums * target_sum
    return torch.log1p(torch.clamp(normed, min=0.0))

# ================= Data Loading | 数据读取 =================
sc_times = {t: sc.read_h5ad(os.path.join(DATA_ROOT, f"sc_{t}h_HVG1000.h5ad")) for t in TIMEPOINTS}
st_times = {t: sc.read_h5ad(os.path.join(DATA_ROOT, f"st_{t}h_HVG1000.h5ad")) for t in TIMEPOINTS}

A_count, B_log, M = {}, {}, {}

for t in TIMEPOINTS:
    ad_sc, ad_st = sc_times[t], st_times[t]

    X_sc_count = get_counts_matrix(ad_sc)
    X_st_count = get_counts_matrix(ad_st)
    assert X_sc_count.shape[1] == X_st_count.shape[1], f"{t}h: gene number mismatch."

    X_st_log = normalize_log1p_np(X_st_count, target_sum=1e4)

    A_count[t] = torch.tensor(X_sc_count, dtype=torch.float32, device=device)
    B_log[t]   = torch.tensor(X_st_log,   dtype=torch.float32, device=device)

    n_cells, _ = A_count[t].shape
    n_spots, _ = B_log[t].shape
    M[t] = nn.Parameter(torch.randn(n_cells, n_spots, device=device) * 0.01)

# ================= Optimizer | 优化器 =================
optimizer = optim.Adam(list(M.values()), lr=lr, weight_decay=weight_decay)

# ================= Loss Functions | 损失函数 =================
def static_loss(pred_log, true_log, eps=1e-8):
    """Bidirectional cosine loss (gene-wise + spot-wise) | 双向余弦损失"""
    P_col = F.normalize(pred_log.T + eps, dim=1)
    T_col = F.normalize(true_log.T + eps, dim=1)
    loss_col = - (P_col * T_col).sum(dim=1).mean()
    P_row = F.normalize(pred_log + eps, dim=1)
    T_row = F.normalize(true_log + eps, dim=1)
    loss_row = - (P_row * T_row).sum(dim=1).mean()
    return loss_col + loss_row

# ================= Training Loop | 训练循环 =================
loss_static_hist, loss_temp_hist = [], []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    loss_static = torch.tensor(0.0, device=device)
    pred_log = {}

    # Stage-wise reconstruction | 各时间点表达重构
    for t in TIMEPOINTS:
        pred_raw = torch.clamp(M[t].T @ A_count[t], min=0)
        pred_log[t] = normalize_log1p_torch(pred_raw)
        loss_static += static_loss(pred_log[t], B_log[t])

    # Temporal consistency | 时序一致性约束
    loss_temp = torch.tensor(0.0, device=device)
    E = {t: F.normalize(pred_log[t] + 1e-8, dim=1) for t in TIMEPOINTS}
    S0_12 = E[0] @ E[12].T
    S12_24 = E[12] @ E[24].T
    loss_temp += -0.5 * (S0_12.max(dim=1).values.mean() + S0_12.max(dim=0).values.mean())
    loss_temp += -0.5 * (S12_24.max(dim=1).values.mean() + S12_24.max(dim=0).values.mean())

    loss = loss_static + lambda_temp * loss_temp
    loss.backward()
    optimizer.step()

    loss_static_hist.append(loss_static.item())
    loss_temp_hist.append(loss_temp.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | static: {loss_static.item():.4f} | temp: {loss_temp.item():.4f} | total: {loss.item():.4f}")

# ================= Save Outputs | 保存结果 =================
os.makedirs(SAVE_DIR, exist_ok=True)
for t in TIMEPOINTS:
    np.save(os.path.join(SAVE_DIR, f"A_raw_{t}h.npy"), A_count[t].cpu().numpy())
    np.save(os.path.join(SAVE_DIR, f"B_{t}h.npy"),     B_log[t].cpu().numpy())
    torch.save(M[t].cpu(), os.path.join(SAVE_DIR, f"mapping_matrix_{t}h.pt"))

print(f"Training finished. Results saved to: {SAVE_DIR}")

# ================= Visualization | 可视化 =================
x = range(1, epochs + 1)
plt.figure()
plt.plot(x, loss_static_hist, label='Static Loss')
plt.plot(x, loss_temp_hist, label='Temporal Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Static vs Temporal Loss (PSTN)")
plt.legend(); plt.tight_layout(); plt.show()
