import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp

# 1. 读取各时点的 scRNA 和 ST 数据（请替换为实际路径）
sc_0h  = sc.read_h5ad("E:/code/deeptalk/time_splits_sc_with_raw/snRNA_qc_full_time_0h_norm1e6.h5ad")
sc_12h = sc.read_h5ad("E:/code/deeptalk/time_splits_sc_with_raw/snRNA_qc_full_time_12h_norm1e6.h5ad")
sc_24h = sc.read_h5ad("E:/code/deeptalk/time_splits_sc_with_raw/snRNA_qc_full_time_24h_norm1e6.h5ad")
st_0h  = sc.read_h5ad("E:/code/deeptalk/L6_full_spatial.h5ad")
st_12h = sc.read_h5ad("E:/code/deeptalk/L6_12h_full_spatial.h5ad")
st_24h = sc.read_h5ad("E:/code/deeptalk/L6_24h_full_spatial.h5ad")

# 2. （可选）标准化 + 对数化
# for ad in [sc_0h, sc_12h, sc_24h, st_0h, st_12h, st_24h]:
#     sc.pp.normalize_total(ad, target_sum=1e4)
#     sc.pp.log1p(ad)

# 3. 合并所有数据，取基因交集
adata_all = sc.concat([
    sc_0h, sc_12h, sc_24h,
    st_0h, st_12h, st_24h
], join='inner', label='modality_time',
    keys=['sc0','sc12','sc24','st0','st12','st24'])
print(f"合并后共有基因数（交集）: {adata_all.n_vars}")

# —— 2. 读取 Marker 基因列表 —— #
marker_file = "E:/code/deeptalk/markerdb_separated_merged.txt"
marker_df   = pd.read_csv(marker_file, sep='\t', dtype=str)
markers     = set(marker_df['msu'].str.replace('_', '-'))
print(f"原始 marker 数: {len(markers)}")

# —— 3. 在 adata_all 中取交集 —— #
present = markers & set(adata_all.var_names)
print(f"在 adata_all 中检测到的 marker 数: {len(present)}")

# —— 4. 在这些 marker 上计算方差并挑出前 N 个 HVG —— #
# 子集化
adata_markers = adata_all[:, list(present)].copy()
Xm = adata_markers.X
if sp.issparse(Xm):
    Xm = Xm.toarray()
# 计算每个基因（列）的方差
vars_m = np.nanvar(Xm, axis=0)
# 选择 top N
N = min(4000, len(present))  # 这里设 N=1000，或者你想要的其他数目
top_idx = np.argsort(-vars_m)[:N]
hvg_markers = [adata_markers.var_names[i] for i in top_idx]
print(f"在这些 markers 中方差排名前 {N} 的 gene 数: {len(hvg_markers)}")

out_dir = "E:/code/deeptalk/selected_HVG4000/"
import os
os.makedirs(out_dir, exist_ok=True)

# 六个数据集的名字和对象
datasets = {
    "sc_0h":  sc_0h,
    "sc_12h": sc_12h,
    "sc_24h": sc_24h,
    "st_0h":  st_0h,
    "st_12h": st_12h,
    "st_24h": st_24h,
}

for name, ad in datasets.items():
    ad_copy = ad.copy()

    # —— 针对 ST 数据，用 raw.X 覆盖 X —— #
    if name.startswith("st_"):
        raw_mat = ad_copy.raw.X
        if sp.issparse(raw_mat):
            raw_mat = raw_mat.toarray()
        ad_copy.X = raw_mat

    # 再对子集做 HVG1000 筛选
    ad_sub = ad_copy[:, hvg_markers].copy()

    # （可选）如果你也想让 raw 只保留这 1000 个基因，可以：
    # ad_sub.raw = ad_sub  # 或者 ad_sub.raw = ad_copy.raw[:, hvg_markers].copy()

    # 保存
    out_path = os.path.join(out_dir, f"{name}_HVG4000.h5ad")
    ad_sub.write_h5ad(out_path)
    print(f"{name} 已保存 {ad_sub.n_vars} 个基因到 {out_path}")