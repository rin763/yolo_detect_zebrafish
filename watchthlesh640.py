# -*- coding: utf-8 -*-
"""
plot_z_raw_twice_range.py
- Z_raw_m_twice 列の値を使い、1.0〜1.3の範囲にあるデータで
  ヒストグラム + KDE分布を描画・保存
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ====== 設定 ======
CSV_IN = "./box/gate3_imputed_long.csv"
OUT_PNG = "./box/gate3_imputed_long.png"
Z_COL = "Z_m"

# ====== パラメータ ======
Z_MIN, Z_MAX = 1.0, 1.3   # 使用するZ範囲

# ====== データ読込 ======
df = pd.read_csv(CSV_IN)

if Z_COL not in df.columns:
    raise ValueError(f"{Z_COL} がCSV内に存在しません。列名を確認してください。")

# ====== 範囲フィルタ ======
z_values = df[Z_COL].dropna()
before_count = len(z_values)
z_values = z_values[(z_values >= Z_MIN) & (z_values <= Z_MAX)]
after_count = len(z_values)
removed_ratio = (before_count - after_count) / before_count * 100

# ====== 統計 ======
mean_z = z_values.mean()
var_z = z_values.var()
std_z = z_values.std()

print(f"==== Z_raw_m_twice 統計情報 ====")
print(f"総データ数: {before_count}")
print(f"範囲内データ数: {after_count}（除外率: {removed_ratio:.2f}%）")
print(f"平均値: {mean_z:.4f}")
print(f"分散: {var_z:.6f}")
print(f"標準偏差: {std_z:.4f}")
print("==========================")

# ====== KDE ======
kde = gaussian_kde(z_values)
x_range = np.linspace(Z_MIN, Z_MAX, 300)
kde_values = kde(x_range)

# ====== グラフ描画 ======
plt.figure(figsize=(8, 5))
plt.hist(z_values, bins=50, density=True, alpha=0.5, label='Histogram')
plt.plot(x_range, kde_values, 'r-', label='KDE')
plt.axvline(mean_z, color='k', linestyle='--', label=f'Mean = {mean_z:.3f}')
plt.title(f"Z_raw_m_twice Distribution ({Z_MIN} ≤ Z ≤ {Z_MAX})")
plt.xlabel("Z_raw_m_twice (m)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"グラフを {OUT_PNG} に保存しました。")
