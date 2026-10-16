# -*- coding: utf-8 -*-
"""供应链：需求波动聚类 + 安全库存/再订货点计算。
- KMeans 按 SKU×仓需求特征（均值/波动/前置时间/覆盖率）分组 → 差异化补货策略
- 安全库存 SS = z·σ_L，再订货点 ROP = d·L + SS（统计公式，解释原理）
- 诚实：无缺货正样本(DL分类不可做)、需求预测已给定 → 不预测，做决策支持
输出：ml_result.json + sku_clusters.parquet
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
sku = pd.read_parquet(OUT / "sku_level.parquet")

r = {}
r["n_sku_wh"] = len(sku)

# 特征
feat = pd.DataFrame({
    "demand_mean_log": np.log1p(sku["demand_mean"]),
    "demand_cv": sku["demand_cv"].clip(0, 3),
    "lead_time_days": sku["lead_time_days"],
    "coverage": sku["coverage_mean"].clip(0, 5),
})
r["features"] = list(feat.columns)

# KMeans（需求波动为主的分组）
scaler = StandardScaler()
X = scaler.fit_transform(feat)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
sku["cluster"] = km.fit_predict(X)

# 组画像
prof = sku.groupby("cluster").agg(
    n=("sku", "count"), demand_mean=("demand_mean", "mean"),
    cv=("demand_cv", "mean"), lead_time=("lead_time_days", "mean"),
    coverage=("coverage_mean", "mean"),
).reset_index()
r["clusters"] = prof.to_dict("records")

# 安全库存 & 再订货点（按 cluster 差异化 z，需求波动大→高安全库存）
zmap = {0: 1.28, 1: 1.28, 2: 1.28}
# 实际按 CV 分档：低波动 z=1.28，高波动 z=1.65
sku["z"] = np.where(sku["demand_cv"] > 0.4, 1.65, 1.28)
sku["ss"] = (sku["z"] * sku["demand_std"] * np.sqrt(sku["lead_time_days"])).round(1)
sku["rop"] = (sku["demand_mean"] * sku["lead_time_days"] + sku["ss"]).round(1)
sku["gap_to_rop"] = (sku["inv_mean"] - sku["rop"]).round(1)
r["ss_median"] = float(sku["ss"].median())
r["rop_median"] = float(sku["rop"].median())
r["below_rop_pct"] = round((sku["inv_mean"] < sku["rop"]).mean() * 100, 2)
r["high_cv_share"] = round((sku["demand_cv"] > 0.4).mean() * 100, 2)

# 补货优先级：gap 越负越需补
sku["priority"] = pd.qcut(sku["gap_to_rop"].rank(method="first"), 4, labels=["P1", "P2", "P3", "P4"])
r["priority_table"] = sku.groupby("priority", observed=False).agg(
    n=("sku", "count"), mean_gap=("gap_to_rop", "mean"),
    mean_cv=("demand_cv", "mean"),
).reset_index().to_dict("records")

r["top_needed"] = sku.sort_values("gap_to_rop").head(8)[
    ["sku", "wh", "region", "demand_mean", "lead_time_days", "inv_mean", "rop", "gap_to_rop", "demand_cv"]
].to_dict("records")

sku.to_parquet(OUT / "sku_clusters.parquet", index=False)
with open(OUT / "ml_result.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)
