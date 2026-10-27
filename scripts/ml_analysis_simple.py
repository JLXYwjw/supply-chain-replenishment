# -*- coding: utf-8 -*-
"""
供应链需求波动聚类 + 安全库存/再订货点 —— 基础可读版（ml_analysis_simple.py）
==========================================================================

  1. 读入 SKU×仓 表 sku_level.parquet
  2. 构造 4 个特征（对数需求均值、需求CV、前置时间、覆盖率），标准化后用
     KMeans 聚成 3 类，看不同类的画像
  3. 按统计公式算安全库存 SS = z·σ·√L 和再订货点 ROP = d·L + SS，
     并算当前库存距 ROP 的差值 gap
  4. 按 gap 从小到大排，把 250 个 SKU×仓 分成 P1~P4 四个补货优先级
  5. 挑出最需要补货的前 8 个
  6. 结果写入 sku_clusters.parquet 和 ml_result.json

诚实说明：这份合成数据没有缺货正样本，需求预测也是给定的，
所以这里不做预测，只做"差异化补货策略"的决策支持。

运行方式（不带命令行参数）：
    python scripts/ml_analysis_simple.py


English:
Demand-volatility clustering + safety stock / reorder point (plain, readable version).

Steps:
  1. Read the SKU x warehouse table sku_level.parquet
  2. Build 4 features (log mean demand, demand CV, lead time, coverage),
     standardize, then cluster with KMeans into 3 classes and look at each
     class's profile
  3. Compute safety stock SS = z * sigma * sqrt(L) and reorder point
     ROP = d * L + SS by formula, plus the gap between current stock and ROP
  4. Rank the 250 SKU x warehouse combos by gap ascending into 4
     replenishment priorities P1-P4
  5. Pick out the top 8 most in need of replenishment
  6. Write the results into sku_clusters.parquet and ml_result.json

Honest note: the synthetic data has no stockout positives and the demand
forecast is given, so this is not prediction - it is decision support for
a differentiated replenishment strategy.

Run (no command-line arguments):
    python scripts/ml_analysis_simple.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"


def main():
    # ---- 1. 读入 SKU×仓 表 ----
    sku = pd.read_parquet(OUT / "sku_level.parquet")

    r = {}
    r["n_sku_wh"] = len(sku)

    # ---- 2. 构造聚类特征 ----
    # demand_mean_log: 对需求均值取 log1p（压缩量级差异，让大值不主导聚类）
    # demand_cv: 需求变异系数，截断到 [0, 3] 防止极端值
    # coverage: 平均库存覆盖率，截断到 [0, 5]
    feat = pd.DataFrame({
        "demand_mean_log": np.log1p(sku["demand_mean"]),
        "demand_cv": sku["demand_cv"].clip(0, 3),
        "lead_time_days": sku["lead_time_days"],
        "coverage": sku["coverage_mean"].clip(0, 5),
    })
    r["features"] = list(feat.columns)

    # ---- 3. KMeans 聚类（先标准化，再聚 3 类） ----
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    sku["cluster"] = km.fit_predict(X)

    # ---- 4. 各类画像：逐类算平均需求、CV、前置时间、覆盖率 ----
    # 用 groupby().agg() 也能一把算完，这里逐类遍历
    clusters_list = []
    for cid in sorted(sku["cluster"].unique()):
        g = sku[sku["cluster"] == cid]
        clusters_list.append({
            "cluster": int(cid),
            "n": int(len(g)),
            "demand_mean": float(g["demand_mean"].mean()),
            "cv": float(g["demand_cv"].mean()),
            "lead_time": float(g["lead_time_days"].mean()),
            "coverage": float(g["coverage_mean"].mean()),
        })
    r["clusters"] = clusters_list

    # ---- 5. 安全库存 & 再订货点 ----
    # z 值：需求波动大(CV>0.4)的用 1.65（95% 服务水平），否则用 1.28（90%）
    # 用 np.where 也能一行搞定，这里逐行判断
    z_list = []
    for cv_val in sku["demand_cv"]:
        if cv_val > 0.4:
            z_list.append(1.65)
        else:
            z_list.append(1.28)
    sku["z"] = z_list

    # 安全库存 SS = z × σ_demand × √L（L=前置时间天数），保留 1 位小数
    sku["ss"] = (sku["z"] * sku["demand_std"] * np.sqrt(sku["lead_time_days"])).round(1)

    # 再订货点 ROP = 日均需求 × 前置时间 + 安全库存
    sku["rop"] = (sku["demand_mean"] * sku["lead_time_days"] + sku["ss"]).round(1)

    # 当前库存距 ROP 的差值：负=库存不够，正=库存富余
    sku["gap_to_rop"] = (sku["inv_mean"] - sku["rop"]).round(1)

    # 汇总统计
    r["ss_median"] = float(sku["ss"].median())
    r["rop_median"] = float(sku["rop"].median())
    r["below_rop_pct"] = round((sku["inv_mean"] < sku["rop"]).mean() * 100, 2)
    r["high_cv_share"] = round((sku["demand_cv"] > 0.4).mean() * 100, 2)

    # ---- 6. 补货优先级 P1~P4 ----
    # 手动复现 qcut 的分档逻辑（pd.qcut(sku["gap_to_rop"].rank(method="first"), 4, labels=[...])）：
    # 含义：按 gap_to_rop 从小到大排（相同值按原始出现顺序排），
    # 然后按 25%/50%/75% 分位数切成 4 档：P1（最需要补）→ P4。
    # 这里手动复现同样的分位数边界：
    #   1) 把所有行按 gap 值稳定排序
    #   2) 排序后第 j 名的"排名"是 j+1（从 1 开始）
    #   3) 用 np.percentile 算出 25%/50%/75% 分位线，和 pandas qcut 一致
    def gap_value_of(item):
        """从 (行号, gap值) 元组里取出 gap 值，供排序用。"""
        return item[1]

    gap_values = sku["gap_to_rop"].tolist()
    n = len(gap_values)
    # 构造 (原始行号, gap值) 列表，按 gap 值稳定排序
    indexed = list(enumerate(gap_values))
    indexed.sort(key=gap_value_of)   # 按 gap 值升序；相同 gap 保持原顺序

    # 排名是 1..n，算 4 档的分位边界（线性插值，和 pandas 默认一致）
    rank_axis = np.arange(1, n + 1, dtype=float)
    edges = np.percentile(rank_axis, [0, 25, 50, 75, 100])
    # 比如 n=250 时 edges ≈ [1.0, 63.25, 125.5, 187.75, 250.0]

    labels = ["P1", "P2", "P3", "P4"]
    priority_by_row = [None] * n
    for j in range(n):
        rank = j + 1                       # 当前行在排序后的名次（1 起）
        orig_row = indexed[j][0]          # 它在原始 DataFrame 里的行号
        # 从 P1 到 P4 逐档判断：rank 落在哪个区间就归哪档
        # 第一档含左边界，后续档是 (上一边界, 当前边界]
        assigned = False
        for b in range(4):
            if b == 0:
                if rank <= edges[b + 1]:
                    priority_by_row[orig_row] = labels[b]
                    assigned = True
                    break
            else:
                if edges[b] < rank <= edges[b + 1]:
                    priority_by_row[orig_row] = labels[b]
                    assigned = True
                    break
        if not assigned:
            priority_by_row[orig_row] = labels[3]   # 兜底：最大那档
    sku["priority"] = priority_by_row

    # ---- 7. 优先级汇总表 ----
    # 用 groupby("priority").agg(...) 也能一把算完，这里逐档统计
    priority_table = []
    for lab in labels:
        g = sku[sku["priority"] == lab]
        priority_table.append({
            "priority": lab,
            "n": int(len(g)),
            "mean_gap": float(g["gap_to_rop"].mean()),
            "mean_cv": float(g["demand_cv"].mean()),
        })
    r["priority_table"] = priority_table

    # ---- 8. 最需要补货的前 8 个（gap 最小的） ----
    top = sku.sort_values("gap_to_rop").head(8)
    top_cols = ["sku", "wh", "region", "demand_mean", "lead_time_days",
                "inv_mean", "rop", "gap_to_rop", "demand_cv"]
    r["top_needed"] = top[top_cols].to_dict("records")

    # ---- 9. 写出产物 ----
    sku.to_parquet(OUT / "sku_clusters.parquet", index=False)
    with open(OUT / "ml_result.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)

    print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
