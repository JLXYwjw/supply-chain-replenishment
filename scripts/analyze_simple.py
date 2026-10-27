# -*- coding: utf-8 -*-
"""
供应链补货指标汇总 —— 基础可读版（analyze_simple.py）
======================================================

  1. 读入行级干净表 supply_clean.parquet 和 SKU×仓 表 sku_level.parquet
  2. 计算一组补货关键指标：库存覆盖率分布、需求均值、前置时间、
     促销影响、区域分布、需求波动(CV)分布、临缺货/富余的 SKU×仓 数量
  3. 把结果写成 metrics.json 并打印出来

区别只是：本文件用"最朴素的写法"（for 循环、if 判断），
把 groupby().agg() 和字典推导拆成了逐行可读的步骤。

运行方式（不带命令行参数）：
    python scripts/analyze_simple.py


English:
Supply-chain replenishment metric summary (plain, readable version).

Steps:
  1. Read the row-level clean table supply_clean.parquet and the
     SKU x warehouse table sku_level.parquet
  2. Compute a set of replenishment key metrics: inventory-coverage
     distribution, mean demand, lead time, promo impact, regional
     distribution, demand-volatility (CV) distribution, and the number of
     SKU x warehouse combos near stockout / in surplus
  3. Write the results into metrics.json and print them

The code uses the plainest style (for-loops, if-statements): the
groupby().agg() and the dict comprehensions are unfolded into
step-by-step readable code.

Run (no command-line arguments):
    python scripts/analyze_simple.py
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"


def main():
    # ---- 1. 读入上游产物 ----
    clean = pd.read_parquet(OUT / "supply_clean.parquet")
    sku = pd.read_parquet(OUT / "sku_level.parquet")

    m = {}

    # ---- 2. 基础规模指标 ----
    m["rows"] = len(clean)
    m["skus"] = int(sku["sku"].nunique())
    m["whs"] = int(sku["wh"].nunique())

    # ---- 3. 库存覆盖率分布 ----
    # inventory_coverage = 当前库存 / 补货点
    cov = clean["inventory_coverage"]
    m["coverage_median"] = round(float(cov.median()), 2)
    m["coverage_le12_pct"] = round((cov <= 1.2).mean() * 100, 2)   # 临缺货（≤1.2倍补货点）
    m["coverage_ge20_pct"] = round((cov >= 2.0).mean() * 100, 2)   # 富余（≥2倍补货点）
    m["coverage_lt1_pct"] = round((cov < 1.0).mean() * 100, 2)     # 已低于补货点

    # ---- 4. 需求与前置时间 ----
    m["demand_mean"] = round(float(clean["units_sold"].mean()), 2)
    m["lead_time_mean"] = round(float(clean["lead_time_days"].mean()), 2)
    m["lead_time_max"] = int(clean["lead_time_days"].max())

    # ---- 5. 促销影响 ----
    # 按 is_promo 分组，逐组算销量均值，存到字典里
    # （一行也能算：clean.groupby("is_promo")["units_sold"].mean().to_dict()）
    promo_means = {}
    for flag, g in clean.groupby("is_promo"):
        promo_means[int(flag)] = float(g["units_sold"].mean())

    promo_on = promo_means.get(1, 0)    # 促销日平均销量
    promo_off = promo_means.get(0, 0)   # 平日平均销量
    m["promo_on_mean"] = round(promo_on, 2)
    m["promo_off_mean"] = round(promo_off, 2)
    # 促销提升幅度 = (促销日均值/平日均值 - 1) × 100%
    m["promo_lift_pct"] = round((promo_on / promo_off - 1) * 100, 1)

    # ---- 6. 区域库存分布 ----
    # 按 region 分组，逐组数有多少个 SKU×仓
    # （一行也能算：sku.groupby("region").agg(rows=("sku","count")).reset_index().to_dict("records")）
    region_dist = []
    for region_name, g in sku.groupby("region", sort=True):
        region_dist.append({"region": region_name, "rows": len(g)})
    m["region_dist"] = region_dist

    # ---- 7. 需求波动（CV）分布 ----
    m["cv_median"] = round(float(sku["demand_cv"].median()), 3)
    m["cv_high_pct"] = round((sku["demand_cv"] > 0.4).mean() * 100, 2)

    # ---- 8. 库存富余/临缺货的 SKU×仓 数量 ----
    m["sku_wh_le12"] = int((sku["coverage_mean"] <= 1.2).sum())
    m["sku_wh_ge20"] = int((sku["coverage_mean"] >= 2.0).sum())

    # ---- 9. 写出 metrics.json 并打印 ----
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)

    print(json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
