# -*- coding: utf-8 -*-
"""供应链补货指标 → metrics.json。"""
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
clean = pd.read_parquet(OUT / "supply_clean.parquet")
sku = pd.read_parquet(OUT / "sku_level.parquet")

m = {}
m["rows"] = len(clean)
m["skus"] = int(sku["sku"].nunique())
m["whs"] = int(sku["wh"].nunique())

# 库存覆盖率分布
cov = clean["inventory_coverage"]
m["coverage_median"] = round(float(cov.median()), 2)
m["coverage_le12_pct"] = round((cov <= 1.2).mean() * 100, 2)   # 临缺货
m["coverage_ge20_pct"] = round((cov >= 2.0).mean() * 100, 2)   # 富余
m["coverage_lt1_pct"] = round((cov < 1.0).mean() * 100, 2)     # 已低于补货点

# 需求与前置时间
m["demand_mean"] = round(float(clean["units_sold"].mean()), 2)
m["lead_time_mean"] = round(float(clean["lead_time_days"].mean()), 2)
m["lead_time_max"] = int(clean["lead_time_days"].max())

# 促销影响（促销日 vs 非促销日销量）
promo = clean.groupby("is_promo")["units_sold"].mean().to_dict()
m["promo_on_mean"] = round(float(promo.get(1, 0)), 2)
m["promo_off_mean"] = round(float(promo.get(0, 0)), 2)
m["promo_lift_pct"] = round((promo.get(1, 0) / promo.get(0, 1) - 1) * 100, 1)

# 区域库存（SKU×仓→区域映射）
reg = sku.groupby("region").agg(rows=("sku", "count")).reset_index()
m["region_dist"] = reg.to_dict("records")

# 需求波动（CV）分布
m["cv_median"] = round(float(sku["demand_cv"].median()), 3)
m["cv_high_pct"] = round((sku["demand_cv"] > 0.4).mean() * 100, 2)

# 库存富余/临缺货的 SKU×仓 数量
m["sku_wh_le12"] = int((sku["coverage_mean"] <= 1.2).sum())
m["sku_wh_ge20"] = int((sku["coverage_mean"] >= 2.0).sum())

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=1)
print(json.dumps(m, ensure_ascii=False), flush=True)
