# -*- coding: utf-8 -*-
"""供应链清洗 + SKU×仓级聚合（合成网格，无缺失，重点派生补货指标）。
输出：supply_clean.parquet（行级）、sku_level.parquet（SKU×仓）、cleaning_report.json
"""
import json
from pathlib import Path
import pandas as pd

RAW = Path(r"D:\B编程\powerBI\github\datasets\inventory\supply_chain_dataset1.csv")
OUT = Path(__file__).resolve().parent.parent / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
report = {}
report["rows_in"] = len(df)
report["missing"] = {c: int(df[c].isna().sum()) for c in df.columns}

df["Date"] = pd.to_datetime(df["Date"])
df["ym"] = df["Date"].dt.to_period("M").astype(str)
df["weekday"] = df["Date"].dt.dayofweek
report["date_min"] = str(df["Date"].min().date())
report["date_max"] = str(df["Date"].max().date())
report["skus"] = int(df["SKU_ID"].nunique())
report["warehouses"] = int(df["Warehouse_ID"].nunique())

# 派生补货指标
df["revenue"] = df["Units_Sold"] * df["Unit_Price"]
df["inventory_coverage"] = (df["Inventory_Level"] / df["Reorder_Point"]).round(3)
df["demand_vs_order"] = df["Order_Quantity"] - df["Units_Sold"]
df["stockout"] = df["Stockout_Flag"]  # 全 0，诚实记录
report["stockout_positive"] = int(df["Stockout_Flag"].sum())
report["stockout_rate"] = round(df["Stockout_Flag"].mean() * 100, 4)

clean = df.rename(columns={
    "SKU_ID": "sku", "Warehouse_ID": "wh", "Supplier_ID": "supplier",
    "Region": "region", "Units_Sold": "units_sold", "Inventory_Level": "inventory_level",
    "Supplier_Lead_Time_Days": "lead_time_days", "Reorder_Point": "reorder_point",
    "Order_Quantity": "order_quantity", "Unit_Cost": "unit_cost", "Unit_Price": "unit_price",
    "Promotion_Flag": "is_promo", "Stockout_Flag": "is_stockout", "Demand_Forecast": "demand_forecast",
})
clean = clean[["Date", "ym", "weekday", "sku", "wh", "supplier", "region",
               "units_sold", "inventory_level", "lead_time_days", "reorder_point",
               "order_quantity", "unit_cost", "unit_price", "is_promo", "is_stockout",
               "demand_forecast", "revenue", "inventory_coverage", "demand_vs_order"]]
clean = clean.sort_values(["sku", "wh", "Date"])
clean.to_parquet(OUT / "supply_clean.parquet", index=False)
report["rows_out"] = len(clean)

# SKU×仓聚合
sku = clean.groupby(["sku", "wh"]).agg(
    demand_mean=("units_sold", "mean"),
    demand_std=("units_sold", "std"),
    demand_cv=("units_sold", lambda s: s.std() / s.mean() if s.mean() > 0 else 0),
    inv_mean=("inventory_level", "mean"),
    reorder_point=("reorder_point", "first"),
    lead_time_days=("lead_time_days", "first"),
    unit_price=("unit_price", "first"),
    region=("region", "first"),
    supplier=("supplier", "first"),
).reset_index()
sku["coverage_mean"] = (sku["inv_mean"] / sku["reorder_point"]).round(2)
sku["n_days"] = 365
sku.to_parquet(OUT / "sku_level.parquet", index=False)
report["sku_wh_rows"] = len(sku)

with open(OUT / "cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False), flush=True)
