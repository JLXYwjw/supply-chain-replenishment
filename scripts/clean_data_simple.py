# -*- coding: utf-8 -*-
"""
供应链清洗 + SKU×仓级聚合 —— 基础可读版（clean_data_simple.py）
================================================================
这个文件做的事，和原版 clean_data.py 完全一样：
  1. 读入合成供应链原始 CSV（supply_chain_dataset1.csv，共 91250 行）
  2. 把 Date 列转成真正的日期，并派生"年月 ym"和"星期几 weekday"
  3. 派生补货相关指标：收入 revenue、库存覆盖率 inventory_coverage、
     需求 vs 订货量差值 demand_vs_order、缺货标记 stockout
  4. 把列名统一改成下划线小写，按 SKU+仓+日期排序，存成行级干净表
     supply_clean.parquet
  5. 按 SKU×仓 聚合，算每个组合的需求均值/标准差/变异系数等，
     存成 sku_level.parquet
  6. 把上述统计写成 cleaning_report.json 并打印出来

区别只是：本文件故意用"最朴素的写法"（for 循环、if 判断、普通函数），
每一步都写清楚了在做什么、为什么。

运行方式（和原版一样，不带命令行参数）：
    python scripts/clean_data_simple.py
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
# 本文件在 scripts/ 下，上一级就是项目根目录
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "supply_chain_dataset1.csv"   # 原始 CSV
OUT = ROOT / "data" / "output"                               # 产物输出目录
OUT.mkdir(parents=True, exist_ok=True)                        # 确保输出目录存在


def main():
    # ---- 1. 读入原始 CSV ----
    df = pd.read_csv(RAW)

    # report 用来收集整个清洗过程中的统计信息，最后写成 json
    report = {}
    report["rows_in"] = len(df)

    # 统计每一列有多少个缺失值（原版是一行字典推导，这里拆开写）
    missing = {}
    for col in df.columns:
        missing[col] = int(df[col].isna().sum())
    report["missing"] = missing

    # ---- 2. 把 Date 列转成真正的日期 ----
    df["Date"] = pd.to_datetime(df["Date"])

    # 派生"年月"：比如 2024-03，方便后面按月分组
    df["ym"] = df["Date"].dt.to_period("M").astype(str)

    # 派生"星期几"：0=周一, 6=周日（原版用 dayofweek，等价的新写法是 day_of_week）
    df["weekday"] = df["Date"].dt.day_of_week

    # 记录数据覆盖的日期范围、SKU 数、仓库数
    report["date_min"] = str(df["Date"].min().date())
    report["date_max"] = str(df["Date"].max().date())
    report["skus"] = int(df["SKU_ID"].nunique())
    report["warehouses"] = int(df["Warehouse_ID"].nunique())

    # ---- 3. 派生补货相关指标 ----
    # 收入 = 销量 × 单价
    df["revenue"] = df["Units_Sold"] * df["Unit_Price"]

    # 库存覆盖率 = 当前库存 / 补货点，保留 3 位小数；
    # 大于 1 说明库存还够卖，小于 1 说明已经低于补货点了
    df["inventory_coverage"] = (df["Inventory_Level"] / df["Reorder_Point"]).round(3)

    # 需求 vs 订货量：订了多少减去实际卖了多少
    df["demand_vs_order"] = df["Order_Quantity"] - df["Units_Sold"]

    # 缺货标记直接搬过来（这份合成数据里全是 0，诚实地记录下来）
    df["stockout"] = df["Stockout_Flag"]

    # 记录缺货相关统计
    report["stockout_positive"] = int(df["Stockout_Flag"].sum())
    report["stockout_rate"] = round(df["Stockout_Flag"].mean() * 100, 4)

    # ---- 4. 统一列名（原版是 rename 一次性映射，这里照抄映射表） ----
    rename_map = {
        "SKU_ID": "sku",
        "Warehouse_ID": "wh",
        "Supplier_ID": "supplier",
        "Region": "region",
        "Units_Sold": "units_sold",
        "Inventory_Level": "inventory_level",
        "Supplier_Lead_Time_Days": "lead_time_days",
        "Reorder_Point": "reorder_point",
        "Order_Quantity": "order_quantity",
        "Unit_Cost": "unit_cost",
        "Unit_Price": "unit_price",
        "Promotion_Flag": "is_promo",
        "Stockout_Flag": "is_stockout",
        "Demand_Forecast": "demand_forecast",
    }
    clean = df.rename(columns=rename_map)

    # 挑出我们要保留的列，按这个顺序排列
    keep_cols = [
        "Date", "ym", "weekday", "sku", "wh", "supplier", "region",
        "units_sold", "inventory_level", "lead_time_days", "reorder_point",
        "order_quantity", "unit_cost", "unit_price", "is_promo", "is_stockout",
        "demand_forecast", "revenue", "inventory_coverage", "demand_vs_order",
    ]
    clean = clean[keep_cols]

    # 按 SKU → 仓 → 日期 排序，方便后面看时间序列
    clean = clean.sort_values(["sku", "wh", "Date"])

    # 存成行级干净表
    clean.to_parquet(OUT / "supply_clean.parquet", index=False)
    report["rows_out"] = len(clean)

    # ---- 5. 按 SKU×仓 聚合 ----
    # 原版用 groupby().agg(...) 一把算完，里面有个 lambda 算变异系数。
    # 这里改成：先按 (sku, wh) 分组，再逐组用普通 Python 算每个统计量，
    # 这样每一步在算什么一目了然。
    groups = clean.groupby(["sku", "wh"], sort=True)

    # 准备好要写回结果的列表（每行是一个 SKU×仓 的统计）
    sku_rows = []

    for (one_sku, one_wh), g in groups:
        # 这一组里所有天的销量
        sales = g["units_sold"]
        sales_mean = sales.mean()                       # 销量均值
        sales_std = sales.std()                         # 销量标准差（pandas 默认 ddof=1）

        # 变异系数 CV = 标准差/均值；均值为 0 时记 0，避免除零
        if sales_mean > 0:
            demand_cv = sales_std / sales_mean
        else:
            demand_cv = 0

        inv_mean = g["inventory_level"].mean()         # 库存均值
        # 下面这几个字段在同一个 SKU×仓 里是不变的，取第一行的值即可
        rp = g["reorder_point"].iloc[0]                 # 补货点
        lt = g["lead_time_days"].iloc[0]               # 前置时间
        up = g["unit_price"].iloc[0]                   # 单价
        reg = g["region"].iloc[0]                      # 区域
        sup = g["supplier"].iloc[0]                    # 供应商

        sku_rows.append({
            "sku": one_sku,
            "wh": one_wh,
            "demand_mean": sales_mean,
            "demand_std": sales_std,
            "demand_cv": demand_cv,
            "inv_mean": inv_mean,
            "reorder_point": rp,
            "lead_time_days": lt,
            "unit_price": up,
            "region": reg,
            "supplier": sup,
        })

    # 把逐组统计结果拼成 DataFrame
    sku = pd.DataFrame(sku_rows)

    # 派生：平均库存覆盖率 = 库存均值 / 补货点，保留 2 位小数
    sku["coverage_mean"] = (sku["inv_mean"] / sku["reorder_point"]).round(2)

    # 这份数据覆盖 365 天，直接写死
    sku["n_days"] = 365

    # 存成 SKU×仓 级表
    sku.to_parquet(OUT / "sku_level.parquet", index=False)
    report["sku_wh_rows"] = len(sku)

    # ---- 6. 写出清洗报告 json 并打印 ----
    with open(OUT / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
