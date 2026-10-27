# -*- coding: utf-8 -*-
r"""
供应链补货分析 → Tableau + PowerBI 双份报表 —— 基础可读版（build_bi_simple.py）
==============================================================================

  1. 读入行级干净表和 SKU 聚类表，派生报表需要的列（收入百万、覆盖率分桶、
     促销分组、簇标签等），算出一组汇总指标 f
  2. 导出 supply_fact.csv 和 supply_sku.csv
  3. 用 twbkit 生成 Tableau 工作簿（6 个工作表 + 1 个仪表盘）
  4. 用 pbipkit 生成 PowerBI 工程（语义模型 + 1 个报表页）

诚实说明：这份数据是合成数据，需求几乎无差异
  （50 个 SKU 各约 36.6k，5 个仓各约 36.6 万，缺货率 0%，0 个 SKU 低于再订货点）。
所以结论不能编"哪些 SKU 危险"，只能讲策略本身过度保守。

运行方式（不带命令行参数）：
    python scripts/build_bi_simple.py


English:
Supply-chain analysis -> Tableau + Power BI twin reports (plain, readable version).

Steps:
  1. Read the row-level clean table and the SKU cluster table, derive the
     columns the reports need (revenue in millions, coverage buckets, promo
     groups, cluster labels), and compute a set of summary metrics f
  2. Export supply_fact.csv and supply_sku.csv
  3. Use twbkit to generate the Tableau workbook (6 worksheets + 1 dashboard)
  4. Use pbipkit to generate the Power BI project (semantic model +
     1 report page)

Honest note: this data is synthetic, demand shows almost no variance
(50 SKUs at ~36.6k each, 5 warehouses at ~366k each, 0% stockout rate,
0 SKUs below the reorder point). So the conclusion cannot be "which SKUs
are dangerous"; it can only be that the strategy itself is
over-conservative.

Run (no command-line arguments):
    python scripts/build_bi_simple.py
"""

import os
import sys

import pandas as pd

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 把 bi_kit 工具目录加到 Python 搜索路径里
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "tools", "bi_kit"))
import csvkit                      # noqa: E402
import pbipkit as K                # noqa: E402
import twbkit                      # noqa: E402

SRC = os.path.join(ROOT, "data", "output", "supply_clean.parquet")
CLU = os.path.join(ROOT, "data", "output", "sku_clusters.parquet")
BI = os.path.join(ROOT, "data", "bi")

# 覆盖率分桶标签（和分桶区间一一对应）
COV_LAB = ["<1 周", "1-1.5 周", "1.5-2 周", "2-3 周", ">3 周"]


# ------------------------------------------------------------------
# 1. 数据准备：读 parquet、派生列、算汇总指标
# ------------------------------------------------------------------
def prep():
    # ---- 1.1 读入行级干净表 ----
    d = pd.read_parquet(SRC)

    # 收入换算成百万
    d["rev_m"] = d["revenue"] / 1e6

    # 把覆盖率切成 5 个桶（左闭右开），转成字符串标签
    d["cov_bucket"] = pd.cut(
        d["inventory_coverage"],
        [0, 1, 1.5, 2, 3, 10 ** 6],
        labels=COV_LAB, right=False,
    ).astype(str)

    # 给每个桶一个排序序号（用循环建映射表）
    cov_order_map = {}
    for i, lab in enumerate(COV_LAB):
        cov_order_map[lab] = i
    d["cov_order"] = d["cov_bucket"].map(cov_order_map)

    # 促销分组：1=促销日，0=平日，缺失也算平日
    d["promo_group"] = d["is_promo"].map({1: "促销日", 0: "平日"}).fillna("平日")

    # ---- 1.2 组装事实表（日级明细） ----
    fact = d[["ym", "sku", "wh", "region", "promo_group", "cov_bucket", "cov_order",
              "units_sold", "inventory_coverage", "lead_time_days", "rev_m"]].copy()
    fact["line_count"] = 1                              # 每行计 1，方便求和
    fact["stockout_flag"] = d["is_stockout"].astype(int).values

    # ---- 1.3 组装 SKU 维度表 ----
    c = pd.read_parquet(CLU).copy()

    # 簇标签：拆成两步，先算每个簇的平均前置时间，再逐行拼标签
    #   "C" + cluster + " 提前期" + 平均前置时间 + "天"
    cluster_lt_mean = {}
    for cid, g in c.groupby("cluster"):
        cluster_lt_mean[cid] = int(round(g["lead_time_days"].mean()))

    cluster_labels = []
    for _, row in c.iterrows():
        cid = row["cluster"]
        lt = cluster_lt_mean[cid]
        cluster_labels.append("C" + str(int(cid)) + " 提前期" + str(lt) + "天")
    c["cluster_label"] = cluster_labels
    c["one"] = 1

    sku = c[["sku", "wh", "cluster", "cluster_label", "priority", "demand_mean",
             "demand_cv", "lead_time_days", "coverage_mean", "ss", "rop",
             "gap_to_rop", "one"]].copy()

    # ---- 1.4 算汇总指标 f ----
    promo = d[d["is_promo"] == 1]["units_sold"].mean()
    plain = d[d["is_promo"] == 0]["units_sold"].mean()

    # 各覆盖率桶的行数：d 里没有 line_count 列，所以走 else 分支用 size()
    if "line_count" in d:
        cov = d.groupby("cov_bucket", observed=True)["line_count"].sum()
    else:
        cov = d.groupby("cov_bucket", observed=True).size()
    cov_pct = cov / cov.sum() * 100

    f = dict(
        rows=len(d),
        skus=d["sku"].nunique(),
        whs=d["wh"].nunique(),
        rev=d["rev_m"].sum(),
        units=int(d["units_sold"].sum()),
        cov_med=d["inventory_coverage"].median(),
        cov_mode=cov_pct.idxmax(),
        cov_mode_pct=cov_pct.max(),
        cov_low=(d["inventory_coverage"] < 1).mean() * 100,
        promo=promo,
        plain=plain,
        lift=(promo / plain - 1) * 100,
        stockout=d["is_stockout"].mean() * 100 if "is_stockout" in d else 0.0,
        below_rop=(c["gap_to_rop"] < 0).mean() * 100,
        ss_med=c["ss"].median(),
        gap_med=c["gap_to_rop"].median(),
        p1=int((c["priority"] == "P1").sum()),
        lt_min=c.groupby("cluster")["lead_time_days"].mean().min(),
        lt_max=c.groupby("cluster")["lead_time_days"].mean().max(),
        wh_max=d.groupby("wh")["units_sold"].sum().max(),
        wh_min=d.groupby("wh")["units_sold"].sum().min(),
    )
    f["wh_ratio"] = f["wh_max"] / f["wh_min"]
    f["lt_ratio"] = f["lt_max"] / f["lt_min"]
    return fact, sku, f


# ------------------------------------------------------------------
# 2. 生成 Tableau 工作簿
# ------------------------------------------------------------------
def build_tableau(fact, sku, f):
    F_DS, F_TS, F_CAP = "federated.0supply", "textscan.0supply", "需求明细"
    S_DS, S_TS, S_CAP = "federated.0skucl", "textscan.0skucl", "SKU 分层"

    # 事实表字段定义（字段名, 数据类型, 角色, 聚合方式, 宽度, 中文别名）
    fact_fields = [
        ("ym",                 "string",  "dimension", "nominal",      129, "年月"),
        ("sku",                "string",  "dimension", "nominal",      129, "SKU"),
        ("wh",                 "string",  "dimension", "nominal",      129, "仓库"),
        ("region",             "string",  "dimension", "nominal",      129, "区域"),
        ("promo_group",        "string",  "dimension", "nominal",      129, "促销"),
        ("cov_bucket",         "string",  "dimension", "nominal",      129, "覆盖天数段"),
        ("cov_order",          "integer", "measure",   "quantitative",  20, "覆盖序"),
        ("units_sold",         "integer", "measure",   "quantitative",  20, "销量"),
        ("inventory_coverage", "real",    "measure",   "quantitative",   5, "库存覆盖"),
        ("lead_time_days",     "real",    "measure",   "quantitative",   5, "提前期"),
        ("rev_m",              "real",    "measure",   "quantitative",   5, "收入(百万)"),
        ("line_count",         "integer", "measure",   "quantitative",  20, "行数"),
        ("stockout_flag",      "integer", "measure",   "quantitative",  20, "缺货标记"),
    ]
    # SKU 维度表字段定义
    sku_fields = [
        ("sku",           "string",  "dimension", "nominal",      129, "SKU"),
        ("wh",            "string",  "dimension", "nominal",      129, "仓库"),
        ("cluster",       "integer", "dimension", "ordinal",       20, "需求簇"),
        ("cluster_label", "string",  "dimension", "nominal",      129, "簇标签"),
        ("priority",      "string",  "dimension", "nominal",      129, "优先级"),
        ("demand_mean",   "real",    "measure",   "quantitative",   5, "日均需求"),
        ("demand_cv",     "real",    "measure",   "quantitative",   5, "需求变异"),
        ("lead_time_days", "real",   "measure",   "quantitative",   5, "提前期"),
        ("coverage_mean", "real",    "measure",   "quantitative",   5, "平均覆盖"),
        ("ss",            "real",    "measure",   "quantitative",   5, "安全库存"),
        ("rop",           "real",    "measure",   "quantitative",   5, "再订货点"),
        ("gap_to_rop",    "real",    "measure",   "quantitative",   5, "距再订货点"),
        ("one",           "integer", "measure",   "quantitative",  20, "SKU数"),
    ]

    F, S = F_DS, S_DS
    SF = f"[{F}].[sum:units_sold:qk]"
    AVG_U = f"[{F}].[avg:units_sold:qk]"
    CNT = f"[{F}].[sum:line_count:qk]"

    # 6 个工作表的配置（名称、依赖字段、行列、标记类型、排序、标题等）
    sheets = [
        dict(name="1 覆盖分布", ds=F,
             deps=[("string", "cov_bucket", "dimension", "nominal", "none:cov_bucket:nk", "None", "nominal"),
                   ("integer", "cov_order", "measure", "quantitative", "sum:cov_order:qk", "Sum", "quantitative"),
                   ("integer", "line_count", "measure", "quantitative", "sum:line_count:qk", "Sum", "quantitative")],
             rows=f"[{F}].[none:cov_bucket:nk]", cols=CNT, mark="Bar",
             encodings=[("text", CNT)], labels=True,
             sort=(f"[{F}].[none:cov_bucket:nk]", f"[{F}].[sum:cov_order:qk]", "ASC"),
             title=f"{f['cov_mode_pct']:.0f}% 落在 {f['cov_mode']}"),

        dict(name="2 促销提升", ds=F,
             deps=[("string", "promo_group", "dimension", "nominal", "none:promo_group:nk", "None", "nominal"),
                   ("integer", "units_sold", "measure", "quantitative", "avg:units_sold:qk", "Avg", "quantitative")],
             rows=f"[{F}].[none:promo_group:nk]", cols=AVG_U, mark="Bar",
             encodings=[("text", AVG_U)], labels=True,
             num_format=(AVG_U, "*0.0"),
             title=f"促销拉高需求 {f['lift']:.1f}%"),

        dict(name="3 各仓销量", ds=F,
             deps=[("string", "wh", "dimension", "nominal", "none:wh:nk", "None", "nominal"),
                   ("integer", "units_sold", "measure", "quantitative", "sum:units_sold:qk", "Sum", "quantitative")],
             rows=f"[{F}].[none:wh:nk]", cols=SF, mark="Bar",
             encodings=[("text", SF)], labels=True,
             sort=(f"[{F}].[none:wh:nk]", SF, "DESC"),
             num_format=(SF, "#,##0"),
             title=f"5 个仓销量仅差 {(f['wh_ratio'] - 1) * 100:.1f}%"),

        dict(name="4 提前期分簇", ds=S,
             deps=[("string", "cluster_label", "dimension", "nominal", "none:cluster_label:nk", "None", "nominal"),
                   ("real", "lead_time_days", "measure", "quantitative", "avg:lead_time_days:qk", "Avg", "quantitative")],
             rows=f"[{S}].[none:cluster_label:nk]",
             cols=f"[{S}].[avg:lead_time_days:qk]", mark="Bar",
             encodings=[("text", f"[{S}].[avg:lead_time_days:qk]")], labels=True,
             sort=(f"[{S}].[none:cluster_label:nk]",
                   f"[{S}].[avg:lead_time_days:qk]", "ASC"),
             num_format=(f"[{S}].[avg:lead_time_days:qk]", "*0.0"),
             title=f"提前期差 {f['lt_ratio']:.1f} 倍"),

        dict(name="5 优先级", ds=S,
             deps=[("string", "priority", "dimension", "nominal", "none:priority:nk", "None", "nominal"),
                   ("integer", "one", "measure", "quantitative", "sum:one:qk", "Sum", "quantitative")],
             rows=f"[{S}].[none:priority:nk]", cols=f"[{S}].[sum:one:qk]", mark="Bar",
             encodings=[("text", f"[{S}].[sum:one:qk]")], labels=True,
             sort=(f"[{S}].[none:priority:nk]", f"[{S}].[sum:one:qk]", "DESC"),
             title=f"P1 有 {f['p1']} 个 SKU"),

        dict(name="6 覆盖散点", ds=S,
             deps=[("real", "demand_mean", "measure", "quantitative", "avg:demand_mean:qk", "Avg", "quantitative"),
                   ("real", "coverage_mean", "measure", "quantitative", "avg:coverage_mean:qk", "Avg", "quantitative"),
                   ("string", "sku", "dimension", "nominal", "none:sku:nk", "None", "nominal"),
                   ("string", "priority", "dimension", "nominal", "none:priority:nk", "None", "nominal")],
             rows=f"[{S}].[avg:coverage_mean:qk]", cols=f"[{S}].[avg:demand_mean:qk]",
             mark="Circle",
             encodings=[("color", f"[{S}].[none:priority:nk]"),
                        ("lod", f"[{S}].[none:sku:nk]")],
             axis_titles={"cols": "日均需求", "rows": "平均覆盖"},
             title="覆盖率与需求无关"),
    ]

    twbkit.configure(BI, os.path.join(ROOT, "SupplyChain_Tableau105.twb"),
                     {F_DS: F_CAP, S_DS: S_CAP})
    out = twbkit.build(
        datasources=[
            dict(ds=F_DS, ts=F_TS, cap=F_CAP, csv="supply_fact.csv", fields=fact_fields),
            dict(ds=S_DS, ts=S_TS, cap=S_CAP, csv="supply_sku.csv", fields=sku_fields,
                 palette=(f"[none:priority:nk]",
                          [("P1", twbkit.C_RED), ("P2", twbkit.C_NAVY),
                           ("P3", twbkit.C_BLUE), ("P4", twbkit.C_LBLUE)])),
        ],
        sheets=sheets,
        dash=dict(name="供应链补货总览", title="供应链补货 · 5 仓 50 SKU", ncols=3, nrows=2),
    )
    twbkit.validate(out)


# ------------------------------------------------------------------
# 3. PowerBI 度量值定义（DAX 文本，原样保留）
# ------------------------------------------------------------------
MEASURES = """
	measure '收入' = SUM('supply_fact'[rev_m])
		formatString: $#.0

	measure '销量' = SUM('supply_fact'[units_sold])
		formatString: #,0

	measure '覆盖中位' = MEDIAN('supply_fact'[inventory_coverage])
		formatString: 0.00

	measure '缺货率' = AVERAGE('supply_fact'[stockout_flag])
		formatString: 0.0%

	measure '促销提升' =
			DIVIDE(CALCULATE(AVERAGE('supply_fact'[units_sold]), 'supply_fact'[promo_group] = "促销日"),
			       CALCULATE(AVERAGE('supply_fact'[units_sold]), 'supply_fact'[promo_group] = "平日")) - 1
		formatString: 0.0%
"""


def build_pbip(f):
    K.configure(proj="SupplyChain", logical_id="a1b2c3d4-1111-4a2b-9c3d-000000000005",
                root=ROOT, csv_dir=BI)
    K.build_semantic_model(tables=[
        dict(name="supply_fact", csv="supply_fact.csv", measures=MEASURES,
             desc="日级需求明细：91,250 行 = 250 个 SKU-仓 × 365 天",
             money_cols=["rev_m"], pct_cols=[]),
        # supply_sku 不进 Power BI 模型：它是 SKU-仓 粒度（250 行），和日级明细
        # 粒度不同、建不上关系，也没有视觉引用 → 模型里是一张没连线的空表。
        # CSV 保留（Tableau 侧在用）。
    ])

    F, S = "supply_fact", "supply_sku"
    vs = [
        K.textbox("v100", 16, 12, 1248, 36, "供应链补货 · 覆盖与促销"),
        K.visual("v101", "card", {"Values": [K.meas_ref(F, "收入")]},
                 16, 64, 303, 96, "收入（百万）", accent="primary", panel=True),
        K.visual("v102", "card", {"Values": [K.meas_ref(F, "覆盖中位")]},
                 331, 64, 303, 96, "覆盖中位（周）", accent="primary", panel=True),
        K.visual("v103", "card", {"Values": [K.meas_ref(F, "缺货率")]},
                 646, 64, 303, 96, "缺货率", accent="primary", panel=True),
        K.visual("v104", "card", {"Values": [K.meas_ref(F, "促销提升")]},
                 961, 64, 303, 96, "促销提升", accent="alert", panel=True),
        K.visual("v105", "clusteredColumnChart",
                 {"Category": [K.col_ref(F, "cov_bucket")],
                  "Y": [K.agg_ref(F, "units_sold", 5)]},
                 16, 172, 764, 532, "42% 落在 1–1.5 周",
                 accent="primary", panel=True, labels=True),
        K.visual("v106", "clusteredColumnChart",
                 {"Category": [K.col_ref(F, "promo_group")],
                  "Y": [K.agg_ref(F, "units_sold", 1)]},
                 792, 172, 472, 532, "促销拉高需求 27.7%",
                 accent="alert", panel=True, labels=True),
    ]
    K.build_report(pages=[("ReportSection1", "1 覆盖与促销", vs)])
    K.validate()


# ------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------
def main():
    print("[1/4] 数据准备 ...")
    fact, sku, f = prep()
    print(f"  {f['rows']:,} 行 / {f['skus']} SKU / {f['whs']} 仓 / "
          f"收入 ${f['rev']:.2f}M / 销量 {f['units']:,}")
    print(f"  覆盖中位 {f['cov_med']:.2f} 周 | {f['cov_mode']} 占 {f['cov_mode_pct']:.1f}% "
          f"| <1 周 {f['cov_low']:.1f}%")
    print(f"  促销 {f['promo']:.2f} vs 平日 {f['plain']:.2f} = +{f['lift']:.1f}% "
          f"| 缺货率 {f['stockout']:.1f}%")
    print(f"  低于再订货点的 SKU {f['below_rop']:.1f}% | 安全库存中位 {f['ss_med']:.1f} "
          f"| 距 ROP 中位 {f['gap_med']:.0f}")
    print(f"  提前期 {f['lt_min']:.1f}~{f['lt_max']:.1f} 天（{f['lt_ratio']:.1f} 倍）"
          f" | 各仓销量最大最小差 {(f['wh_ratio'] - 1) * 100:.1f}%")

    print("[2/4] 导出 CSV ...")
    p1 = csvkit.export(fact, os.path.join(BI, "supply_fact.csv"))
    p2 = csvkit.export(sku, os.path.join(BI, "supply_sku.csv"))
    csvkit.check(p1); csvkit.check(p2)

    print("\n[3/4] Tableau 工作簿 ...")
    build_tableau(fact, sku, f)

    print("\n[4/4] PowerBI 工程 ...")
    build_pbip(f)
    print("\n完成。")


if __name__ == "__main__":
    main()
