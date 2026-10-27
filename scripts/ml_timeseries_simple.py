# -*- coding: utf-8 -*-
"""
供应链日销量时序预测 baseline 对比 —— 基础可读版（ml_timeseries_simple.py）
==========================================================================

  取代表性 SKU_1 × WH_1 的 365 天日销量序列，按时间顺序 80%/20% 切分，
  用 4 个模型在测试集上比 RMSE/MAE：
    1. Naive（persistence）：明天=今天，最简单基线
    2. Naive-7：用上周同一天的值（抓周季节性）
    3. ARIMA(1,1,1)：差分+自回归+移动平均
    4. 随机森林：用 lag1/lag7/lag_avg7/周几/促销 做特征回归
  最后看哪个模型相对 Naive 提升最大。

诚实说明：这份数据的主产出是结构诊断（安全库存/再订货点），
时序预测只是辅助验证。结果写入 ml_result_ts.json。

运行方式（不带命令行参数）：
    python scripts/ml_timeseries_simple.py


English:
Daily-sales time-series forecasting baseline comparison (plain, readable version).

Takes the 365-day daily-sales series of the representative SKU_1 x WH_1,
splits it chronologically 80% / 20%, and compares 4 models on the test
set by RMSE / MAE:
  1. Naive (persistence): tomorrow = today, the simplest baseline
  2. Naive-7: use the value of the same day last week (captures weekly
     seasonality)
  3. ARIMA(1,1,1): differencing + autoregression + moving average
  4. Random forest: regression on lag1 / lag7 / lag_avg7 / weekday / promo
Finally it shows which model improves the most over Naive.

Honest note: the main output of this data is the structural diagnosis
(safety stock / reorder point); the time-series forecast is only an
auxiliary validation. Results go to ml_result_ts.json.

Run (no command-line arguments):
    python scripts/ml_timeseries_simple.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# ------------------------------------------------------------------
# 0. 路径配置
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"


# ------------------------------------------------------------------
# 1. 工具函数
# ------------------------------------------------------------------
def split_ts(y, test_frac=0.2):
    """按时间顺序把序列切成 (切分位置, 训练段, 测试段)。"""
    n = len(y)
    cut = int(n * (1 - test_frac))      # 前 80% 训练，后 20% 测试
    return cut, y[:cut], y[cut:]


def rmse_mae(yt, yp):
    """算 RMSE 和 MAE，都保留 3 位小数，返回 (rmse, mae)。"""
    rmse = round(float(np.sqrt(mean_squared_error(yt, yp))), 3)
    mae = round(float(mean_absolute_error(yt, yp)), 3)
    return rmse, mae


def main():
    # ---- 2. 读入行级干净表，挑出 SKU_1 × WH_1 ----
    clean = pd.read_parquet(OUT / "supply_clean.parquet")
    sub = clean[(clean["sku"] == "SKU_1") & (clean["wh"] == "WH_1")]
    sub = sub.sort_values("Date").reset_index(drop=True)
    y = sub["units_sold"].values

    # 时间顺序切分：前 80% 训练，后 20% 测试
    cut, ytr, yte = split_ts(y)
    n_te = len(yte)

    r = {}
    r["series"] = "SKU_1 × WH_1"
    r["n_train"] = int(len(ytr))
    r["n_test"] = int(n_te)

    # ---- 3. 模型 1：Naive（persistence）----
    # 测试集每个点都预测成训练集最后一天的值
    p_naive = np.full(n_te, ytr[-1])

    # ---- 4. 模型 2：Naive-7（上周同天）----
    # 拆成普通 for 循环：第 t 个测试点，用它 7 天前的真实值
    # （一行也能算：[y[cut+t-7] for t in range(n_te)]）
    p_naive7_list = []
    for t in range(n_te):
        p_naive7_list.append(y[cut + t - 7])
    p_naive7 = np.array(p_naive7_list)

    # ---- 5. 模型 3：ARIMA(1,1,1) ----
    model = ARIMA(ytr, order=(1, 1, 1)).fit()
    p_arima = model.forecast(n_te)

    # ---- 6. 模型 4：随机森林 + 滞后特征 ----
    df = sub.copy()

    # lag1 = 前一天销量
    df["lag1"] = df["units_sold"].shift(1)

    # lag7 = 上周同一天销量
    df["lag7"] = df["units_sold"].shift(7)

    # lag_avg7 = 前 7 天销量的均值（用普通循环逐行算：
    # 第 t 天的 lag_avg7 = mean(第 t-1 到 t-7 天销量)）
    sales = df["units_sold"].tolist()
    lag_avg7 = []
    for t in range(len(sales)):
        if t < 7:
            lag_avg7.append(np.nan)          # 前 7 天不够算，记缺失
        else:
            window = sales[t - 7:t]          # 第 t-7 到 t-1 天（共 7 个）
            lag_avg7.append(sum(window) / len(window))
    df["lag_avg7"] = lag_avg7

    # 特征列：lag1, lag7, lag_avg7, 周几, 是否促销；缺失填 0
    feat_cols = ["lag1", "lag7", "lag_avg7", "weekday", "is_promo"]
    Xf = df[feat_cols].fillna(0).values
    yf = df["units_sold"].values

    rf = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(Xf[:cut], yf[:cut])
    p_rf = rf.predict(Xf[cut:])

    # ---- 7. 计算各模型在测试集上的 RMSE / MAE ----
    r["naive_rmse"], r["naive_mae"] = rmse_mae(yte, p_naive)
    r["naive7_rmse"], r["naive7_mae"] = rmse_mae(yte, p_naive7)
    r["arima_rmse"], r["arima_mae"] = rmse_mae(yte, p_arima)
    r["ml_rmse"], r["ml_mae"] = rmse_mae(yte, p_rf)

    # ---- 8. 相对 Naive 的增益（负=比 Naive 好）----
    r["arima_gain_vs_naive_pct"] = round(
        (r["arima_rmse"] - r["naive_rmse"]) / r["naive_rmse"] * 100, 2)
    r["ml_gain_vs_naive_pct"] = round(
        (r["ml_rmse"] - r["naive_rmse"]) / r["naive_rmse"] * 100, 2)

    # ---- 9. 找 RMSE 最小的模型 ----
    # 拆成普通循环：逐模型比较 RMSE，记下最小的那个
    # （一行也能算：min([...], key=lambda k: r[f"{k}_rmse"])）
    model_names = ["naive", "naive7", "arima", "ml"]
    best_name = model_names[0]
    best_rmse = r[f"{best_name}_rmse"]
    for name in model_names[1:]:
        if r[f"{name}_rmse"] < best_rmse:
            best_rmse = r[f"{name}_rmse"]
            best_name = name
    r["best_model"] = best_name

    # ---- 10. 主序列的均值/标准差（评估增益时看绝对量级） ----
    r["y_mean"] = round(float(y.mean()), 2)
    r["y_std"] = round(float(y.std()), 2)
    r["train_info"] = {
        "cut": "80/20 time-ordered",
        "arima_order": "(1,1,1)",
        "ml_model": "RandomForestRegressor(n=500, max_depth=10)",
        "ml_features": feat_cols,
        "metric": "RMSE & MAE on test",
    }

    # ---- 11. 写出结果 ----
    with open(OUT / "ml_result_ts.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)

    print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
