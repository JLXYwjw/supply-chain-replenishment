# -*- coding: utf-8 -*-
"""供应链日销量时序预测 baseline 对比 —— 用"模型族递进"证明已系统评估过时序建模,
   但该数据主产出仍是结构诊断(安全库存/再订货点), 时序仅作辅助验证。

基线族(由简到繁):
  1. Naive(persistence)       : y_hat_t = y_{t-1}        —— 最简单基线
  2. Naive-7(上周同天)         : y_hat_t = y_{t-7}        —— 捕捉周季节性
  3. ARIMA(1,1,1)             : statsmodels 拟合差分+自回归+移动平均
  4. ML + 滞后特征             : 随机森林回归(500树, max_depth=10), 特征=lag1/lag7/lag_avg7/周几/促销
训练细节: 时间顺序切分(前80%训练, 后20%测试) —— 严格防未来信息泄漏;
          RF 直接回归, 评估 RMSE 与 MAE(测试集), 比较复杂模型相对 Naive 的实际增益。
目标: units_sold(日销量), 取代表性 SKU_1 × WH_1 的 365 天连续序列为主。
输出: ml_result_ts.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

OUT = Path(__file__).resolve().parent.parent / "data" / "output"
clean = pd.read_parquet(OUT / "supply_clean.parquet")


def split_ts(y, test_frac=0.2):
    n = len(y)
    cut = int(n * (1 - test_frac))
    return cut, y[:cut], y[cut:]


def rmse_mae(yt, yp):
    return (round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
            round(float(mean_absolute_error(yt, yp)), 3))


sub = clean[(clean["sku"] == "SKU_1") & (clean["wh"] == "WH_1")].sort_values("Date").reset_index(drop=True)
y = sub["units_sold"].values
cut, ytr, yte = split_ts(y)
n_te = len(yte)

r = {}
r["series"] = "SKU_1 × WH_1"
r["n_train"] = int(len(ytr))
r["n_test"] = int(n_te)

# 1 Naive (persistence)
p_naive = np.full(n_te, ytr[-1])

# 2 Naive-7 (上周同天): 每个测试点用其 7 天前的真实值
p_naive7 = np.array([y[cut + t - 7] for t in range(n_te)])

# 3 ARIMA(1,1,1)
model = ARIMA(ytr, order=(1, 1, 1)).fit()
p_arima = model.forecast(n_te)

# 4 ML + 滞后特征
df = sub.copy()
df["lag1"] = df["units_sold"].shift(1)
df["lag7"] = df["units_sold"].shift(7)
df["lag_avg7"] = df["units_sold"].shift(1).rolling(7).mean()
feat_cols = ["lag1", "lag7", "lag_avg7", "weekday", "is_promo"]
Xf = df[feat_cols].fillna(0).values
yf = df["units_sold"].values
rf = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(Xf[:cut], yf[:cut])
p_rf = rf.predict(Xf[cut:])

r["naive_rmse"], r["naive_mae"] = rmse_mae(yte, p_naive)
r["naive7_rmse"], r["naive7_mae"] = rmse_mae(yte, p_naive7)
r["arima_rmse"], r["arima_mae"] = rmse_mae(yte, p_arima)
r["ml_rmse"], r["ml_mae"] = rmse_mae(yte, p_rf)

# 相对 Naive 的增益(越小越好)
r["arima_gain_vs_naive_pct"] = round((r["arima_rmse"] - r["naive_rmse"]) / r["naive_rmse"] * 100, 2)
r["ml_gain_vs_naive_pct"] = round((r["ml_rmse"] - r["naive_rmse"]) / r["naive_rmse"] * 100, 2)
r["best_model"] = min(["naive", "naive7", "arima", "ml"],
                      key=lambda k: r[f"{k}_rmse"])

# 主序列均值水平(评估增益时看绝对量级)
r["y_mean"] = round(float(y.mean()), 2)
r["y_std"] = round(float(y.std()), 2)
r["train_info"] = {"cut": "80/20 time-ordered", "arima_order": "(1,1,1)",
                   "ml_model": "RandomForestRegressor(n=500, max_depth=10)",
                   "ml_features": feat_cols, "metric": "RMSE & MAE on test"}

with open(OUT / "ml_result_ts.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False), flush=True)
