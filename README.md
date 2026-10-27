# Supply Chain — Replenishment Optimization
![CI](https://github.com/JLXYwjw/supply-chain-replenishment/actions/workflows/ci.yml/badge.svg)


**The business problem.** Retail/manufacturing inventory sits between two failure
modes: **stock out (lost revenue)** and **over-stocking (tied-up capital)**. On
real inventory data (91,250 rows, 50 SKUs × 5 warehouses × 365 days, 2024), this
project identifies *which* SKU/warehouse to watch and *how* to replenish — safety
stock and reorder point optimized by demand volatility and lead time.

## Honest data boundary (read first)

This dataset is a **synthetic grid** (50×5×365, no missing values):
- **Stockout_Flag is all 0** — no positive samples → a stockout classifier is
  impossible here.
- **Demand_Forecast correlates 0.949 with Units_Sold** — demand is essentially
  already given; re-training a demand forecast would be self-deception.

So I do **no fake prediction**. This project delivers **inventory-health +
replenishment decision support** instead, and says so plainly.

## What it does

- Derives inventory coverage (`Inventory/Reorder_Point`), daily demand, demand
  volatility (CV), and promotion lift.
- **Unsupervised clustering** (KMeans, no labels) on demand/volatility/
  lead-time/coverage → groups by lead time (short/mid/long ≈ 4/7/12 days).
- **Safety stock & reorder point** via the standard formulas `SS = z·σ·√L`,
  `ROP = d·L + SS` — explained, not magic.
- Ranks replenishment priority by the `inventory − ROP` gap.

## Charts

![Coverage distribution](assets/chart_EN_01.png)
![Promotion lift](assets/chart_EN_02.png)
![Demand volatility](assets/chart_EN_03.png)
![Lead-time tiers](assets/chart_EN_04.png)
![Replenishment priority](assets/chart_EN_05.png)

> Full interactive report: open `index.html` (English) or `index.zh-CN.html` (Chinese) after cloning — GitHub doesn't render HTML.

## Key verified numbers

- 22.1% of days have coverage ≤1.2 (near reorder point); 5.3% already below
- Median coverage 1.53 (healthy overall; stockouts are sporadic, not systemic)
- Promotions lift sales +27.7% (24.9 vs 19.5 units/day)
- All SKU×warehouse have CV>0.4 (high volatility); median CV 0.45
- SS median 41.8; ROP median 202.1; 0% below ROP on annual average

## Models / why not DL

KMeans clustering + statistical safety-stock formulas — tabular, 250 groups, no
labels; interpretable and directly deployable. **No DL**: no labels, needs to be
explainable to inventory teams, and tree/statistical methods already fit.

## Data source

Raw data lives in `data/raw/` (excluded from git — after cloning, download it yourself).

| File | What it is | Kaggle download |
|---|---|---|
| `supply_chain_dataset1.csv` | SKU×warehouse×day grid, 91,250 rows (2024) | `kaggle datasets download -d ziya07/high-dimensional-supply-chain-inventory-dataset -p data/raw --unzip` |

Kaggle page: https://www.kaggle.com/datasets/ziya07/high-dimensional-supply-chain-inventory-dataset

```bash
pip install kaggle
kaggle datasets download -d ziya07/high-dimensional-supply-chain-inventory-dataset -p data/raw --unzip
# requires a kaggle.json / KAGGLE_USERNAME + KAGGLE_KEY
```

## Run

```bash
pip install -r requirements.txt
python scripts/clean_data_simple.py
python scripts/analyze_simple.py
python scripts/ml_analysis_simple.py
pytest tests/ -q
```

## Layout

```
scripts/ clean_data.py / analyze.py / ml_analysis.py
tests/   test_cleaning.py, test_report_numbers.py
index.html   # English interactive report
index.zh-CN.html  # Chinese report (indigo-violet theme, each chart Insight+Action)
```

— Jiawei Wang

## CI / Quality Gate

Every push or pull request runs a [GitHub Actions](.github/workflows/ci.yml) workflow: a quality gate checks that English docs contain no Chinese, every referenced chart exists, and the bilingual README + index are all present; then `pytest` runs the test suite.

![CI](https://github.com/JLXYwjw/supply-chain-replenishment/actions/workflows/ci.yml/badge.svg)
