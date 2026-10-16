# Supply Chain Replenishment Optimization — Analysis Plan

**Business direction**: retail/manufacturing supply-chain inventory optimization. Stockout = lost orders and revenue; overstocking = tied-up capital. This project analyzes inventory health, demand volatility, and replenishment strategy, and gives actionable safety-stock / reorder-point recommendations.

## 1. Honest Data Boundaries (stated first)
This dataset is **synthetic grid data** (50 SKUs × 5 warehouses × 365 days = 91,250 rows, full year 2024, no missing). Two key limits:
- **Stockout_Flag is all 0** (no stockout positive samples) → **stockout classification is not possible** (supervised stockout prediction is meaningless).
- **Demand_Forecast correlates 0.949 with Units_Sold** → the demand column almost already gives the answer; retraining a demand forecast would be self-deception.
Therefore this project **does not fake prediction**; instead it does **replenishment decision support**: using demand volatility, lead time, and inventory vs reorder point, identify SKU/warehouse combinations with unreasonable inventory and give optimization directions.

## 2. Analysis Purpose
- Identify SKUs/warehouses **too close to reorder point** (near-stockout) and **far above demand** (overstocking).
- Cluster SKUs by **demand volatility** to set differentiated replenishment strategy (high volatility → high safety stock).
- Given **lead time**, give reorder-point / safety-stock recommendations.

## 3. Analysis Metrics
- Inventory coverage `Inventory_Level / Reorder_Point`.
- Share near reorder point (coverage ≤1.2), share with surplus (coverage ≥2).
- Demand volatility `CV = std/mean` (SKU level).
- Lead-time distribution, Order_Quantity vs daily demand.
- Region/warehouse inventory distribution, promotion effect on sales.

## 4. Modeling (ML + statistics; principle/why/how)
- **Demand-volatility clustering**: KMeans cluster SKUs by demand features (mean/volatility/seasonality) → strategy groups.
- **Safety stock / reorder point**: statistical formula `SS = z × σ_L`, `ROP = d × L + SS` (explain the principle).
- No stockout classification (no positive samples); no DL (tabular, clustering, needs explanation; trees/statistics are enough).

## 5. Visualizations (each chart Insight+Action)
- Inventory coverage distribution (near-stockout / surplus).
- Demand-volatility clustering (strategy groups).
- Lead-time distribution.
- Inventory vs reorder point scatter/bars.
- Regional inventory distribution.

## 6. Recommendations (landing)
- Raise safety stock for high-volatility SKUs.
- Replenish earlier and raise reorder point for long-lead-time SKUs.
- De-stock surplus SKUs to free capital.
- Differentiate replenishment by warehouse/region.
