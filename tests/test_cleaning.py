# -*- coding: utf-8 -*-
"""供应链清洗口径测试。"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "output"


def _clean():
    return pd.read_parquet(OUT / "supply_clean.parquet")


def test_clean_shape():
    df = _clean()
    assert len(df) == 91250
    for c in ["inventory_coverage", "revenue", "demand_vs_order", "is_promo"]:
        assert c in df.columns


def test_no_missing():
    df = _clean()
    assert df.isna().sum().sum() == 0


def test_stockout_all_zero():
    df = _clean()
    assert df["is_stockout"].sum() == 0


def test_sku_level():
    sku = pd.read_parquet(OUT / "sku_level.parquet")
    assert len(sku) == 250
    assert {"demand_cv", "reorder_point", "lead_time_days", "coverage_mean"} <= set(sku.columns)


def test_clusters():
    c = pd.read_parquet(OUT / "sku_clusters.parquet")
    assert len(c) == 250
    assert {"cluster", "ss", "rop", "priority"} <= set(c.columns)
