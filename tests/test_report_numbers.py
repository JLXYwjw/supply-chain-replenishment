# -*- coding: utf-8 -*-
"""供应链报告数字一致性。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "output"
HTML = ROOT / "报告" / "供应链补货优化分析.html"


def _html():
    return HTML.read_text(encoding="utf-8")


def _m():
    return json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def test_coverage_match_html():
    m = _m()
    assert abs(m["coverage_le12_pct"] - 22.12) < 0.1
    assert "22.1%" in _html()
    assert abs(m["coverage_lt1_pct"] - 5.25) < 0.1
    assert "5.3%" in _html()


def test_promo_match_html():
    m = _m()
    assert abs(m["promo_lift_pct"] - 27.7) < 0.1
    assert "27.7%" in _html()


def test_lead_time_match_html():
    m = _m()
    assert abs(m["lead_time_mean"] - 7.98) < 0.1
    assert "8天" in _html()


def test_clusters_match_html():
    ml = json.loads((OUT / "ml_result.json").read_text(encoding="utf-8"))
    assert len(ml["clusters"]) == 3
    assert abs(ml["ss_median"] - 41.75) < 0.5
    assert "41.8" in _html()
    assert abs(ml["rop_median"] - 202.1) < 1
    assert "202.1" in _html()


def test_honest_boundary_in_html():
    h = _html()
    assert "缺货分类" in h and "需求" in h and "0.949" in h
