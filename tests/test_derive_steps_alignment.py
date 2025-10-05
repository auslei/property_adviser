import pandas as pd
import numpy as np

from property_adviser.preprocess.derive.steps.base import DeriveContext
from property_adviser.preprocess.derive.config import StepSpec
from property_adviser.preprocess.derive.steps.aggregate import AggregateStep
from property_adviser.preprocess.derive.steps.time_aggregate import TimeAggregateStep
from property_adviser.preprocess.derive.steps.rolling import RollingStep
from property_adviser.preprocess.derive.steps.binning import BinStep
from property_adviser.preprocess.derive.steps.simple import SimpleStep
from property_adviser.preprocess.derive.steps.join import JoinStep


def _spec(step_id: str, step_type: str, cfg: dict) -> StepSpec:
    cfg = dict(cfg)
    cfg.setdefault("enabled", True)
    return StepSpec(id=step_id, type=step_type, enabled=True, raw=cfg)


def test_aggregate_step_alignment():
    df = pd.DataFrame({
        "suburb": ["A", "A", "B", "B", "B"],
        "price": [500, 550, 400, 420, 380],
    })
    ctx = DeriveContext(settings={})
    spec = _spec(
        "agg1",
        "aggregate",
        {"group_by": "suburb", "target": "price", "outputs": {"mean": "suburb_price_mean"}},
    )
    out = AggregateStep(spec).execute(df, ctx).frame

    assert len(out) == len(df)
    pd.testing.assert_index_equal(out.index, df.index)
    # Expected means: A → 525, B → 400
    assert out.loc[0, "suburb_price_mean"] == 525
    assert out.loc[2, "suburb_price_mean"] == 400


def test_time_aggregate_step_alignment():
    df = pd.DataFrame({
        "suburb": ["A", "A", "A", "B", "B"],
        "saleYearMonth": [202301, 202302, 202303, 202301, 202303],
        "price": [500, 550, 530, 400, 420],
    })
    ctx = DeriveContext(settings={})
    spec = _spec(
        "ta1",
        "time_aggregate",
        {
            "group_by": "suburb",
            "time_col": "saleYearMonth",
            "target": "price",
            "window": {"unit": "months", "past": 1, "future": 0, "include_current": True},
            "outputs": {"mean": "suburb_price_mean_1m"},
        },
    )
    out = TimeAggregateStep(spec).execute(df, ctx).frame
    assert len(out) == len(df)
    # Expected sequence: [500, 525, 540, 400, 410]
    expected = [500.0, 525.0, 540.0, 400.0, 410.0]
    np.testing.assert_allclose(out["suburb_price_mean_1m"].to_numpy(), np.array(expected), rtol=1e-6, atol=1e-6)


def test_rolling_step_alignment():
    df = pd.DataFrame({
        "suburb": ["A", "A", "A", "B", "B"],
        "saleYearMonth": [202301, 202302, 202303, 202301, 202303],
        "price": [500, 550, 530, 400, 420],
    })
    ctx = DeriveContext(settings={})
    spec = _spec(
        "roll1",
        "rolling",
        {
            "group_by": "suburb",
            "sort_by": "saleYearMonth",
            "target": "price",
            "window": 2,
            "config": {"min_periods": 1},
            "outputs": {"mean": "suburb_price_roll2"},
        },
    )
    out = RollingStep(spec).execute(df, ctx).frame
    assert len(out) == len(df)
    # For A: windows over 2 → [500], [500,550], [550,530]; means → [500, 525, 540]
    # For B: [400], [400,420] → [400, 410]
    expected = [500.0, 525.0, 540.0, 400.0, 410.0]
    np.testing.assert_allclose(out["suburb_price_roll2"].to_numpy(), np.array(expected), rtol=1e-6, atol=1e-6)


def test_bin_step_alignment():
    df = pd.DataFrame({"landSize": [150, 350, 1200, np.nan]})
    ctx = DeriveContext(settings={})
    spec = _spec(
        "bin1",
        "bin",
        {
            "source": "landSize",
            "output": "land_bucket",
            "method": "fixed",
            "config": {"edges": [200, 800], "labels": ["small", "medium", "large"], "fill_value": "Unknown"},
        },
    )
    out = BinStep(spec).execute(df, ctx).frame
    assert len(out) == len(df)
    assert out["land_bucket"].tolist() == ["small", "medium", "large", "Unknown"]


def test_simple_expr_alignment():
    df = pd.DataFrame({"bed": [2, 3, 4], "bath": [1, 2, 3]})
    ctx = DeriveContext(settings={})
    spec = _spec(
        "expr1",
        "simple",
        {"method": "expr", "output": "rooms", "expr": "bed + bath"},
    )
    out = SimpleStep(spec).execute(df, ctx).frame
    assert len(out) == len(df)
    assert out["rooms"].tolist() == [3, 5, 7]


def test_join_step_left_merge_alignment():
    left = pd.DataFrame({
        "suburb": ["A", "A", "B", "B"],
        "price": [500, 550, 400, 420],
    })
    right = pd.DataFrame({"suburb": ["A", "B"], "median_income": [80000, 75000]})
    ctx = DeriveContext(settings={})
    spec = _spec(
        "join1",
        "join",
        {"right": right, "on": ["suburb"], "how": "left"},
    )
    out = JoinStep(spec).execute(left, ctx).frame
    assert len(out) == len(left)
    assert out["median_income"].tolist() == [80000, 80000, 75000, 75000]

