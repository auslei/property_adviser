import pytest
import pandas as pd

from property_adviser.preprocess.preprocess_derive import extract_street, build_segments


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("112 BRUNSWICK ROAD", "Brunswick Road"),
        ("10/6-10 Creek Road", "Creek Road"),
        ("105/435-439 Whitehorse Road", "Whitehorse Road"),
        ("G01/569 Whitehorse Road", "Whitehorse Road"),
        ("UNIT 4/14-16 Mcghee Avenue", "Mcghee Avenue"),
    ],
)
def test_extract_street_removes_unit_numbers(raw, expected):
    cfg = {"unknown_value": "Unknown"}
    assert extract_street(raw, cfg) == expected


def test_future_target_derivations_are_config_driven():
    df = pd.DataFrame({
        'saleYearMonth': [202301, 202302, 202303, 202304],
        'suburb': ['A'] * 4,
        'current_price_median': [100.0, 102.0, 104.0, 106.0],
        'salePrice': [101.0, 103.0, 105.0, 107.0],
    })

    config = {
        'grouping': {
            'enabled': True,
            'keys': ['suburb'],
        },
        'month_column': 'saleYearMonth',
        'aggregations': {
            'enabled': True,
            'metrics': [
                {'name': 'current_price_median', 'source': 'current_price_median', 'agg': 'mean'},
            ],
        },
        'future_targets': [
            {
                'name': 'price_future_1m',
                'source': 'salePrice',
                'agg': 'mean',
                'horizon': 1,
                'window': 1,
                'min_periods': 1,
                'drop_na': False,
                'base_column': 'current_price_median',
                'derived': [
                    {'type': 'delta'},
                    {'type': 'diff'},
                ],
                'smooth': {
                    'window': 2,
                    'min_periods': 1,
                    'base_column': 'current_price_median',
                    'derived': [
                        {'type': 'delta'},
                        {'type': 'diff'},
                    ],
                },
            }
        ],
    }

    segments, _ = build_segments(df, config)
    expected_cols = {
        'price_future_1m',
        'price_future_1m_delta',
        'price_future_1m_diff',
        'price_future_1m_smooth',
        'price_future_1m_smooth_delta',
        'price_future_1m_smooth_diff',
    }
    assert expected_cols.issubset(set(segments.columns))
    assert segments['price_future_1m_delta'].notna().any()
    assert segments['price_future_1m_smooth_delta'].notna().any()
