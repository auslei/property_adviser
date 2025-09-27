import pandas as pd

from property_adviser.feature import compute as compute_mod


def test_compute_feature_scores_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "target": [100000, 120000, 150000, 200000, 210000],
            "bed": [2, 3, 3, 4, 5],
            "bath": [1, 2, 2, 3, 3],
            "propertyType": ["House", "Unit", "House", "House", "Unit"],
        }
    )

    scores = compute_mod.compute_feature_scores(
        df=df,
        target="target",
        exclude={"bath"},
        mi_rs=42,
    )

    assert "bed" in scores
    assert "propertyType" in scores
    assert "bath" not in scores  # excluded

    bed_metrics = scores["bed"]
    assert 0.0 <= bed_metrics["pearson_abs"] <= 1.0
    assert 0.0 <= bed_metrics.get("mutual_info", 0.0) <= 1.0

    type_metrics = scores["propertyType"]
    assert 0.0 <= type_metrics["eta"] <= 1.0
