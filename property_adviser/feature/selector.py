from __future__ import annotations
from typing import Dict, List, Tuple

def select_top_features(
    scores: Dict[str, Dict[str, float]],
    *,
    correlation_threshold: float
) -> List[Dict[str, Dict[str, float]]]:
    """
    Keep a feature if its best metric >= threshold.
    Input:  {feature: {metric: score, ...}}
    Output: [{feature: {best_metric: score}}, ...] sorted by score desc.
    """
    kept: List[Tuple[str, str, float]] = []
    for col, metrics in scores.items():
        if not metrics:
            continue
        best_metric, best_score = max(metrics.items(), key=lambda kv: kv[1])
        if best_score >= correlation_threshold:
            kept.append((col, best_metric, best_score))

    kept.sort(key=lambda t: t[2], reverse=True)
    return [{col: {metric: score}} for col, metric, score in kept]