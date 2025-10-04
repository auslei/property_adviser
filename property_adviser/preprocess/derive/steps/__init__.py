"""Step registry for derive engine."""
from __future__ import annotations

from typing import Dict, Type

from .base import DeriveStep
from .simple import SimpleStep
from .aggregate import AggregateStep
from .time_aggregate import TimeAggregateStep
from .join import JoinStep
from .binning import BinStep
from .rolling import RollingStep


STEP_REGISTRY: Dict[str, Type[DeriveStep]] = {
    SimpleStep.step_type: SimpleStep,
    AggregateStep.step_type: AggregateStep,
    TimeAggregateStep.step_type: TimeAggregateStep,
    JoinStep.step_type: JoinStep,
    BinStep.step_type: BinStep,
    RollingStep.step_type: RollingStep,
}


def get_step_class(step_type: str) -> Type[DeriveStep]:
    step_type_lower = step_type.lower()
    if step_type_lower not in STEP_REGISTRY:
        raise ValueError(f"Unsupported derive step type '{step_type}'.")
    return STEP_REGISTRY[step_type_lower]


__all__ = ["get_step_class", "STEP_REGISTRY"]
