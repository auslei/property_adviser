from __future__ import annotations

import argparse
from typing import Optional, Sequence

from property_adviser.core.app_logging import setup_logging
from property_adviser.train.promotion import PromotionError, promote_models


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Promote trained models into the final bundle directory")
    parser.add_argument(
        "--report",
        type=str,
        help="Path to a training report. If omitted, the latest is auto-discovered under models/*/training_report.json or legacy models/training_report_*.json",
    )
    parser.add_argument(
        "--target",
        action="append",
        help="Target name or target column to promote (can be passed multiple times)",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Promote every target listed in the training report",
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="Directory that will receive the promoted artefacts (default: models/model_final)",
    )
    parser.add_argument(
        "--copy-scores",
        action="store_true",
        help="Copy model_scores CSVs alongside promoted bundles",
    )
    parser.add_argument(
        "--activate",
        type=str,
        help="Target to mark as the active deployment bundle (defaults to single promoted target)",
    )
    parser.add_argument(
        "--no-best-per-window",
        dest="best_per_window",
        action="store_false",
        help="Promote all selected targets even if they share a forecast window (default groups by window)",
    )
    parser.set_defaults(best_per_window=True)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")

    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)

    try:
        summary = promote_models(
            report_path=args.report,
            destination=args.destination,
            targets=args.target,
            include_all_targets=args.all_targets,
            copy_scores=args.copy_scores,
            activate_target=args.activate,
            best_per_window=args.best_per_window,
        )
    except PromotionError as exc:
        raise SystemExit(str(exc)) from exc

    promoted = summary.get("promotions", [])
    if not promoted:
        print("No models were promoted.")
        return

    for entry in promoted:
        print(
            f"Promoted {entry['target_name']} ({entry['model']}) → {entry['promoted_model']}"
        )
    activated = summary.get("activated_target")
    if activated:
        print(f"Active bundle: {activated}")


if __name__ == "__main__":
    main()
