import argparse

from .config import PREPROCESS_DIR
from .data_tracking import load_preprocess_metadata, raw_data_has_changed
from .feature_selection import run_feature_selection
from .model_training import train_models
from .preprocess import preprocess



def main(force: bool = False) -> None:
    """
    Runs the entire data processing and model training pipeline.

    Args:
        force: If True, forces preprocessing even if raw data has not changed.
    """
    metadata = load_preprocess_metadata()
    need_preprocess = force or raw_data_has_changed(metadata)

    if need_preprocess:
        print("Starting preprocessing...")
        preprocess_path = preprocess()
        print(f"Preprocessed data stored at {preprocess_path}.")
    else:
        preprocess_path = PREPROCESS_DIR / "cleaned.parquet"
        if preprocess_path.exists():
            print("No new raw data detected. Using existing preprocessed dataset.")
        else:
            print("Preprocessed dataset missing. Running preprocessing now.")
            preprocess_path = preprocess()
            print(f"Preprocessed data stored at {preprocess_path}.")

    print("Running feature selection...")
    feature_info = run_feature_selection()
    print(f"Selected features: {feature_info['selected_features']}")

    print("Training models...")
    outcome = train_models()
    print(f"Best model: {outcome['best_model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end property price pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force preprocessing even if raw data has not changed",
    )
    args = parser.parse_args()
    main(force=args.force)
