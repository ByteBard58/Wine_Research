import sys
from pathlib import Path
import joblib
import numpy as np

# Where the model artifacts are expected to live (same folder as this script)
HERE = Path(__file__).parent
MODEL_FILE = HERE / "wine_pipeline.joblib"
FEATURES_FILE = HERE / "wine_features.joblib"
ALL_FEATURES_FILE = HERE / "wine_all_features.joblib"

CLASS_MAP = {0: "Low quality (3-4)", 1: "Medium quality (5-6)", 2: "High quality (7-8)"}


def load_artifacts():
    """Load the saved pipeline and feature list. Exit with a helpful message if
    artifacts are missing."""
    if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
        print("Could not find the required model artifacts.")
        print(f"Expected files:\n  {MODEL_FILE}\n  {FEATURES_FILE}")
        print("Run `wine_fit.py` to train and save the pipeline first.")
        sys.exit(2)

    pipeline = joblib.load(MODEL_FILE)
    # Prefer the full 11-feature list if available so the CLI can ask for all
    # original features. Fall back to the selected features file for
    # backward-compatibility.
    if ALL_FEATURES_FILE.exists():
        features = joblib.load(ALL_FEATURES_FILE)
    else:
        features = joblib.load(FEATURES_FILE)
    return pipeline, features


def prompt_for_features(feature_names):
    """Prompt the user for numeric values for each feature name. Returns a 2D
    numpy array with shape (1, n_features) suitable for the pipeline's
    predict method."""
    values = []
    print("\nEnter feature values (numeric). If you don't know a value, press Enter to use NaN and the pipeline's imputer will handle it.")
    for name in feature_names:
        while True:
            raw = input(f"  {name}: ").strip()
            if raw == "":
                # Let imputer handle missing value
                values.append(np.nan)
                break
            try:
                val = float(raw)
                values.append(val)
                break
            except ValueError:
                print("    Invalid number, please enter a numeric value or press Enter to skip.")
    arr = np.array(values, dtype=float).reshape(1, -1)
    return arr


def main():
    pipeline, feature_names = load_artifacts()

    print("\nWine quality classifier CLI")
    print("Classification threshold mapping:\n  class 0 -> original quality 3,4\n  class 1 -> original quality 5,6\n  class 2 -> original quality 7,8\n")

    print("Selected features used by the trained model in order:")
    for i, f in enumerate(feature_names, start=1):
        print(f"  {i}. {f}")

    user_array = prompt_for_features(feature_names)

    # The pipeline includes imputation, scaling, feature selector and model
    pred = pipeline.predict(user_array)
    pred_proba = None
    if hasattr(pipeline.named_steps.get("model"), "predict_proba"):
        try:
            pred_proba = pipeline.predict_proba(user_array)
        except Exception:
            pred_proba = None

    cls = int(pred[0])
    print("\nPrediction result:")
    print(f"  Predicted class: {cls} -> {CLASS_MAP.get(cls, 'Unknown')}")
    if pred_proba is not None:
        probs = pred_proba[0]
        print("  Class probabilities:")
        for i, p in enumerate(probs):
            print(f"    class {i}: {p:.3f}")

    print("\nNotes:")
    print(" - The mapping from original numeric quality to class is: 3/4->0, 5/6->1, 7/8->2.")
    print(" - If you provided blank values, the model used its imputer to fill them.")


if __name__ == '__main__':
    main()
