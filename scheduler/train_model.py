"""
Train Cost-Fidelity Prediction Model
-------------------------------------
Trains a gradient boosted regression model on the synthetic dataset
and saves it to cost_fidelity_model.pkl.

Run:
    python generate_training_data.py   # create dataset first
    python train_model.py              # train and save model

Output:
    cost_fidelity_model.pkl
    training_report.txt
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH  = Path(__file__).parent / "training_data.csv"
MODEL_PATH = Path(__file__).parent / "cost_fidelity_model.pkl"
REPORT_PATH= Path(__file__).parent / "training_report.txt"

from cost_fidelity_model import FEATURE_NAMES

FEATURE_COLS = FEATURE_NAMES
TARGET_COL = "post_transpilation_fidelity"

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Run: python generate_training_data.py first")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples from {DATA_PATH}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2")

    report = f"""
Cost-Fidelity Prediction Model Training Report
===============================================
WARNING: labels are SYNTHETIC (physics-informed formula + Gaussian noise).
These metrics measure how well the model reproduces that formula; they do
not measure prediction accuracy on real quantum hardware.

Dataset size  : {len(df)} samples
Features      : {FEATURE_COLS}
Target        : {TARGET_COL}

Test MAE      : {mae:.5f}   (mean absolute error in fidelity units)
Test R²       : {r2:.4f}
CV R² (5-fold): {cv.mean():.4f} ± {cv.std():.4f}

Feature importance (GBR):
"""
    gbr = model.named_steps["gbr"]
    importances = sorted(
        zip(FEATURE_COLS, gbr.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        report += f"  {name:<35} {imp:.4f}\n"

    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")

    abs_err = np.abs(y_test - y_pred)
    bundle = {
        "model": model,
        "feature_names": FEATURE_COLS,
        "label_source": "synthetic-physics-formula",
        "metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "p95_abs_error": float(np.percentile(abs_err, 95)),
            "cv_r2_mean": float(cv.mean()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
