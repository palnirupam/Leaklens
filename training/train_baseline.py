"""Baseline training pipeline for the LeakLens demo.

Uses the Bank Marketing dataset. Because the demo task is PRE-CALL prediction,
the ``duration`` column is unavailable at prediction time and is excluded from
the valid baseline by default (pass ``--include-duration`` to compare).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DEMO_CSV = ROOT / "data" / "demo" / "bank_marketing.csv"

TARGET = "y"
DURATION = "duration"


def load_demo() -> pd.DataFrame:
    path = DEMO_CSV
    if not path.exists():
        # build demo if missing
        import subprocess
        import sys
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_demo_dataset.py")], check=True)
    return pd.read_csv(path)


def build_pipeline(include_duration: bool) -> tuple[ColumnTransformer, list[str]]:
    df = load_demo()
    features = [c for c in df.columns if c != TARGET]
    if not include_duration and DURATION in features:
        features.remove(DURATION)
    numeric = df[features].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median")), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
        ]
    )
    return preprocessor, features


def main() -> int:
    parser = argparse.ArgumentParser(description="LeakLens baseline training")
    parser.add_argument("--include-duration", action="store_true", help="include 'duration' feature (leaky for pre-call task)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_demo()
    preprocessor, features = build_pipeline(args.include_duration)
    X = preprocessor.fit_transform(df[features])
    y = (df[TARGET] == "yes").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )
    model = LogisticRegression(max_iter=2000, random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Baseline trained on {X.shape[0]} rows x {features} features")
    print(f"  (duration included: {args.include_duration})")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 (pos): {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
