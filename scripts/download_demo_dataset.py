"""Download the UCI Bank Marketing dataset (id 222) for the LeakLens demo.

If the download fails (no network, package issues), a synthetic demo dataset is
written instead so the project never depends entirely on internet access.

Output:
  data/demo/bank_marketing.csv   (feature + target table)
  data/demo/bank_marketing_meta.json (metadata / fallback flag)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"

TARGET_COL = "y"  # UCI target column name
DURATION_COL = "duration"  # unavailable at prediction time (pre-call task)


def _write_demo(df: pd.DataFrame, meta: dict) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DEMO_DIR / "bank_marketing.csv"
    df.to_csv(csv_path, index=False)
    (DEMO_DIR / "bank_marketing_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[demo] wrote {csv_path} ({df.shape[0]} rows x {df.shape[1]} cols)")


def _synthetic_fallback() -> pd.DataFrame:
    """Produce a small, plausible tabular marketing-style dataset."""
    rng = __import__("numpy").random.default_rng(7)
    n = 800
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n).astype(int),
            "job": rng.choice(
                ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"],
                n,
            ),
            "marital": rng.choice(["divorced", "married", "single"], n),
            "education": rng.choice(
                ["primary", "secondary", "tertiary", "unknown"], n
            ),
            "default": rng.choice(["no", "yes"], n, p=[0.9, 0.1]),
            "balance": rng.integers(-3000, 60000, n).astype(int),
            "housing": rng.choice(["no", "yes"], n),
            "loan": rng.choice(["no", "yes"], n),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n),
            "day": rng.integers(1, 32, n).astype(int),
            "month": rng.choice(
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                 "sep", "oct", "nov", "dec"],
                n,
            ),
            "campaign": rng.integers(1, 30, n).astype(int),
            "pdays": rng.integers(-1, 100, n).astype(int),
            "previous": rng.integers(0, 20, n).astype(int),
            "poutcome": rng.choice(["failure", "other", "success", "unknown"], n),
            DURATION_COL: rng.integers(1, 2000, n).astype(int),
            TARGET_COL: rng.choice(["no", "yes"], n, p=[0.88, 0.12]),
        }
    )
    return df


def main() -> int:
    # Try the official UCI downloader.
    try:
        from ucimlrepo import fetch_ucirepo
        repo = fetch_ucirepo(id=222)
        X = repo.data.features
        y = repo.data.targets
        df = pd.concat([X, y], axis=1)
        meta = {
            "source": "ucimlrepo",
            "uci_id": 222,
            "name": "Bank Marketing (UCI)",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": TARGET_COL,
            "duration_column": DURATION_COL,
            "fallback": False,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] UCI download failed ({exc!r}); using synthetic fallback.")
        df = _synthetic_fallback()
        meta = {
            "source": "synthetic_fallback",
            "uci_id": None,
            "name": "Bank Marketing (synthetic fallback)",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": TARGET_COL,
            "duration_column": DURATION_COL,
            "fallback": True,
        }

    _write_demo(df, meta)
    print(f"[demo] target column: {TARGET_COL}; prediction-time-unavailable: {DURATION_COL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
