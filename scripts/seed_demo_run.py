"""Seed real investigations via the running API (used for browser verification)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import httpx

sys.path.insert(0, "backend")
from app.data.synthetic import generate_case  # noqa: E402

BASE = "http://127.0.0.1:8000"
client = httpx.Client(timeout=120.0)


def main() -> int:
    cases = [
        generate_case("target_derived_feature", seed=21),
        generate_case("duplicate_contamination", seed=22),
        generate_case("temporal_leakage", seed=23),
        generate_case("clean", seed=24),
    ]
    num = []
    with tempfile.TemporaryDirectory() as td:
        for c in cases:
            p = os.path.join(td, c.name + ".csv")
            c.df.to_csv(p, index=False)
            with open(p, "rb") as f:
                data = {"target_column": c.target, "random_seed": "42"}
                if c.timestamp_column:
                    data["timestamp_column"] = c.timestamp_column
                if c.entity_column:
                    data["entity_column"] = c.entity_column
                if c.unavailable_at_prediction:
                    data["unavailable_at_prediction"] = json.dumps(c.unavailable_at_prediction)
                r = client.post(
                    f"{BASE}/api/investigations",
                    data=data,
                    files={"file": (os.path.basename(p), f, "text/csv")},
                )
                r.raise_for_status()
                num.append(r.json()["id"])

    for inv_id in num:
        for _ in range(200):
            s = client.get(f"{BASE}/api/investigations/{inv_id}").json()
            if s["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        print(inv_id, s["status"], "findings:", s["finding_count"], "exp:", s["experiment_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
