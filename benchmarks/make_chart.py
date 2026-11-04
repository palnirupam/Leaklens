"""Generate a dependency-free SVG chart from benchmark results.

Reads ``benchmark_results.json`` and writes ``benchmark_chart.svg``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "benchmark_results.json"
OUT = ROOT / "benchmark_chart.svg"


def _bars(data: dict, keys: list[str]) -> str:
    baseline = data["rule_based_baseline"]
    adaptive = data["leaklens_adaptive"]
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1",
              "false_positive_rate_clean": "FPR (clean)", "root_cause_accuracy": "Root-cause acc"}
    bar_w = 34
    gap = 72
    y0 = 240
    max_v = 1.0
    scale = 190.0
    parts = []
    x = 60
    for key in keys:
        v1 = baseline[key]
        v2 = adaptive[key]
        h1 = v1 * scale
        h2 = v2 * scale
        # group label
        parts.append(f"<text x='{x + gap/2}' y='{y0 + 26}' text-anchor='middle' "
                     f"font-size='12' fill='#94a3b8'>{labels[key]}</text>")
        # baseline bar
        parts.append(f"<rect x='{x}' y='{y0 - h1}' width='{bar_w}' height='{h1}' "
                     f"fill='#64748b' rx='3'/>")
        # adaptive bar
        parts.append(f"<rect x='{x + gap - bar_w}' y='{y0 - h2}' width='{bar_w}' height='{h2}' "
                     f"fill='#38bdf8' rx='3'/>")
        for v, bx, color in ((v1, x, "#94a3b8"), (v2, x + gap - bar_w, "#e6ecf5")):
            parts.append(f"<text x='{bx + bar_w/2}' y='{y0 - v*scale - 6}' "
                         f"text-anchor='middle' font-size='11' fill='{color}'>{v:.3f}</text>")
        x += 2 * gap
    return "\n".join(parts)


def main() -> int:
    if not RESULT.exists():
        print("No benchmark_results.json found; run benchmarks/run_benchmark.py first.")
        return 1
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    keys = ["precision", "recall", "f1", "false_positive_rate_clean", "root_cause_accuracy"]
    inner = _bars(data, keys)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="340" viewBox="0 0 760 340">
  <rect width="100%" height="100%" fill="#0b1220"/>
  <text x="40" y="40" font-size="18" font-weight="700" fill="#e6ecf5">LeakLens benchmark ({data['n_cases']} synthetic cases)</text>
  <line x1="40" y1="240" x2="720" y2="240" stroke="#24304d"/>
  <rect x="40" y="295" width="16" height="12" fill="#64748b" rx="2"/>
  <text x="64" y="306" font-size="12" fill="#94a3b8">Rule-based baseline</text>
  <rect x="220" y="295" width="16" height="12" fill="#38bdf8" rx="2"/>
  <text x="244" y="306" font-size="12" fill="#94a3b8">LeakLens adaptive</text>
  {inner}
</svg>"""
    OUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
