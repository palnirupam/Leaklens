<div align="center">

# 🔬 LeakLens

**Gemini-powered agentic AI system for autonomous machine-learning data-leakage investigation**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat-square&logo=vite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-verified-34d399?style=flat-square)

*Detect · Investigate · Explain — with real experimental evidence.*

</div>

> 📚 **Research purpose.** LeakLens is an academic/research prototype built to
> study *evidence-based* machine-learning data-leakage detection. It is not a
> production security/audit tool: findings are experimental and must be reviewed
> by a human before any business decision. Per the [License](#-license), this
> project is made available for research and educational use.

---

## 📖 Table of Contents

- [What is LeakLens?](#-what-is-leaklens)
- [Key Features](#-key-features)
- [How it works (Architecture)](#-how-it-works)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation (Windows 11)](#-installation-windows-11)
- [Quickstart](#-quickstart)
- [Configuring Gemini](#-configuring-gemini)
- [API Security](#-api-security)
- [Using the Web UI](#-using-the-web-ui)
- [Demo Dataset & Baseline](#-demo-dataset--baseline)
- [Running Tests & Benchmark](#-running-tests--benchmark)
- [Research Contribution & Evaluation](#-research-contribution--evaluation)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations](#-known-limitations)
- [Technology Stack](#-technology-stack)
- [License](#-license)

---

## 🔍 What is LeakLens?

Machine-learning models sometimes score misleadingly high accuracy because
information from the **target**, **future data**, **test data**, or
**preprocessing operations** accidentally leaks into training.

**LeakLens** automatically investigates these problems. You give it:

- an ML dataset (CSV),
- a target column,
- an optional Python training script,
- optional details about prediction timing, timestamp columns, and entity/group columns.

LeakLens then **profiles** the dataset, **inspects** the code, **generates**
leakage hypotheses, **runs controlled experiments**, and produces an
**evidence-based report**.

It *deliberately* distinguishes between:

| Category | Meaning |
|----------|---------|
| ✅ **Confirmed / strong** | Supported by experimental or static-analysis evidence |
| ⚠️ **Potential / suspicious** | Patterns that need manual review |
| 🟢 **No leakage** | None found by the supported checks |

> **Design rule:** A finding is never called *confirmed* based only on an LLM
> guess or a feature correlation — only experiments can establish a finding.

---

## ✨ Key Features

| Area | What it does |
|------|--------------|
| **Dataset Investigator** | Column typing, missing values, duplicates, target distribution, feature–target association (Cramer's V / \|r\|) |
| **AST Code Analysis** | Static inspection of training scripts: preprocessing-before-split (scaling/imputation/oversampling), fit-on-all-data patterns, missing split |
| **Leakage Detectors** | Target-derived features, duplicate/entity contamination (feature-level, conflict-aware), temporal random-vs-chronological comparison, prediction-time availability |
| **Hypothesis Planner** | Builds structured candidate hypotheses from signals; optional Gemini enrichment (validated, offline fallback) |
| **Adaptive Experiment Executor** | Deterministic, budget-aware selection: `score = evidence × diagnostic_value / (1 + cost)`; updates hypothesis confidence as experiments run |
| **Experiments** | Feature ablation, feature permutation test, duplicate/conflict check, temporal split comparison, availability check, static review |
| **Evidence Verification** | Only experimental/static evidence becomes a finding; unsupported Gemini claims are rejected; corrections attached |
| **Gemini Integration** | Official `google-genai` SDK, structured JSON, retries + rate-limit backoff, caching, never sends raw datasets by default, works without a key |
| **REST API** | Upload CSV + script, background investigation, live status polling, findings, JSON/HTML reports, dashboard stats |
| **Frontend** | React + TypeScript + Vite dashboard: Dashboard, New Investigation, Live Investigation, Findings, Reports |
| **Storage** | SQLite for history/results; local file storage for datasets and reports |
| **Benchmark** | 30 reproducible synthetic cases across 5 leakage categories; compares rule-based baseline vs LeakLens adaptive |

---

## 🧠 How it Works

The system is an **agentic orchestration pipeline** (a custom state machine —
no LangGraph) over four logical agents:

```
DatasetInvestigator ─┐
Static AST analysis ──┤
Signal detection ─────┘
            │
            ▼
   Hypothesis Planner  ── (Gemini optional)
            │
            ▼
Adaptive Experiment Executor  ── runs experiments, updates confidence
            │
            ▼
   Evidence Verifier   ── produces findings + report
```

Each stage produces structured data; no agent "guesses" results — Python runs
all numerical experiments.

### The budget-aware adaptive experiment selection

Instead of running every possible check in a fixed order, LeakLens scores each
candidate experiment with an explainable, deterministic priority formula:

```
score(experiment) = evidence_strength(hypothesis) × diagnostic_value(experiment)
                    ─────────────────────────────────────────────────────────
                               1 + normalized_cost(experiment)
```

tie-broken by hypothesis and action order so identical inputs produce the same experiment decisions
(fully reproducible).

---

## 📁 Project Structure

```
LeakLens/
├── backend/
│   ├── app/
│   │   ├── api/            FastAPI routes (investigations, stats, reports)
│   │   ├── agents/         HypothesisPlanner, AdaptiveExperimentExecutor, EvidenceVerifier
│   │   ├── core/           config (pydantic-settings), domain models, SQLite store, security
│   │   ├── detectors/      dataset investigator, AST analyzer, leakage signals
│   │   ├── experiments/    experiment registry + runner
│   │   ├── ml/             scikit-learn pipeline helpers
│   │   ├── reporting/      JSON / HTML report rendering
│   │   ├── services/       Gemini client, orchestrator (agentic state machine)
│   │   ├── data/           synthetic dataset generator (fallback + benchmark)
│   │   └── main.py         FastAPI app entry-point
│   ├── tests/              pytest unit and regression suite
│   ├── .env.example
│   └── requirements.txt
├── frontend/               Vite + React + TypeScript SPA
├── data/                   demo dataset + LeakLens storage (sqlite, uploads, reports)
├── training/               baseline training script
├── scripts/                demo downloader, helpers, smoke checks
├── benchmarks/             evaluation + runner + chart
├── docs/
└── README.md
```

---

## 🧰 Prerequisites

| Tool | Version |
|------|---------|
| **Python** | 3.11+ (developed on 3.14) |
| **Node.js** | 18+ (developed on 24) |
| **npm** | 9+ |
| OS | Windows 11 (also works on Linux/macOS) |

---

## ⚙️ Installation (Windows 11)

### 1. Backend

```powershell
cd D:\LeakLens
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Frontend

```powershell
cd D:\LeakLens
npm --prefix frontend install
```

### 3. Configure environment (optional)

```powershell
Copy-Item backend\.env.example backend\.env
```

Edit `backend\.env` to add your Gemini key and/or API token (see below).

---

## 🚀 Quickstart

### Fastest: one command (backend + frontend together)

From the project root (`D:\LeakLens`):

```powershell
python start_dev.py
```

(or `python scripts\start_dev.py` — both are equivalent.)

This starts **both** servers for you:

- Backend → `http://127.0.0.1:8000`
- Frontend → `http://127.0.0.1:5173`

Press **Ctrl+C** to stop them both (process trees are cleaned up automatically).
Then open **http://127.0.0.1:5173** in your browser.

### Manual alternative

```powershell
# 1. Download the demo dataset (UCI Bank Marketing; falls back to synthetic if offline)
python scripts\download_demo_dataset.py

# 2. Start the backend (terminal 1)
python -m uvicorn app.main:app --app-dir backend --reload --port 8000

# 3. Start the frontend (terminal 2)
npm --prefix frontend run dev
```

- Backend health check: <http://127.0.0.1:8000/api/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

---

## 🔑 Configuring Gemini (optional)

LeakLens runs **fully offline** without a Gemini key. To enable Gemini-assisted
hypothesis generation:

```powershell
Copy-Item backend\.env.example backend\.env
```

Then edit `backend\.env`:

```ini
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-2.0-flash
ALLOW_DATA_TO_GEMINI=false
```

**Privacy defaults:**

- The key is only read from `backend\.env` and is never exposed to React.
- Full datasets and private code are **never** sent by default — only sanitized
  column names and aggregated statistics are shared.
- Set `ALLOW_DATA_TO_GEMINI=true` only if you accept that for *your* data.
- If Gemini fails or is unconfigured, the system transparently falls back to
  rule-based analysis. Python always runs the numerical experiments regardless.

---

## 🔐 API Security

LeakLens is a **local research tool** and enforces two safeguards:

**1. Loopback-only binding by default**

The server listens on `127.0.0.1`. If you try to start it on a non-loopback
interface (e.g. `--host 0.0.0.0`) **without** an API key, startup refuses:

```
Refusing to bind to a non-loopback interface without authentication
```

**2. Shared-token auth (recommended for any exposure)**

Set `API_KEY=...` in `backend\.env`. When set, every `/api` route (except
`/api/health`) requires the `X-API-Key` header; missing/wrong tokens get `401`.

```powershell
# backend\.env
API_KEY=my-secret-token

# frontend\.env.local  (same value)
VITE_API_KEY=my-secret-token
```

> ⚠️ **Never expose the backend to the public network without setting `API_KEY`**
> — and preferably place it behind a TLS-terminating reverse proxy. The health
> endpoint is intentionally open (it reveals no data).

---

## 🖥️ Using the Web UI

| Page | Purpose |
|------|---------|
| **Dashboard** | Total investigations, confirmed/potential findings, average duration, mini-chart of recent findings, history table |
| **New Investigation** | Upload CSV, set target, upload optional training script, timestamp/entity columns, unavailable-at-prediction features, Gemini toggle |
| **Live Investigation** | Real-time pipeline timeline, progress, agent, live hypotheses (with confidence meters) and experiments as they run, findings on completion |
| **Findings** | Evidence-backed findings: category, severity, evidence status, affected features, corrections |
| **Reports** | Download full JSON/HTML reports + inline quick preview |

> 💡 **Demo tip:** use `data\demo\bank_marketing.csv` (target = `y`). The
> `duration` column is unavailable at prediction time for this pre-call task —
> list it under "Features unavailable at prediction time". To see a
> target-derived leak, generate a synthetic case via the benchmark and upload it.

---

## 📊 Demo Dataset & Baseline

### Demo dataset

```powershell
python scripts\download_demo_dataset.py
```

Downloads the UCI **Bank Marketing** dataset (45,211 rows) to `data\demo\`.
If the network is unavailable, a synthetic fallback demo is written instead, so
the project never depends entirely on an internet download.

### Baseline training

The demo is configured as a **pre-call prediction** task, so the `duration`
column (call duration) is excluded from the valid baseline:

```powershell
python training\train_baseline.py                       # valid baseline (no duration)
python training\train_baseline.py --include-duration    # comparison only
```

---

## ✅ Running Tests & Benchmark

```powershell
# Unit and regression tests
python -m pytest backend\tests -q

# Frontend type checks and production build
npm --prefix frontend run build

# Research benchmark (30 synthetic cases across 5 categories)
python -m benchmarks.run_benchmark

# Generate the comparison chart (SVG)
python -m benchmarks.make_chart
```

The browser regression check is [scripts/check_frontend.js](scripts/check_frontend.js).
With the dev servers running and matching nonempty `API_KEY` / `VITE_API_KEY`,
load it through Playwright's `browser_run_code_unsafe` using the `filename`
argument. It creates a synthetic investigation and checks upload retention,
typing focus, authenticated JSON/HTML downloads, keyboard preview, and retry
after a failed download. Use temporary `STORAGE_DIR` storage for this check.

### Recorded benchmark results (30 cases, 6 per category)

See [benchmark_results.json](benchmarks/benchmark_results.json) for the latest
recorded metrics and [benchmark_chart.svg](benchmarks/benchmark_chart.svg) for
the comparison chart. Regenerate both after changing detection or evaluation.

Precision, recall and F1 count reported categories, including potential and
suspicious review flags. They do not measure confirmed leakage alone.
Root-cause accuracy includes missed categories in its denominator.
Experiment counts cover registered adaptive experiments and exclude profiling
and signal calculations; the run-all bound uses the executor's actual candidate
list. Runtimes depend on the machine and workload.

### Real-dataset benchmark

The same harness also runs on **real datasets** (sklearn's bundled Wine and
Breast Cancer, plus the UCI Bank Marketing demo) with one documented leakage
injection per case:

```powershell
python -m benchmarks.run_real_benchmark
```

See [real_benchmark_results.json](benchmarks/real_benchmark_results.json) for
recorded metrics and each case's expected and reported categories. The Wine
duplicate case preserves the original three target classes. Each adaptive case
runs once, and the same findings feed both its verdict and the aggregate scores.

Target-derived findings require an ablation or permutation AUC drop above 0.10;
correlated valid predictors can hide an injected leak. Temporal split comparison
can also miss a precomputed future-derived feature that remains predictive in
both splits. These missed categories remain in the benchmark's false negatives.

---

## 🔬 Research Contribution & Evaluation

**Budget-aware adaptive leakage investigation.** LeakLens does not run every
check in a fixed order. It scores candidate experiments with a deterministic,
explainable priority formula and stops once hypothesis confidence is established.
Tests check reproducible experiment decisions, unique experiment IDs, evidence
verification and a bounded experiment budget. The recorded benchmark results
show its measured performance relative to the signal-level baseline; an
accuracy or efficiency improvement must be established by those results.

**Red-team hardening (2026).** Two adversarial reviews exposed and led to
fixes:

- *Legitimate strong feature trap* — a strong but valid predictor (e.g. tumour
  size) was being labelled "confirmed critical leak" purely because removing it
  dropped AUC. Fixed by gating confirmation on mechanism-level evidence, so
  findings are downgraded to "suspicious — manual review" when no mechanism is
  proven, and the advice text is now conditional.
- *Stealth XOR interaction* — a 100%-informative pairwise interaction with zero
  individual-feature signal was invisible. Fixed with a shallow-tree pairwise
  interaction scan that surfaces such cases as review flags (never confirmed).
  Regression tests in `backend/tests/test_redteam_fixes.py` cover both.

The synthetic harness runs Gemini hypothesis evaluation when `GEMINI_API_KEY`
is configured, using the same sanitized metadata and computed signals without
adaptive verification. It records metrics only after every response passes
validation. Missing credentials produce **skipped**; unavailable or invalid
responses produce **failed** without fabricated scores. Offline tests exercise
this path with a fake provider; a live run requires a working Gemini key.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install` fails on Python 3.12+ | Use a fresh virtualenv; ensure Python 3.11+ is on PATH |
| Backend won't start on `0.0.0.0` | Set `API_KEY` or bind to `127.0.0.1` (see Security) |
| Frontend shows `401` | `API_KEY` is set in backend but `VITE_API_KEY` is missing/mismatched in `frontend\.env.local` |
| `npm run dev` port already in use | Change `server.port` in `frontend/vite.config.ts` or kill the other process |
| No Gemini suggestions | Check `GEMINI_API_KEY`; the app still works offline |
| UCI download fails | Script automatically writes a synthetic fallback demo |

---

## ⚠️ Known Limitations

- First release targets **scikit-learn-compatible tabular workflows**; deep
  learning / multimodal pipelines are out of scope.
- Static AST analysis flags *suspected* patterns — it cannot prove runtime
  leakage without a runtime experiment.
- Temporal detection relies on split-comparison and feature ablation; subtle or
  weak leaks may be missed.
- Benchmarks cover controlled injections on both synthetic and real datasets;
  real-world leakage patterns may differ from the injected scenarios.
- Gemini suggestions affect only hypothesis generation; Python always executes
  the numerical experiments.
- Target-derived leaks are never auto-labelled "confirmed" from predictive power
  alone: confirmation additionally requires mechanism-level evidence (e.g.
  static analysis showing the feature is computed from target/future). Without
  it the finding is kept at "potential/suspicious — manual review". On
  feature-rich real datasets a weak leak may still be conservatively missed
  (precision-first). Verified against a real-data benchmark
  (`python -m benchmarks.run_real_benchmark`).
- Pairwise **interaction leaks** (weak individual signals, strong joint signal —
  e.g. XOR) are surfaced as review-only flags (`feature_interaction_leakage`);
  they are never auto-confirmed, since strong legitimate interactions are normal.
- Uploads/reports are stored locally; no production-grade authentication or
  multi-user features are included.

---

## 🧩 Technology Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2, pandas, NumPy, scikit-learn,
  Google GenAI SDK, SQLite, pytest
- **Frontend:** React 18, TypeScript, Vite 5
- **UI/UX:** Custom professional responsive CSS (no UI framework dependency)

---

## ⚖️ License

LeakLens is released under the **Academic Research-Only License**
(see [`LICENSE`](./LICENSE)) — **Copyright © 2026 Nirupam Pal**.

This is a research-only license:

- **Permitted:** use, modification, and redistribution for **non-commercial
  research and educational purposes** only (theses, coursework, teaching,
  internal research).
- **Prohibited:** **any commercial use** — selling, licensing, or monetizing
  the software or derivative works — without prior written permission from the
  author.
- **Attribution required:** publications/projects using the software must credit
  the author (Nirupam Pal).

The software is provided "as is" with **no warranty**.

---

<div align="center">
<i>Built as a reproducible research prototype. Backend, agents, detectors,
experiments, Gemini integration, frontend, tests, and benchmark are all
implemented and verified.</i>
</div>
