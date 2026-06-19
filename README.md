# FAC Single Audit Findings Prediction

DBA research project predicting Single Audit findings from Federal Audit
Clearinghouse (FAC) data using logistic regression and gradient boosting with
a forward-year holdout split.

## Dashboard

Live dashboard (no login needed):

**https://alissa-king.github.io/DBA-FAC/**

Shows headline metrics, model comparison, feature importance, triage confusion
matrix, calibration chart, and the profile-vs-enriched RQ3 comparison. The
page is self-contained — you can save it as a file and share it offline.

## Canonical Scripts

| Script | Purpose |
|---|---|
| `fac_ingest.py` | Pull FAC API endpoints, stage raw parquet files, build the analytic table |
| `fac_model.py` | Train and evaluate models; write results JSON for the dashboard |

These are the primary scripts to use. The `outputs/` folder contains the
committee-review versions plus pilot data and pre-computed metrics.

## Quick Start

Install dependencies (Python 3.8+):

```bash
pip install pandas scikit-learn pyarrow requests shap
```

**No FAC key yet? Run the synthetic demo:**

```bash
python fac_model.py --demo
python fac_model.py --demo --results-out outputs/fac_demo_metrics.json
```

**Full FAC pull** (requires a free API key from https://www.fac.gov/api/signup/):

```bash
export FAC_API_KEY="your_key_here"
python fac_ingest.py --smoke-test                         # verify auth
python fac_ingest.py --start-year 2019 --end-year 2023 --out ./fac_data
python fac_model.py --input fac_data/analytic_audit_level.parquet \
                    --results-out outputs/fac_pilot_metrics.json
```

**Bounded pilot pull** (quick walkthrough, no final evidence):

```bash
python fac_ingest.py --start-year 2022 --end-year 2023 --max-pages 1 --out outputs/fac_pilot
python fac_model.py --input outputs/fac_pilot/analytic_audit_level.parquet \
                    --results-out outputs/fac_pilot_metrics.json
```

## Contents

| Path | Description |
|---|---|
| `fac_ingest.py` | Canonical ingestion script |
| `fac_model.py` | Canonical modeling script (SHAP, TimeSeriesSplit CV, SMOTE) |
| `docs/index.html` | Self-contained dashboard (GitHub Pages) |
| `outputs/fac_demo_metrics.json` | Pre-computed synthetic demo metrics (with SHAP) |
| `outputs/fac_pilot_metrics.json` | Pre-computed pilot metrics (with SHAP) |
| `outputs/fac_pilot/` | Bounded 2022–2023 FAC pull |
| `outputs/committee_readiness_memo.md` | Research positioning summary |
| `outputs/requirements_committee.txt` | Pinned dependency list |

## Run Online (no install)

Use the GitHub Actions workflow for a no-install demo:

1. Open the repository on GitHub.
2. Select the **Actions** tab → **FAC Model Demo**.
3. Click **Run workflow**.
4. Download the `fac-model-metrics` artifact from the completed run.

No FAC API key needed — the workflow uses the pilot data already in the repo.
