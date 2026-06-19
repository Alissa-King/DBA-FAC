# FAC Single Audit Findings Prediction

Committee-review package for a DBA research project using Federal Audit
Clearinghouse (FAC) Single Audit data.

## Contents

- `outputs/fac_ingest_committee.py` pulls FAC API endpoint data, stages raw
  parquet files, and builds an audit-level analytic table.
- `outputs/fac_model_committee.py` trains profile-only and enriched prediction
  models with a forward-year holdout.
- `outputs/committee_readiness_memo.md` summarizes research positioning,
  committee-review risks, and demo commands.
- `outputs/requirements_committee.txt` lists Python dependencies.
- `outputs/fac_demo_metrics.json` contains synthetic demo metrics.
- `outputs/fac_pilot/` contains a bounded 2022-2023 pilot pull.
- `outputs/fac_pilot_metrics.json` contains model metrics from the pilot pull.

## Reproduce

Install dependencies:

```powershell
pip install -r outputs\requirements_committee.txt
```

Run the synthetic demo:

```powershell
python outputs\fac_model_committee.py --demo --results-out outputs\fac_demo_metrics.json
```

Run a live FAC smoke test after setting `FAC_API_KEY`:

```powershell
$env:FAC_API_KEY="your_key_here"
python outputs\fac_ingest_committee.py --smoke-test
```

Run a bounded pilot pull:

```powershell
python outputs\fac_ingest_committee.py --start-year 2022 --end-year 2023 --max-pages 1 --out outputs\fac_pilot
python outputs\fac_model_committee.py --input outputs\fac_pilot\analytic_audit_level.parquet --results-out outputs\fac_pilot_metrics.json
```

The pilot data are for committee walkthrough and code verification, not final
dissertation evidence. The full study should remove `--max-pages` and use the
approved year range.
