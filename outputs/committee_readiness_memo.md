# Committee Readiness Memo

## Project Positioning

This package supports a DBA research project on predicting Federal Audit
Clearinghouse Single Audit findings from FAC-derived organization, program,
and prior-history features.

The code is framed around three reviewable research questions:

1. Which FAC-derived characteristics are most associated with audit findings?
2. How accurately can findings be predicted using information available before
   the future audit outcome is known?
3. Does adding program-level and prior-finding history improve performance over
   a profile-only baseline?

## What Is Ready

- `fac_ingest_committee.py` stages raw FAC endpoint pulls before transformation,
  preserving an auditable trail from source data to analytic table.
- `fac_model_committee.py` uses a forward-year holdout rather than a random
  split, which is appropriate for a prediction study.
- The modeling script compares a profile-only baseline against an enriched
  feature set, reports ROC-AUC, PR-AUC, recall, precision, F1, confusion matrix,
  calibration, and feature importance.
- The model can run in `--demo` mode without live FAC data, which is useful for
  committee walkthroughs before the full API pull is complete.
- The revised modeling script can save a JSON metrics artifact with
  `--results-out`, making results easier to archive with the dissertation
  methods appendix.

## Source Documentation Checked

- FAC API examples document the `https://api.fac.gov` base URL, `X-Api-Key`
  authentication, and PostgREST-style filters such as `audit_year=eq.2024`.
- FAC data dictionary documents the `general`, `federal_awards`,
  `notes_to_sefa`, `findings`, and related endpoint fields used by the ingest
  and modeling stages.
- FAC states that collected data are free to use and in the public domain.
- FAC also documents ongoing curation, reliability, migration, and known
  concerns, which should be acknowledged as data-quality limitations.
- api.data.gov documents hourly rate limits, `X-RateLimit-Limit`,
  `X-RateLimit-Remaining`, and HTTP 429 handling.

## Review Risks To Address In The Proposal

- Define the unit of analysis explicitly: one auditee/report-year row keyed by
  `report_id` and `auditee_uei`.
- State whether 2016-2022 migrated records will be used, excluded, or analyzed
  with sensitivity checks because FAC documents migration and curation issues.
- Explain that the prediction target is not auditor judgment itself; it is the
  FAC-recorded presence of findings and finding types.
- Justify treating first-observed organizations as having no known prior
  finding history rather than imputing unobserved prior outcomes.
- Add a fairness/ethics paragraph: model outputs should support risk triage and
  oversight prioritization, not replace professional audit judgment.
- Pre-register the primary metric. ROC-AUC is useful for ranking; PR-AUC is
  important if findings are relatively rare; recall at a selected threshold
  supports the practical triage use case.

## Suggested Committee Demo

Run the synthetic demonstration:

```powershell
python .\fac_model_committee.py --demo --results-out .\fac_demo_metrics.json
```

Then run a small live-data smoke test after setting `FAC_API_KEY`:

```powershell
python .\fac_ingest_committee.py --smoke-test
```

For a bounded pilot pull:

```powershell
python .\fac_ingest_committee.py --start-year 2022 --end-year 2023 --max-pages 1 --out .\fac_pilot
python .\fac_model_committee.py --input .\fac_pilot\analytic_audit_level.parquet --results-out .\fac_pilot_metrics.json
```

For the full study, remove `--max-pages`, use the approved year range, archive
the raw staged files, analytic table, model output log, and JSON metrics file.
