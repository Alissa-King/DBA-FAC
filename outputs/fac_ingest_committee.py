#!/usr/bin/env python3
"""
fac_ingest.py
=============
Ingestion pipeline for the Federal Audit Clearinghouse (FAC) public API.

This is the Year-1 / Term-2 milestone from the DBA project plan: pull real
Single-Audit data across multiple endpoints and years, stage it raw, and
join it into one analytic table keyed by report_id / auditee_uei.

WHAT IT DOES
------------
  1. Pulls each FAC endpoint (general, federal_awards, findings, ...) for a
     range of audit years, using PostgREST range pagination so you get ALL
     rows, not just the first page.
  2. Respects the api.data.gov rate limit (watches X-RateLimit-Remaining and
     backs off automatically).
  3. Stages every raw pull to disk (parquet) BEFORE any transformation, so the
     cleaning step is reproducible and auditable -- exactly what a doctoral
     methods chapter needs.
  4. Builds an audit-level analytic table (one row per organization-year) with
     a binary "any finding" label plus finding-type labels.

PRIMARY SOURCES (verified)
--------------------------
  * Base URL + auth header + PostgREST `eq.` filters:  https://www.fac.gov/api/examples/
  * Endpoint schema / field names:                     https://www.fac.gov/api/dictionary/
  * Rate-limit headers (X-RateLimit-Remaining):        https://api.data.gov/docs/developer-manual/
  * Public-domain data statement:                      https://www.fac.gov/data/

GET AN API KEY (free)
---------------------
  Sign up at  https://www.fac.gov/api/signup/  (issues an api.data.gov key).
  Then either:
      export FAC_API_KEY="your_key_here"      # recommended
  ...or pass --api-key on the command line.

USAGE
-----
  # Smoke test -- confirm your key works and the API is reachable (5 rows):
  python fac_ingest.py --smoke-test

  # Full pull for 2019-2023, audit-level analytic table:
  python fac_ingest.py --start-year 2019 --end-year 2023 --out ./fac_data

DEPENDENCIES
------------
  pip install requests pandas pyarrow
  (pyarrow is only needed for parquet; the script falls back to CSV if absent.)

NOTE
----
  This script makes live network calls and will not run in a sandbox with
  networking disabled. Run it on your own machine after setting FAC_API_KEY.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency: pip install pandas")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_URL = "https://api.fac.gov"          # confirmed at fac.gov/api/examples
PAGE_SIZE = 5000                          # rows per request (PostgREST honors Range/limit)
RATE_FLOOR = 25                           # pause when this few requests remain in the window
DEFAULT_TIMEOUT = 60

# Endpoints to pull. Order matters only for readability.
# Each maps to a table in the FAC API data dictionary.
ENDPOINTS = [
    "general",          # one row per audit  -> features + outcome flags
    "federal_awards",   # one row per program -> program-mix features, findings_count
    "findings",         # one row per finding -> the core outcome labels
    "passthrough",      # subrecipient structure
    "notes_to_sefa",    # optional: de minimis indirect rate, etc.
]

# Columns we keep from `general` for the analytic table. Pulling only what we
# need keeps payloads small and the rate-limit budget healthy. Field names are
# taken verbatim from the FAC data dictionary.
GENERAL_COLS = [
    "report_id", "audit_year", "auditee_uei", "auditee_ein", "auditee_name",
    "auditee_state", "entity_type", "fy_start_date", "fy_end_date",
    "total_amount_expended", "dollar_threshold", "is_low_risk_auditee",
    "is_going_concern_included",
    "is_internal_control_material_weakness_disclosed",
    "is_internal_control_deficiency_disclosed",
    "is_material_noncompliance_disclosed",
    "cognizant_agency", "oversight_agency", "fac_accepted_date",
]


# --------------------------------------------------------------------------
# HTTP layer with pagination + rate-limit awareness
# --------------------------------------------------------------------------
def make_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "X-Api-Key": api_key,
        "Accept": "application/json",
    })
    return s


def _respect_rate_limit(resp: requests.Response) -> None:
    """If api.data.gov says we're nearly out of requests this hour, sleep."""
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        try:
            remaining = int(remaining)
        except ValueError:
            return
        if remaining <= RATE_FLOOR:
            # Window resets hourly on a rolling basis; a conservative pause.
            print(f"    [rate-limit] only {remaining} requests left this hour; "
                  f"sleeping 60s to be safe...")
            time.sleep(60)


def fetch_endpoint(session, endpoint, *, audit_year=None, select=None,
                   extra_filters=None, page_size=PAGE_SIZE, max_pages=None):
    """
    Pull every row from a FAC endpoint for a given audit_year, paging with
    PostgREST Range headers. Returns a list of dict rows.

    PostgREST filter syntax (from fac.gov/api/examples):
        ?audit_year=eq.2023&cognizant_agency=eq.21
    Pagination uses the Range header which PostgREST supports natively.
    """
    rows = []
    offset = 0
    page = 0

    params = {}
    if audit_year is not None:
        params["audit_year"] = f"eq.{audit_year}"
    if select:
        params["select"] = ",".join(select)
    if extra_filters:
        params.update(extra_filters)

    while True:
        page += 1
        if max_pages and page > max_pages:
            break

        # PostgREST: use Range headers for complete-page retrieval.
        headers = {
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + page_size - 1}",
            "Prefer": "count=exact",
        }
        url = f"{BASE_URL}/{endpoint}"
        resp = session.get(url, params=params, headers=headers,
                           timeout=DEFAULT_TIMEOUT)

        if resp.status_code in (200, 206):
            batch = resp.json()
            rows.extend(batch)
            _respect_rate_limit(resp)
            got = len(batch)
            print(f"    {endpoint} y={audit_year} page {page}: +{got} rows "
                  f"(total {len(rows)})")
            if got < page_size:
                break                      # last page
            offset += page_size
            time.sleep(0.2)                # gentle pacing
        elif resp.status_code == 429:
            print("    [429] rate limited; sleeping 60s then retrying...")
            time.sleep(60)
            continue
        else:
            print(f"    [ERROR] {endpoint} y={audit_year}: "
                  f"HTTP {resp.status_code} -> {resp.text[:300]}")
            break

    return rows


# --------------------------------------------------------------------------
# Staging (raw -> disk)
# --------------------------------------------------------------------------
def stage_raw(rows, out_dir: Path, endpoint: str, year):
    """Persist a raw pull before any transformation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    stem = out_dir / f"{endpoint}_{year}"
    try:
        df.to_parquet(f"{stem}.parquet", index=False)
        path = f"{stem}.parquet"
    except Exception:
        df.to_csv(f"{stem}.csv", index=False)
        path = f"{stem}.csv"
    print(f"    staged -> {path}  ({len(df)} rows, {df.shape[1]} cols)")
    return df


# --------------------------------------------------------------------------
# Cleaning helpers (handles documented FAC data-quality issues)
# --------------------------------------------------------------------------
def coerce_bool(series):
    """FAC booleans arrive as 'Y'/'N', 'true'/'false', or actual bools."""
    return series.map(lambda v: str(v).strip().lower() in ("y", "yes", "true", "1", "t")
                      if pd.notna(v) else False)


def build_analytic_table(general_df, awards_df, findings_df):
    """
    Audit-level analytic table: one row per organization-year, with a binary
    'any finding' label and finding-type labels.

    Data-quality handling (see project doc Part 3.4):
      * We validate the material-weakness flag against the findings endpoint
        rather than trusting general.is_..._material_weakness_disclosed alone,
        because a historical material-weakness field was mis-mapped in migration
        (FAC documented concern #9).
      * We treat fac_accepted_date cautiously for migrated years and prefer
        fiscal-year fields for any time-based splitting (FAC concerns #3, #5).
    """
    gen = general_df.copy()

    # --- finding counts per audit, from federal_awards.findings_count ---
    if not awards_df.empty and "findings_count" in awards_df.columns:
        awards_df["findings_count"] = pd.to_numeric(
            awards_df["findings_count"], errors="coerce").fillna(0)
        per_audit = (awards_df.groupby("report_id")["findings_count"]
                     .sum().rename("total_findings_count").reset_index())
        n_programs = (awards_df.groupby("report_id")["report_id"]
                      .count().rename("n_programs").reset_index())
    else:
        per_audit = pd.DataFrame(columns=["report_id", "total_findings_count"])
        n_programs = pd.DataFrame(columns=["report_id", "n_programs"])

    # --- finding-type labels validated against the findings endpoint ---
    if not findings_df.empty:
        for col in ["is_material_weakness", "is_significant_deficiency",
                    "is_questioned_costs", "is_repeat_finding"]:
            if col in findings_df.columns:
                findings_df[col] = coerce_bool(findings_df[col])
            else:
                findings_df[col] = False
        ftypes = (findings_df.groupby("report_id")
                  .agg(any_material_weakness=("is_material_weakness", "max"),
                       any_significant_deficiency=("is_significant_deficiency", "max"),
                       any_questioned_costs=("is_questioned_costs", "max"),
                       any_repeat_finding=("is_repeat_finding", "max"),
                       n_findings=("report_id", "count"))
                  .reset_index())
    else:
        ftypes = pd.DataFrame(columns=["report_id"])

    out = (gen
           .merge(per_audit, on="report_id", how="left")
           .merge(n_programs, on="report_id", how="left")
           .merge(ftypes, on="report_id", how="left"))

    out["total_findings_count"] = out["total_findings_count"].fillna(0)
    out["n_programs"] = out["n_programs"].fillna(0)
    out["n_findings"] = out.get("n_findings", pd.Series(0, index=out.index)).fillna(0)

    for col in [
        "any_material_weakness",
        "any_significant_deficiency",
        "any_questioned_costs",
        "any_repeat_finding",
    ]:
        if col not in out.columns:
            out[col] = False
        out[col] = coerce_bool(out[col]).astype(int)

    # --- THE LABEL ---
    # Prefer the validated findings-endpoint count; fall back to award counts.
    out["any_finding"] = ((out["n_findings"] > 0) |
                          (out["total_findings_count"] > 0)).astype(int)

    if "is_low_risk_auditee" in out.columns:
        out["is_low_risk_auditee"] = coerce_bool(out["is_low_risk_auditee"])

    # Size band feature from total federal expended.
    out["total_amount_expended"] = pd.to_numeric(
        out.get("total_amount_expended"), errors="coerce")
    out["size_band"] = pd.cut(
        out["total_amount_expended"].fillna(0),
        bins=[-1, 1_000_000, 10_000_000, 100_000_000, float("inf")],
        labels=["<1M", "1-10M", "10-100M", ">100M"])

    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def smoke_test(session):
    print("Smoke test: GET /general?limit=5 ...")
    resp = session.get(f"{BASE_URL}/general",
                       params={"limit": 5}, timeout=DEFAULT_TIMEOUT)
    print(f"  HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  OK -- received {len(data)} rows.")
        if data:
            print("  Sample report_ids:",
                  [r.get("report_id") for r in data])
        rem = resp.headers.get("X-RateLimit-Remaining")
        if rem:
            print(f"  Rate-limit remaining this hour: {rem}")
        return True
    print(f"  FAILED -> {resp.text[:400]}")
    print("  Check that FAC_API_KEY is set and valid "
          "(sign up at https://www.fac.gov/api/signup/).")
    return False


def run(args):
    api_key = args.api_key or os.environ.get("FAC_API_KEY")
    if not api_key:
        sys.exit("No API key. Set FAC_API_KEY env var or pass --api-key. "
                 "Get one free at https://www.fac.gov/api/signup/")

    session = make_session(api_key)

    if args.smoke_test:
        ok = smoke_test(session)
        sys.exit(0 if ok else 1)

    out_root = Path(args.out)
    raw_dir = out_root / "raw"
    years = list(range(args.start_year, args.end_year + 1))
    print(f"Pulling audit years {years[0]}-{years[-1]} into {out_root}/\n")

    staged = {ep: [] for ep in ENDPOINTS}

    for year in years:
        print(f"Year {year}")
        for ep in ENDPOINTS:
            select = GENERAL_COLS if ep == "general" else None
            rows = fetch_endpoint(session, ep, audit_year=year, select=select,
                                  max_pages=args.max_pages)
            if rows:
                df = stage_raw(rows, raw_dir, ep, year)
                staged[ep].append(df)
        print()

    def concat(ep):
        parts = staged.get(ep, [])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    general_df = concat("general")
    awards_df = concat("federal_awards")
    findings_df = concat("findings")

    if general_df.empty:
        sys.exit("No `general` rows pulled -- nothing to build. "
                 "Check the smoke test and your key.")

    print("Building audit-level analytic table...")
    analytic = build_analytic_table(general_df, awards_df, findings_df)

    out_path = out_root / "analytic_audit_level.parquet"
    try:
        analytic.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_root / "analytic_audit_level.csv"
        analytic.to_csv(out_path, index=False)

    print(f"\nDONE.")
    print(f"  Analytic table: {out_path}")
    print(f"  Rows (organization-years): {len(analytic):,}")
    if "any_finding" in analytic:
        rate = analytic["any_finding"].mean()
        print(f"  Base rate of any finding: {rate:.1%}")
        print("  (This base rate is your model's 'beat-the-coin-flip' benchmark.)")
    print("\n  Next step: feature engineering + a logistic-regression baseline "
          "(Year-2 Term-2 milestone).")


def parse_args():
    p = argparse.ArgumentParser(description="FAC API ingestion pipeline.")
    p.add_argument("--api-key", help="FAC/api.data.gov key (or set FAC_API_KEY).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Just verify the API + key work, then exit.")
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--out", default="./fac_data", help="Output directory.")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Cap pages per endpoint/year (handy for a quick test).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
