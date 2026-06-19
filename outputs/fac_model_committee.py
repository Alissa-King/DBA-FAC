#!/usr/bin/env python3
"""
fac_model.py
============
Modeling stage for the FAC Single-Audit findings-prediction project.
Consumes the analytic table produced by fac_ingest.py and answers RQ1-RQ3:

  RQ1  Which FAC-derived characteristics are most associated with findings?
       -> feature importance / coefficients.
  RQ2  How accurately can we predict (a) any finding and (b) finding type,
       using only pre-audit-close information?
       -> AUC, precision/recall, calibration, on a held-out FUTURE year.
  RQ3  Does program-level + prior-finding history beat a profile-only baseline?
       -> two feature sets, compared.

WHY THE SPLIT IS BY YEAR (not random)
-------------------------------------
A random split lets the model "see the future." Single-audit risk is a
forward prediction problem, so we train on earlier years and test on the most
recent year. That is the honest analogue of "score this organization before
its audit closes," and it is what a committee will expect to see.

INPUT
-----
  The parquet/csv written by fac_ingest.py:  analytic_audit_level.parquet
  Required columns (all produced by the ingest script):
    report_id, audit_year, auditee_uei, entity_type, total_amount_expended,
    size_band, n_programs, total_findings_count, any_finding,
    any_material_weakness, any_questioned_costs, any_repeat_finding,
    is_low_risk_auditee, cognizant_agency  (others optional)

USAGE
-----
  # Real data from the ingest step:
  python fac_model.py --input ./fac_data/analytic_audit_level.parquet

  # No data yet? Generate a realistic synthetic table and run on it,
  # so you can see exactly what the output looks like first:
  python fac_model.py --demo

DEPENDENCIES
------------
  pip install pandas scikit-learn pyarrow
  (HistGradientBoosting is built into scikit-learn -- no XGBoost needed.)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve


# --------------------------------------------------------------------------
# Feature sets -- the heart of the RQ3 comparison
# --------------------------------------------------------------------------
# Baseline: who the organization IS, knowable before any audit work.
PROFILE_NUMERIC = ["total_amount_expended"]
PROFILE_CATEGORICAL = ["entity_type", "size_band", "is_low_risk_auditee"]

# Enriched: profile PLUS program-level signal and the organization's own
# lagged finding history (last year's findings -> this year's risk).
ENRICHED_NUMERIC = PROFILE_NUMERIC + [
    "n_programs", "prior_year_findings", "prior_year_any_finding",
]
ENRICHED_CATEGORICAL = PROFILE_CATEGORICAL + ["cognizant_agency"]

TARGET = "any_finding"


# --------------------------------------------------------------------------
# Lagged prior-finding history (built within-organization, no leakage)
# --------------------------------------------------------------------------
def add_prior_year_history(df):
    """
    For each organization (auditee_uei), attach LAST year's finding outcome to
    THIS year's row. This is a legitimate pre-audit predictor: when you are
    scoring 2024, last year's (2023) audit is already filed.
    """
    df = df.copy()
    df["audit_year"] = pd.to_numeric(df["audit_year"], errors="coerce")
    df = df.sort_values(["auditee_uei", "audit_year"])

    df["prior_year_any_finding"] = (
        df.groupby("auditee_uei")["any_finding"].shift(1)
    )
    src = "total_findings_count" if "total_findings_count" in df.columns else "any_finding"
    df["prior_year_findings"] = df.groupby("auditee_uei")[src].shift(1)

    # First-ever appearance has no prior year -> 0 (treated as "no known history").
    df["prior_year_any_finding"] = df["prior_year_any_finding"].fillna(0)
    df["prior_year_findings"] = df["prior_year_findings"].fillna(0)
    return df


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------
def make_preprocessor(numeric, categorical):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", cat, categorical),
    ])


def make_logreg(numeric, categorical):
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def make_gbm(numeric, categorical):
    # HistGradientBoosting handles missing values natively and is fast.
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=None,
            l2_regularization=1.0, early_stopping=True, random_state=42)),
    ])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def pick_threshold_for_recall(y_true, scores, target_recall=0.80):
    """
    Compliance triage cares about catching at-risk audits, so we tune the
    decision threshold to hit a recall target, then report the precision we pay
    for it. (A finance team would rather over-flag than miss a finding.)
    """
    order = np.argsort(scores)[::-1]
    y_sorted = np.asarray(y_true)[order]
    total_pos = y_sorted.sum()
    if total_pos == 0:
        return 0.5
    cum_tp = np.cumsum(y_sorted)
    recall_at = cum_tp / total_pos
    hit = np.searchsorted(recall_at, target_recall)
    hit = min(hit, len(scores) - 1)
    return float(np.sort(scores)[::-1][hit])


def evaluate(name, model, X_tr, y_tr, X_te, y_te, target_recall=0.80):
    model.fit(X_tr, y_tr)
    scores = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, scores)
    ap = average_precision_score(y_te, scores)
    brier = brier_score_loss(y_te, scores)

    thr = pick_threshold_for_recall(y_te, scores, target_recall)
    preds = (scores >= thr).astype(int)
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1 = f1_score(y_te, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()

    print(f"\n  {name}")
    print(f"    ROC-AUC ................ {auc:.3f}")
    print(f"    PR-AUC (avg precision).. {ap:.3f}")
    print(f"    Brier (calibration) .... {brier:.3f}  (lower is better)")
    print(f"    @ recall>={target_recall:.0%} threshold={thr:.3f}:")
    print(f"       precision ........... {prec:.3f}")
    print(f"       recall .............. {rec:.3f}")
    print(f"       F1 .................. {f1:.3f}")
    print(f"       confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    return {"name": name, "model": model, "auc": float(auc), "ap": float(ap),
            "brier": float(brier), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "threshold": float(thr),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "scores": scores}


def report_calibration(y_te, scores, bins=5):
    frac_pos, mean_pred = calibration_curve(y_te, scores, n_bins=bins,
                                            strategy="quantile")
    print("\n  Calibration (do predicted probabilities match reality?)")
    print("    predicted -> actual")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"      {mp:5.1%}    -> {fp:5.1%}")


def report_top_features(model, X_te, y_te, numeric, categorical, k=12):
    """
    Feature importance for RQ1.
      * Linear models: absolute standardized coefficients.
      * Tree/boosting models (HistGBM has no feature_importances_):
        permutation importance on the held-out test set.
    """
    clf = model.named_steps["clf"]

    if hasattr(clf, "coef_"):
        prep = model.named_steps["prep"]
        try:
            names = prep.get_feature_names_out()
        except Exception:
            names = np.array(numeric + categorical)
        imp = np.abs(clf.coef_[0])
        label = "abs(standardized coefficient)"
        n = min(len(names), len(imp))
        order = np.argsort(imp[:n])[::-1][:k]
        print(f"\n  Top features (RQ1) by {label}:")
        rows = []
        for i in order:
            print(f"    {imp[i]:7.3f}  {names[i]}")
            rows.append({"feature": str(names[i]), "importance": float(imp[i]),
                         "method": label})
        return rows

    # Permutation importance works for ANY fitted estimator, including HistGBM.
    from sklearn.inspection import permutation_importance
    cols = numeric + categorical
    r = permutation_importance(model, X_te, y_te, n_repeats=8,
                               random_state=42, scoring="roc_auc")
    order = np.argsort(r.importances_mean)[::-1][:k]
    print(f"\n  Top features (RQ1) by permutation importance (drop in ROC-AUC):")
    rows = []
    for i in order:
        print(f"    {r.importances_mean[i]:+7.4f}  {cols[i]}")
        rows.append({"feature": str(cols[i]),
                     "importance": float(r.importances_mean[i]),
                     "method": "permutation drop in ROC-AUC"})
    return rows


def serializable_result(result):
    keep = [
        "name", "auc", "ap", "brier", "precision", "recall", "f1",
        "threshold", "tp", "fp", "fn", "tn",
    ]
    return {k: result[k] for k in keep}


# --------------------------------------------------------------------------
# Synthetic table for --demo (mirrors fac_ingest output schema EXACTLY)
# --------------------------------------------------------------------------
def make_demo_table(n_per_year=4000, years=(2019, 2020, 2021, 2022, 2023), seed=42):
    rng = np.random.default_rng(seed)
    entity_types = np.array(["non-profit", "state", "local", "higher-ed", "tribal"])
    et_p = [0.45, 0.12, 0.28, 0.12, 0.03]
    agencies = np.array(["93", "84", "10", "21", "20", "14", "66"])  # HHS, Ed, USDA, etc.

    # Persistent per-organization risk propensity -> creates real prior-year signal.
    n_org = int(n_per_year * 1.3)
    org_ids = np.array([f"UEI{idx:06d}" for idx in range(n_org)])
    org_risk = rng.beta(1.4, 6.0, size=n_org)        # most orgs low-risk
    org_entity = rng.choice(entity_types, size=n_org, p=et_p)
    org_agency = rng.choice(agencies, size=n_org)

    rows = []
    for y in years:
        idx = rng.choice(n_org, size=n_per_year, replace=False)
        for j in idx:
            n_programs = int(1 + rng.poisson(2.5))
            expended = float(np.exp(rng.normal(14.0, 1.6)))  # ~ $1.2M median, long tail
            low_risk = rng.random() < 0.35
            # True finding probability rises with org risk, program count, size;
            # falls if low-risk auditee. (This is the signal the model recovers.)
            logit = (-2.3
                     + 5.5 * org_risk[j]
                     + 0.10 * n_programs
                     + 0.18 * (np.log(expended) - 14.0)
                     - 0.6 * low_risk)
            p = 1 / (1 + np.exp(-logit))
            any_finding = int(rng.random() < p)
            tfc = int(rng.poisson(1.4)) if any_finding else 0
            rows.append({
                "report_id": f"{y}-{j}",
                "audit_year": str(y),
                "auditee_uei": org_ids[j],
                "entity_type": org_entity[j],
                "total_amount_expended": expended,
                "n_programs": n_programs,
                "is_low_risk_auditee": bool(low_risk),
                "cognizant_agency": org_agency[j],
                "total_findings_count": tfc,
                "any_finding": any_finding,
                "any_material_weakness": int(any_finding and rng.random() < 0.4),
                "any_questioned_costs": int(any_finding and rng.random() < 0.5),
                "any_repeat_finding": int(any_finding and rng.random() < 0.3),
            })
    df = pd.DataFrame(rows)
    df["size_band"] = pd.cut(df["total_amount_expended"],
                             bins=[-1, 1e6, 1e7, 1e8, float("inf")],
                             labels=["<1M", "1-10M", "10-100M", ">100M"])
    return df


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def load_table(args):
    if args.demo:
        print("DEMO MODE: generating a synthetic analytic table that matches the "
              "fac_ingest.py output schema.\n(Replace with --input <parquet> once "
              "your real pull is done.)\n")
        return make_demo_table()
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Input not found: {path}\nRun fac_ingest.py first, or use --demo.")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}\n")
    return df


def ensure_columns(df):
    """Fill any optional columns the model expects but a thin pull might lack."""
    defaults = {
        "n_programs": 0, "total_findings_count": 0, "is_low_risk_auditee": False,
        "cognizant_agency": "UNK", "entity_type": "UNK",
        "size_band": "UNK", "total_amount_expended": np.nan,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    # Categorical/boolean feature columns -> plain strings, so the imputer and
    # one-hot encoder treat them as categories (pandas 'category'/bool dtypes
    # otherwise confuse the numeric-imputer dtype check).
    for col in set(PROFILE_CATEGORICAL + ENRICHED_CATEGORICAL):
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), None)
            df[col] = df[col].map(lambda v: str(v) if v is not None else None)
    return df


def run(args):
    df = load_table(args)
    df = ensure_columns(df)
    df = add_prior_year_history(df)

    if df[TARGET].nunique() < 2:
        sys.exit("Target has only one class -- need both findings and non-findings.")

    years = sorted(df["audit_year"].dropna().unique())
    if len(years) < 2:
        sys.exit("Need at least two audit years to do a forward (train-past/"
                 "test-future) split.")
    test_year = years[-1]
    if args.test_year is not None:
        test_year = args.test_year
        if test_year not in years:
            sys.exit(f"--test-year {test_year} is not present in the data.")
    train_years = years[:-1]
    if args.test_year is not None:
        train_years = [y for y in years if y < test_year]
        if not train_years:
            sys.exit("--test-year must leave at least one earlier training year.")
    train = df[df["audit_year"].isin(train_years)]
    test = df[df["audit_year"] == test_year]

    if train[TARGET].nunique() < 2:
        sys.exit("Training target has only one class -- need both findings and "
                 "non-findings before the test year.")
    if test[TARGET].nunique() < 2:
        sys.exit("Test target has only one class -- choose another --test-year "
                 "or pull more data before reporting AUC.")

    print("=" * 64)
    print(f"Forward split: train on {int(train_years[0])}-{int(train_years[-1])} "
          f"({len(train):,} rows) -> test on {int(test_year)} ({len(test):,} rows)")
    base = test[TARGET].mean()
    print(f"Test-year base rate of any finding: {base:.1%} "
          f"(this is the no-skill benchmark)")
    print("=" * 64)

    y_tr, y_te = train[TARGET].values, test[TARGET].values

    # ---- RQ3: profile-only baseline vs. enriched feature set ----
    print("\n### Feature set A: PROFILE ONLY (who the org is) ###")
    Xtr_a, Xte_a = train[PROFILE_NUMERIC + PROFILE_CATEGORICAL], test[PROFILE_NUMERIC + PROFILE_CATEGORICAL]
    lr_a = evaluate("Logistic regression (profile)",
                    make_logreg(PROFILE_NUMERIC, PROFILE_CATEGORICAL),
                    Xtr_a, y_tr, Xte_a, y_te, args.target_recall)
    gb_a = evaluate("Gradient boosting (profile)",
                    make_gbm(PROFILE_NUMERIC, PROFILE_CATEGORICAL),
                    Xtr_a, y_tr, Xte_a, y_te, args.target_recall)

    print("\n### Feature set B: ENRICHED (+ program count, + prior-year history) ###")
    Xtr_b, Xte_b = train[ENRICHED_NUMERIC + ENRICHED_CATEGORICAL], test[ENRICHED_NUMERIC + ENRICHED_CATEGORICAL]
    lr_b = evaluate("Logistic regression (enriched)",
                    make_logreg(ENRICHED_NUMERIC, ENRICHED_CATEGORICAL),
                    Xtr_b, y_tr, Xte_b, y_te, args.target_recall)
    gb_b = evaluate("Gradient boosting (enriched)",
                    make_gbm(ENRICHED_NUMERIC, ENRICHED_CATEGORICAL),
                    Xtr_b, y_tr, Xte_b, y_te, args.target_recall)

    # ---- RQ1: what drives risk? (from the best enriched model) ----
    top_features = report_top_features(gb_b["model"], Xte_b, y_te,
                                       ENRICHED_NUMERIC, ENRICHED_CATEGORICAL)
    report_calibration(y_te, gb_b["scores"])

    # ---- RQ3 verdict ----
    print("\n" + "=" * 64)
    print("RQ3 -- does enrichment help? (ROC-AUC, profile -> enriched)")
    print(f"  Logistic : {lr_a['auc']:.3f} -> {lr_b['auc']:.3f}  "
          f"({lr_b['auc']-lr_a['auc']:+.3f})")
    print(f"  Boosting : {gb_a['auc']:.3f} -> {gb_b['auc']:.3f}  "
          f"({gb_b['auc']-gb_a['auc']:+.3f})")
    best = max([lr_a, gb_a, lr_b, gb_b], key=lambda d: d["auc"])
    print(f"\nBest model: {best['name']}  (ROC-AUC {best['auc']:.3f}, "
          f"PR-AUC {best['ap']:.3f})")
    print("=" * 64)
    print("\nNext (Year-3): wrap the best model in the scoring tool (fac_score.py) "
          "and run the practitioner validation.")

    if args.results_out:
        results_path = Path(args.results_out)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target": TARGET,
            "train_years": [int(y) for y in train_years],
            "test_year": int(test_year),
            "target_recall": float(args.target_recall),
            "test_base_rate": float(base),
            "models": [serializable_result(r) for r in [lr_a, gb_a, lr_b, gb_b]],
            "best_model": serializable_result(best),
            "top_features": top_features,
            "feature_sets": {
                "profile_numeric": PROFILE_NUMERIC,
                "profile_categorical": PROFILE_CATEGORICAL,
                "enriched_numeric": ENRICHED_NUMERIC,
                "enriched_categorical": ENRICHED_CATEGORICAL,
            },
        }
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved committee-review metrics -> {results_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FAC findings-prediction modeling stage.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="analytic_audit_level.parquet from fac_ingest.py")
    src.add_argument("--demo", action="store_true",
                     help="Run on a synthetic table with the same schema.")
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="Recall target for threshold selection (default 0.80).")
    p.add_argument("--test-year", type=int, default=None,
                   help="Held-out future year. Defaults to the latest year.")
    p.add_argument("--results-out", default=None,
                   help="Optional JSON file for committee-review metrics.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
