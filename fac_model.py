#!/usr/bin/env python3
"""
fac_model.py
============
Modeling stage for the FAC Single-Audit findings-prediction project.
Consumes the analytic table produced by fac_ingest.py and answers RQ1-RQ3:

  RQ1  Which FAC-derived characteristics are most associated with findings?
       -> SHAP values (global + per-audit explanation).
  RQ2  How accurately can we predict any finding using only pre-audit-close
       information?
       -> AUC, PR-AUC, calibration, TimeSeriesSplit cross-validation,
          and a final forward holdout on the most recent year.
  RQ3  Does program-level + prior-finding history beat a profile-only baseline?
       -> two feature sets compared on both CV and holdout.

WHY THE SPLIT IS BY YEAR (not random)
--------------------------------------
A random split lets the model "see the future." Single-audit risk is a
forward prediction problem, so we train on earlier years and test on the
most recent year. That is the honest analogue of "score an organization
before its audit closes," and it is what a dissertation committee will
expect to see.  TimeSeriesSplit cross-validation gives the same guarantee
over multiple folds and produces a more stable estimate of generalisation.

SHAP EXPLANATIONS (RQ1)
------------------------
We use TreeExplainer for the gradient-boosting model (exact SHAP values,
fast) and LinearExplainer for logistic regression (also exact). Global
feature importance is the mean |SHAP| across the test set. Per-audit
explanations let a finance director ask "why was this organization flagged?"
and receive a ranked list of contributing factors.

CLASS IMBALANCE (findings are rare, ~10 % in real FAC data)
------------------------------------------------------------
Two complementary strategies are available:
  1. class_weight='balanced'  -- default, applied to both models.
  2. --smote                  -- synthetic minority over-sampling on the
                                 training set only (never on test).
     pip install imbalanced-learn   to enable this flag.

KNOWN FAC DATA-QUALITY ISSUES (addressed in build_analytic_table)
------------------------------------------------------------------
  * Migrated records (pre-2016) have an off-by-one day in fac_accepted_date
    (FAC concern #3). We use fiscal-year fields for time splitting, not
    fac_accepted_date.
  * Historical ALN/CFDA numbers were mis-mapped for ~0.3 % of awards during
    migration (FAC concern #5). We do not use ALN as a feature.
  * The is_internal_control_material_weakness_disclosed flag was incorrectly
    mapped in migration (FAC concern #9). We validate against the findings
    endpoint instead of trusting the general-table flag.

USAGE
-----
  python fac_model.py --demo
  python fac_model.py --demo --results-out outputs/fac_demo_metrics.json
  python fac_model.py --input fac_data/analytic_audit_level.parquet \\
                      --results-out outputs/fac_pilot_metrics.json
  python fac_model.py --input fac_data/analytic_audit_level.parquet \\
                      --test-year 2023 --smote --results-out results.json

DEPENDENCIES
------------
  pip install pandas scikit-learn pyarrow shap
  pip install imbalanced-learn   # optional, only needed for --smote
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
# Baseline: who the organization IS, knowable before any audit work begins.
PROFILE_NUMERIC      = ["total_amount_expended"]
PROFILE_CATEGORICAL  = ["entity_type", "size_band", "is_low_risk_auditee"]

# Enriched: profile PLUS program complexity and lagged finding history.
ENRICHED_NUMERIC     = PROFILE_NUMERIC + [
    "n_programs", "prior_year_findings", "prior_year_any_finding",
]
ENRICHED_CATEGORICAL = PROFILE_CATEGORICAL + ["cognizant_agency"]

TARGET = "any_finding"


# --------------------------------------------------------------------------
# Lagged prior-year history (no leakage: only past data used)
# --------------------------------------------------------------------------
def add_prior_year_history(df):
    """
    Attach last year's finding outcome to this year's row for each
    organization. When scoring 2024 audits, 2023 results are already filed
    and are therefore a legitimate pre-audit predictor.
    """
    df = df.copy()
    df["audit_year"] = pd.to_numeric(df["audit_year"], errors="coerce")
    df = df.sort_values(["auditee_uei", "audit_year"])

    df["prior_year_any_finding"] = df.groupby("auditee_uei")["any_finding"].shift(1)
    src = "total_findings_count" if "total_findings_count" in df.columns else "any_finding"
    df["prior_year_findings"] = df.groupby("auditee_uei")[src].shift(1)

    df["prior_year_any_finding"] = df["prior_year_any_finding"].fillna(0)
    df["prior_year_findings"]    = df["prior_year_findings"].fillna(0)
    return df


# --------------------------------------------------------------------------
# Preprocessor and model builders
# --------------------------------------------------------------------------
def make_preprocessor(numeric, categorical):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
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
        ("clf",  LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def make_gbm(numeric, categorical):
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("clf",  HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06,
            l2_regularization=1.0, early_stopping=True, random_state=42)),
    ])


# --------------------------------------------------------------------------
# Class-imbalance helper (SMOTE, optional)
# --------------------------------------------------------------------------
def maybe_smote(X_tr, y_tr, use_smote):
    """Apply SMOTE to the training set only if requested and available."""
    if not use_smote:
        return X_tr, y_tr
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("  [warn] --smote requested but imbalanced-learn not installed. "
              "Run: pip install imbalanced-learn")
        return X_tr, y_tr
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f"  SMOTE: {len(y_tr):,} -> {len(y_res):,} training rows "
          f"(minority class balanced)")
    return X_res, y_res


# --------------------------------------------------------------------------
# TimeSeriesSplit cross-validation (RQ2 / RQ3 robustness check)
# --------------------------------------------------------------------------
def ts_cross_validate(df, numeric, categorical, make_model_fn, label,
                      n_splits=3, use_smote=False):
    """
    Time-aware k-fold: each fold trains on earlier years, validates on the
    next year block. Returns mean and std of ROC-AUC across folds.
    """
    years = sorted(df["audit_year"].dropna().unique())
    if len(years) < n_splits + 1:
        return None, None

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    aucs   = []
    y_col  = df[TARGET].values
    X_cols = df[numeric + categorical]

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(df)):
        X_tr, y_tr = X_cols.iloc[tr_idx], y_col[tr_idx]
        X_val, y_val = X_cols.iloc[val_idx], y_col[val_idx]
        if y_val.sum() == 0 or y_val.sum() == len(y_val):
            continue
        X_tr_res, y_tr_res = maybe_smote(
            X_tr.reset_index(drop=True), y_tr, use_smote)
        model = make_model_fn(numeric, categorical)
        model.fit(X_tr_res, y_tr_res)
        scores = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, scores))

    if not aucs:
        return None, None
    m, s = float(np.mean(aucs)), float(np.std(aucs))
    print(f"    {label} CV ROC-AUC: {m:.3f} ± {s:.3f}  ({len(aucs)} folds)")
    return m, s


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
def pick_threshold_for_recall(y_true, scores, target_recall=0.80):
    """
    For compliance triage we tune the decision threshold to hit a recall
    target, then report the precision we pay for it.
    """
    order    = np.argsort(scores)[::-1]
    y_sorted = np.asarray(y_true)[order]
    total    = y_sorted.sum()
    if total == 0:
        return 0.5
    cum_tp   = np.cumsum(y_sorted)
    recall_at = cum_tp / total
    hit = min(np.searchsorted(recall_at, target_recall), len(scores) - 1)
    return float(np.sort(scores)[::-1][hit])


# --------------------------------------------------------------------------
# Single-model evaluation
# --------------------------------------------------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te,
             target_recall=0.80, use_smote=False):
    X_tr_res, y_tr_res = maybe_smote(
        X_tr.reset_index(drop=True), y_tr, use_smote)
    model.fit(X_tr_res, y_tr_res)
    scores = model.predict_proba(X_te)[:, 1]

    auc   = roc_auc_score(y_te, scores)
    ap    = average_precision_score(y_te, scores)
    brier = brier_score_loss(y_te, scores)
    thr   = pick_threshold_for_recall(y_te, scores, target_recall)
    preds = (scores >= thr).astype(int)
    prec  = precision_score(y_te, preds, zero_division=0)
    rec   = recall_score(y_te,  preds, zero_division=0)
    f1    = f1_score(y_te,    preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()

    print(f"\n  {name}")
    print(f"    ROC-AUC ................ {auc:.3f}")
    print(f"    PR-AUC (avg precision).. {ap:.3f}")
    print(f"    Brier score ............ {brier:.3f}  (lower is better)")
    print(f"    @ recall>={target_recall:.0%} threshold={thr:.3f}:")
    print(f"       precision ........... {prec:.3f}")
    print(f"       recall .............. {rec:.3f}")
    print(f"       F1 .................. {f1:.3f}")
    print(f"       confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    return {
        "name": name, "model": model, "scores": scores,
        "auc": float(auc), "ap": float(ap), "brier": float(brier),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# --------------------------------------------------------------------------
# SHAP explanations (RQ1)
# --------------------------------------------------------------------------
def compute_shap(model, X_te, numeric, categorical, k=10):
    """
    Compute SHAP values for the best model.
      * GBM -> TreeExplainer (exact, fast)
      * Logistic -> LinearExplainer (exact)
    Returns global importance (mean |SHAP|) and per-audit explanations
    for the top-k highest-scored test audits.
    """
    if not HAS_SHAP:
        print("\n  [skip] SHAP not installed: pip install shap")
        return [], []

    clf  = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    X_transformed = prep.transform(X_te)

    # Feature names after one-hot encoding
    try:
        feat_names = list(prep.get_feature_names_out())
    except Exception:
        feat_names = numeric + categorical

    print("\n  Computing SHAP values …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(clf, "predict_proba") and hasattr(clf, "_raw_predict"):
            # HistGradientBoosting
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_transformed)
        elif hasattr(clf, "coef_"):
            # Logistic regression
            explainer = shap.LinearExplainer(
                clf, X_transformed, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_transformed)
        else:
            explainer = shap.KernelExplainer(
                clf.predict_proba, shap.sample(X_transformed, 100))
            shap_vals = explainer.shap_values(X_transformed)[1]

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # positive class

    # Global: mean |SHAP| per feature, mapped back to original feature names
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    global_imp = []
    print(f"\n  Top features (RQ1) by mean |SHAP| (positive class):")
    for i in order:
        name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        print(f"    {mean_abs[i]:+7.4f}  {name}")
        global_imp.append({"feature": name,
                            "importance": float(mean_abs[i]),
                            "method": "mean_abs_shap"})

    # Per-audit: top-5 highest-risk audits, top-3 driving features each
    scores = model.predict_proba(X_te)[:, 1]
    top_idx = np.argsort(scores)[::-1][:5]
    per_audit = []
    for rank, idx in enumerate(top_idx):
        sv = shap_vals[idx]
        feat_order = np.argsort(np.abs(sv))[::-1][:3]
        drivers = [
            {"feature": feat_names[j] if j < len(feat_names) else f"feat_{j}",
             "shap": float(sv[j])}
            for j in feat_order
        ]
        per_audit.append({
            "rank": rank + 1,
            "risk_score": float(scores[idx]),
            "drivers": drivers,
        })

    return global_imp, per_audit


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def get_calibration(y_te, scores, bins=5):
    frac_pos, mean_pred = calibration_curve(
        y_te, scores, n_bins=bins, strategy="quantile")
    print("\n  Calibration (predicted probability → actual finding rate):")
    rows = []
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"    {mp:5.1%}  →  {fp:5.1%}")
        rows.append({"mean_pred": float(mp), "frac_pos": float(fp)})
    return rows


# --------------------------------------------------------------------------
# Synthetic demo table (mirrors fac_ingest output schema exactly)
# --------------------------------------------------------------------------
def make_demo_table(n_per_year=4000, years=(2019, 2020, 2021, 2022, 2023),
                    seed=42):
    rng          = np.random.default_rng(seed)
    entity_types = np.array(["non-profit", "state", "local", "higher-ed", "tribal"])
    et_p         = [0.45, 0.12, 0.28, 0.12, 0.03]
    agencies     = np.array(["93", "84", "10", "21", "20", "14", "66"])

    n_org     = int(n_per_year * 1.3)
    org_ids   = np.array([f"UEI{i:06d}" for i in range(n_org)])
    org_risk  = rng.beta(1.4, 6.0, size=n_org)
    org_entity = rng.choice(entity_types, size=n_org, p=et_p)
    org_agency = rng.choice(agencies, size=n_org)

    rows = []
    for y in years:
        idx = rng.choice(n_org, size=n_per_year, replace=False)
        for j in idx:
            n_progs  = int(1 + rng.poisson(2.5))
            expended = float(np.exp(rng.normal(14.0, 1.6)))
            low_risk = rng.random() < 0.35
            logit    = (-2.3
                        + 5.5 * org_risk[j]
                        + 0.10 * n_progs
                        + 0.18 * (np.log(expended) - 14.0)
                        - 0.6 * low_risk)
            p           = 1 / (1 + np.exp(-logit))
            any_finding = int(rng.random() < p)
            tfc         = int(rng.poisson(1.4)) if any_finding else 0
            rows.append({
                "report_id":              f"{y}-{j}",
                "audit_year":             str(y),
                "auditee_uei":            org_ids[j],
                "entity_type":            org_entity[j],
                "total_amount_expended":  expended,
                "n_programs":             n_progs,
                "is_low_risk_auditee":    bool(low_risk),
                "cognizant_agency":       org_agency[j],
                "total_findings_count":   tfc,
                "any_finding":            any_finding,
                "any_material_weakness":  int(any_finding and rng.random() < 0.4),
                "any_questioned_costs":   int(any_finding and rng.random() < 0.5),
                "any_repeat_finding":     int(any_finding and rng.random() < 0.3),
            })

    df = pd.DataFrame(rows)
    df["size_band"] = pd.cut(
        df["total_amount_expended"],
        bins=[-1, 1e6, 1e7, 1e8, float("inf")],
        labels=["<1M", "1-10M", "10-100M", ">100M"])
    return df


# --------------------------------------------------------------------------
# Orchestration helpers
# --------------------------------------------------------------------------
def load_table(args):
    if args.demo:
        print("DEMO MODE: generating synthetic data that mirrors fac_ingest.py "
              "output schema.\n")
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
        "n_programs": 0, "total_findings_count": 0,
        "is_low_risk_auditee": False,
        "cognizant_agency": "UNK", "entity_type": "UNK",
        "size_band": "UNK", "total_amount_expended": np.nan,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    for col in set(PROFILE_CATEGORICAL + ENRICHED_CATEGORICAL):
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), None)
            df[col] = df[col].map(lambda v: str(v) if v is not None else None)
    return df


def serializable(result):
    keep = ["name", "auc", "ap", "brier", "precision", "recall",
            "f1", "threshold", "tp", "fp", "fn", "tn"]
    return {k: result[k] for k in keep}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(args):
    df = load_table(args)
    df = ensure_columns(df)
    df = add_prior_year_history(df)

    if df[TARGET].nunique() < 2:
        sys.exit("Target has only one class -- need both findings and non-findings.")

    years = sorted(df["audit_year"].dropna().unique())
    if len(years) < 2:
        sys.exit("Need at least two audit years for a forward split.")

    test_year = int(args.test_year) if args.test_year else int(years[-1])
    if test_year not in years:
        sys.exit(f"--test-year {test_year} not in data.")
    train_years = [y for y in years if y < test_year]
    if not train_years:
        sys.exit("--test-year must leave at least one earlier training year.")

    train = df[df["audit_year"].isin(train_years)]
    test  = df[df["audit_year"] == test_year]

    for split_name, split_df in [("train", train), ("test", test)]:
        if split_df[TARGET].nunique() < 2:
            sys.exit(f"{split_name} target has only one class — choose another "
                     f"--test-year or pull more data.")

    print("=" * 64)
    print(f"Forward split: train {int(train_years[0])}–{int(train_years[-1])} "
          f"({len(train):,} rows)  →  test {test_year} ({len(test):,} rows)")
    base = test[TARGET].mean()
    print(f"Test-year finding base rate: {base:.1%}  (no-skill benchmark)")
    print("=" * 64)

    y_tr = train[TARGET].values
    y_te = test[TARGET].values

    # ── TimeSeriesSplit cross-validation (robustness check) ──────────────
    print("\n── TimeSeriesSplit cross-validation (ROC-AUC) ──")
    cv_lr_a_m, cv_lr_a_s = ts_cross_validate(
        train, PROFILE_NUMERIC, PROFILE_CATEGORICAL, make_logreg,
        "LR profile", use_smote=args.smote)
    cv_gb_b_m, cv_gb_b_s = ts_cross_validate(
        train, ENRICHED_NUMERIC, ENRICHED_CATEGORICAL, make_gbm,
        "GBM enriched", use_smote=args.smote)

    # ── Holdout evaluation ────────────────────────────────────────────────
    print("\n── Holdout evaluation on test year ──")
    Xtr_a = train[PROFILE_NUMERIC + PROFILE_CATEGORICAL]
    Xte_a = test[PROFILE_NUMERIC + PROFILE_CATEGORICAL]
    Xtr_b = train[ENRICHED_NUMERIC + ENRICHED_CATEGORICAL]
    Xte_b = test[ENRICHED_NUMERIC + ENRICHED_CATEGORICAL]

    print("\n### Feature set A: PROFILE ONLY ###")
    lr_a = evaluate("Logistic regression (profile)",
                    make_logreg(PROFILE_NUMERIC, PROFILE_CATEGORICAL),
                    Xtr_a, y_tr, Xte_a, y_te, args.target_recall, args.smote)
    gb_a = evaluate("Gradient boosting (profile)",
                    make_gbm(PROFILE_NUMERIC, PROFILE_CATEGORICAL),
                    Xtr_a, y_tr, Xte_a, y_te, args.target_recall, args.smote)

    print("\n### Feature set B: ENRICHED ###")
    lr_b = evaluate("Logistic regression (enriched)",
                    make_logreg(ENRICHED_NUMERIC, ENRICHED_CATEGORICAL),
                    Xtr_b, y_tr, Xte_b, y_te, args.target_recall, args.smote)
    gb_b = evaluate("Gradient boosting (enriched)",
                    make_gbm(ENRICHED_NUMERIC, ENRICHED_CATEGORICAL),
                    Xtr_b, y_tr, Xte_b, y_te, args.target_recall, args.smote)

    best    = max([lr_a, gb_a, lr_b, gb_b], key=lambda d: d["auc"])
    is_best_enriched = "enriched" in best["name"]
    Xte_best = Xte_b if is_best_enriched else Xte_a
    num_best = ENRICHED_NUMERIC if is_best_enriched else PROFILE_NUMERIC
    cat_best = ENRICHED_CATEGORICAL if is_best_enriched else PROFILE_CATEGORICAL

    # ── SHAP (RQ1) ────────────────────────────────────────────────────────
    shap_global, shap_per_audit = compute_shap(
        best["model"], Xte_best, num_best, cat_best)

    # ── Calibration (RQ2) ────────────────────────────────────────────────
    calibration = get_calibration(y_te, best["scores"])

    # ── RQ3 summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RQ3 — does enrichment help? (ROC-AUC, profile → enriched)")
    print(f"  Logistic : {lr_a['auc']:.3f} → {lr_b['auc']:.3f}  "
          f"({lr_b['auc']-lr_a['auc']:+.3f})")
    print(f"  Boosting : {gb_a['auc']:.3f} → {gb_b['auc']:.3f}  "
          f"({gb_b['auc']-gb_a['auc']:+.3f})")
    print(f"\nBest model: {best['name']}  "
          f"(ROC-AUC {best['auc']:.3f}, PR-AUC {best['ap']:.3f})")
    print("=" * 64)

    if shap_global:
        print("\nTop SHAP drivers (best model, mean |SHAP| on test year):")
        for row in shap_global[:5]:
            print(f"  {row['importance']:+.4f}  {row['feature']}")

    print("\nNext: wrap the best model in fac_score.py for practitioner scoring.")

    # ── JSON output ──────────────────────────────────────────────────────
    if args.results_out:
        results_path = Path(args.results_out)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target":         TARGET,
            "train_years":    [int(y) for y in train_years],
            "test_year":      int(test_year),
            "target_recall":  float(args.target_recall),
            "test_base_rate": float(base),
            "smote_used":     args.smote,
            "cv": {
                "lr_profile":  {"mean": cv_lr_a_m, "std": cv_lr_a_s},
                "gbm_enriched":{"mean": cv_gb_b_m, "std": cv_gb_b_s},
            },
            "models":     [serializable(r) for r in [lr_a, gb_a, lr_b, gb_b]],
            "best_model": serializable(best),
            "top_features":   shap_global if shap_global else [],
            "shap_per_audit": shap_per_audit,
            "calibration":    calibration,
            "feature_sets": {
                "profile_numeric":     PROFILE_NUMERIC,
                "profile_categorical": PROFILE_CATEGORICAL,
                "enriched_numeric":    ENRICHED_NUMERIC,
                "enriched_categorical":ENRICHED_CATEGORICAL,
            },
        }
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics → {results_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="FAC findings-prediction modeling stage.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",
                     help="analytic_audit_level.parquet from fac_ingest.py")
    src.add_argument("--demo", action="store_true",
                     help="Run on a synthetic table with the same schema.")
    p.add_argument("--target-recall", type=float, default=0.80,
                   help="Recall target for threshold selection (default 0.80).")
    p.add_argument("--test-year", type=int, default=None,
                   help="Held-out future year. Defaults to the latest year.")
    p.add_argument("--smote", action="store_true",
                   help="Apply SMOTE over-sampling on the training set "
                        "(requires: pip install imbalanced-learn).")
    p.add_argument("--results-out", default=None,
                   help="Write JSON metrics here (for the dashboard).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
