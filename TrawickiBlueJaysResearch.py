#!/usr/bin/env python3
"""
TORONTO BLUE JAYS — BASEBALL RESEARCH QUESTIONNAIRE, QUESTION 5

Applicant: Marek Trawicki, Ph.D.
File: TrawickiBlueJaysResearch.py
Date: November 22, 2026
Purpose: Demonstrate a transparent chronological MLB hitter forecasting workflow.
Research question: Does fixed-penalty ridge regression reduce next-season OBP
    prediction error relative to a current-season empirical-shrinkage baseline?

DATA AND DESIGN
    Input: Public Lahman-style Batting.csv with required columns
    playerID, yearID, AB, H, BB, HBP, SF, SO.
    Unit of analysis: Player-season. Multiple team stints are combined first.
    Projection target: Next calendar year's OBP, conditional on both seasons
    satisfying an APPROXIMATE 200-PA cutoff based on AB+BB+HBP+SF.
    Validation: Expanding-window out-of-time evaluation; no future outcomes are
    used in model fitting for a given prediction year.
    Inference: Paired error differences with player-cluster bootstrap intervals.

USAGE
    python TrawickiBlueJaysResearch.py --self-test
    python TrawickiBlueJaysResearch.py --input Batting.csv --output Results
    python TrawickiBlueJaysResearch.py --download --data-dir Data --output Results

PACKAGES
    Python 3.11+, pandas, numpy, scikit-learn, matplotlib.

OUTPUTS
    forward_predictions.csv, metrics_by_year.csv, research_summary.txt,
    forward_mae.png, data_provenance.json (under --output for real-data runs).

INTERPRETATION AND PROVENANCE
    This is an AI-assisted screening sample prepared for Marek Trawicki;
    it is NOT a conversion of an independently validated MATLAB study.
    Synthetic-test outputs are not empirical MLB findings. The real-data
    workflow must be independently reviewed on the acquired dataset.
    Selection based on NEXT-SEASON playing time limits generalization to
    all players who could have been projected at the forecast origin.
    Public dataset availability, citation and permissible redistribution
    should be reviewed before external submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ===========================================================================
# SECTION 1: PUBLIC DATA SOURCE, INPUT SCHEMA, AND MODEL INPUTS
# ===========================================================================

# This public GitHub URL is a MIRROR of the SABR Lahman database. SABR is the
# authoritative publisher (https://sabr.org/lahman-database/). The version
# available as of November 2026 covers MLB batting seasons through 2025.
# Because the mirror points to a mutable branch, every actual download is
# recorded with a SHA-256 checksum and year coverage in data_provenance.json.

PUBLIC_DATA_URL = (
    "https://raw.githubusercontent.com/corbtastik/lahman-baseball-db/"
    "main/Batting.csv"
)

# A secondary public mirror provides automatic failover if the first host
# cannot be reached. This mirrors SABR's historical Batting.csv release.

BACKUP_DATA_URL = (
    "https://raw.githubusercontent.com/cbwinslow/lahman-database-csv/"
    "main/data/Batting.csv"
)

REQUIRED = ["playerID", "yearID", "AB", "H", "BB", "HBP", "SF", "SO"]
FEATURES = [
    "obp", "bb_rate", "so_rate", "log_pa", "prev_obp",
    "prev_available", "last3_obp", "league_obp", "baseline_obp"
]


# ===========================================================================
# SECTION 2: PUBLIC CSV ACQUISITION AND DATA PROVENANCE
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: sha256_file(path)
#
# Purpose: Fingerprint the exact bytes of a cached or downloaded batting CSV.
# Input:   path -- location of the file to audit (pathlib.Path).
# Returns: A 64-character hexadecimal SHA-256 digest.
# Method:  Read in bounded binary chunks to avoid loading the entire file.
# Note:    A checksum documents reproducibility; it does not prove that a
#          third-party mirror contains the official original dataset.
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate a stable fingerprint so the exact dataset can be identified."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# FUNCTION: acquire_public_data(data_dir)
#
# Purpose: Acquire and cache real MLB batting records without a manual step.
# Input:   data_dir -- project Data folder where Batting.csv is stored.
# Returns: (CSV path, source description) for downstream provenance reporting.
# Method:  Reuse a cached file; otherwise try public mirrors in sequence,
#          stream to a temporary file, check size/schema, then atomically save.
# Failure: Raise an informative exception if every candidate source fails.
# Note:    Cached files are intentionally left unchanged on later runs.
# ---------------------------------------------------------------------------

def acquire_public_data(data_dir: Path) -> tuple[Path, str]:
    """Retrieve the real public Batting.csv once, with an automatic fallback.

    Both candidate URLs are public GitHub mirrors of the SABR batting table.
    Downloads stream to a temporary file and undergo a schema/size check.
    An existing cached file is reused unchanged to preserve reproducibility.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "Batting.csv"
    source_record = data_dir / "Batting.source.txt"
    if destination.exists():
        print(f"Using locally cached batting data: {destination}")
        provenance_source = (source_record.read_text(encoding="utf-8").strip()
                             if source_record.is_file() else
                             "Existing cached file; original download URL unknown")
        return destination, provenance_source

    partial = data_dir / "Batting.download.tmp"
    last_error = None
    for url in (PUBLIC_DATA_URL, BACKUP_DATA_URL):
        print(f"Attempting actual public MLB batting-data download from: {url}")
        request = Request(url, headers={"User-Agent": "BlueJays-Research-Sample/3.0"})
        try:
            with urlopen(request, timeout=90) as response, partial.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            if partial.stat().st_size < 100_000:
                raise ValueError("Downloaded CSV is unexpectedly small")
            header = pd.read_csv(partial, nrows=0).columns
            if not set(REQUIRED).issubset(set(header)):
                raise ValueError("Downloaded CSV lacks required batting columns")
            os.replace(partial, destination)
            source_record.write_text(url + "\n", encoding="utf-8")
            print(f"Real batting data cached at: {destination}")
            return destination, url
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            print(f"Source unavailable: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Both public batting-data mirrors failed; check internet access "
        "and the mirror URLs. No real-data analysis was performed."
    ) from last_error


# ---------------------------------------------------------------------------
# FUNCTION: dataset_provenance(path, source)
#
# Purpose: Record the source, coverage and identity of the raw batting data.
# Inputs:  path -- actual input CSV; source -- recorded download/source route.
# Returns: A dictionary suitable for data_provenance.json.
# Checks:  At least one season must parse as a valid numeric year.
# Note:    Source metadata is not an independent certification of authenticity.
# ---------------------------------------------------------------------------

def dataset_provenance(path: Path, source: str) -> dict:
    """Describe the exact raw file processed; no unverifiable source claims."""
    raw = pd.read_csv(path, usecols=["playerID", "yearID"], low_memory=False)
    years = pd.to_numeric(raw.yearID, errors="coerce").dropna()
    if years.empty:
        raise ValueError("Input CSV has no usable season years.")
    return {
        "dataset": "Lahman-style Batting.csv (public historical batting records)",
        "publisher": "Society for American Baseball Research (SABR)",
        "publisher_page": "https://sabr.org/lahman-database/",
        "source_location": source,
        "file_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "raw_rows": int(len(raw)),
        "raw_min_year": int(years.min()),
        "raw_max_year": int(years.max()),
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Source URL identifies the download route, not an independent authenticity check."
    }


# ===========================================================================
# SECTION 3: CLEANING AND PLAYER-SEASON FEATURE CONSTRUCTION
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: load_and_prepare(filepath, min_year, min_pa, shrink_pa)
#
# Purpose: Convert Lahman batting stints into modeled player-season rows.
# Inputs:  filepath -- Batting.csv; min_year -- earliest usable season;
#          min_pa -- approximate qualifying PA in both seasons;
#          shrink_pa -- shrinkage strength for the comparison baseline.
# Returns: One row per eligible forecast-origin player-season, including
#          features known at origin t and the observed target at t+1.
# Method:  Validate identifiers/counts, combine traded-player stints,
#          calculate descriptive features and assemble next-year labels.
# Limits:  PA is approximated as AB+BB+HBP+SF. Requiring qualification at t+1
#          creates a hindsight-selected evaluation population, not a forecast
#          covering every hitter eligible at the original prediction date.
# ---------------------------------------------------------------------------

def load_and_prepare(filepath: Path, min_year: int = 2000,
                     min_pa: int = 200, shrink_pa: float = 200.0) -> pd.DataFrame:
    """Create player-season features and t+1 outcomes.

    Features for origin t use only information available through t. Joining
    to t+1 records is performed solely to construct labels, although the
    two-season eligibility restriction creates a hindsight-selected cohort.
    """
    raw = pd.read_csv(filepath, low_memory=False)
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Batting file is missing required columns: {missing}")
    raw = raw[REQUIRED].copy()
    # -----------------------------------------------------------------------
    # 3A. IDENTIFIERS AND CALENDAR YEARS
    # The player identifier must be cleaned before string conversion, or
    # missing identifiers can accidentally become a shared "nan" player.
    # -----------------------------------------------------------------------

    # Clean identifiers BEFORE converting to ordinary Python strings.  The
    # previous version turned missing values into the literal text "nan",
    # which caused unrelated missing-ID records to be joined as one player.
    ids = raw["playerID"].astype("string").str.strip()
    invalid_id = (
        ids.isna()
        | ids.eq("").fillna(True)
        | ids.str.lower().isin(["nan", "none", "null"])
    )
    raw = raw.loc[~invalid_id].copy()
    raw["playerID"] = ids.loc[~invalid_id]
    raw["yearID"] = pd.to_numeric(raw["yearID"], errors="coerce")
    raw = raw.dropna(subset=["yearID"])
    # Numeric errors are treated as missing and then filled with zeros.  This
    # pragmatic convention should be audited against any real-data release.
    raw = raw.loc[raw.yearID >= min_year].copy()
    # -----------------------------------------------------------------------
    # 3B. COUNT DATA VALIDATION
    # Reject invalid core batting columns instead of silently manufacturing
    # zero at-bats, hits, or walks. Some ancillary event counts (HBP, SF, SO)
    # may be blank in historical source releases; the zero-fill convention is
    # a deliberate approximation, documented here rather than hidden.
    # -----------------------------------------------------------------------

    core = ["AB", "H", "BB"]
    for c in ["AB", "H", "BB", "HBP", "SF", "SO"]:
        values = pd.to_numeric(raw[c], errors="coerce")
        if c in core and values.isna().any():
            raise ValueError(f"Missing/non-numeric modern batting count in {c}")
        if (values.dropna() < 0).any():
            raise ValueError(f"Negative batting count in {c}")
        raw[c] = values.fillna(0)
    if (raw["H"] > raw["AB"]).any():
        raise ValueError("Hits exceed at-bats on at least one input row.")
    if raw.empty:
        raise ValueError("No records remain after the minimum-year filter.")
    # -----------------------------------------------------------------------
    # 3C. PLAYER-SEASON AGGREGATION
    # A traded player can appear on multiple rows in the same year. Sum
    # counting statistics across stints before deriving rates to avoid
    # treating one player as multiple independent observations.
    # -----------------------------------------------------------------------

    # Multiple-team stints in the same season are combined BEFORE computing rates.
    totals = raw.groupby(["playerID", "yearID"], as_index=False)[
        ["AB", "H", "BB", "HBP", "SF", "SO"]].sum()
    totals = totals.sort_values(["playerID", "yearID"]).reset_index(drop=True)
    # -----------------------------------------------------------------------
    # 3D. STATISTICS AVAILABLE AT THE FORECAST ORIGIN
    # Compute an APPROXIMATE plate-appearance denominator and OBP. It does
    # not include every official PA event; this is an explicit limitation.
    # -----------------------------------------------------------------------

    # AB+BB+HBP+SF is an approximate PA denominator: sacrifice bunts and
    # certain rare events are omitted.  Do not describe this as official PA.
    totals["PA"] = totals[["AB", "BB", "HBP", "SF"]].sum(axis=1)
    totals["onbase"] = totals[["H", "BB", "HBP"]].sum(axis=1)
    totals = totals.loc[totals.PA > 0].copy()
    totals["obp"] = totals.onbase / totals.PA
    totals["bb_rate"] = totals.BB / totals.PA
    totals["so_rate"] = totals.SO / totals.PA
    totals["log_pa"] = np.log1p(totals.PA)

    # -----------------------------------------------------------------------
    # 3E. PUBLIC LEAGUE BASELINE AND EMPIRICAL SHRINKAGE
    # Shrink current-season OBP toward a season-t league reference. This
    # comparator is inexpensive to calculate and is known at forecast time.
    # -----------------------------------------------------------------------

    # A season-specific league reference is computed only using season t.
    # This is available at the end-of-season forecast origin.
    league = totals.groupby("yearID")[["onbase", "PA"]].sum()
    league["league_obp"] = league.onbase / league.PA
    totals = totals.join(league["league_obp"], on="yearID")
    totals["baseline_obp"] = (
        totals.onbase + shrink_pa * totals.league_obp
    ) / (totals.PA + shrink_pa)

    # -----------------------------------------------------------------------
    # 3F. PERSONAL HISTORY FEATURES
    # The preceding observed season is not necessarily the previous calendar
    # season, while the rolling three-observation feature includes year t.
    # -----------------------------------------------------------------------

    # Keep the preceding observed season separate from the current season.
    # The three-observation history below INCLUDES season t, never season t+1.
    g = totals.groupby("playerID", sort=False)
    totals["prev_obp"] = g.obp.shift(1)
    totals["prev_available"] = totals.prev_obp.notna().astype(int)
    # Rolling window refers to the last three OBSERVED player seasons, not 3 calendar years.
    last3_onbase = g.onbase.transform(lambda s: s.rolling(3, min_periods=1).sum())
    last3_pa = g.PA.transform(lambda s: s.rolling(3, min_periods=1).sum())
    totals["last3_obp"] = last3_onbase / last3_pa

    # -----------------------------------------------------------------------
    # 3G. FORECAST TARGET AND ELIGIBILITY
    # Attach outcomes observed at t+1 while keeping them OUT of predictors.
    # The next-season PA filter is a future-dependent evaluation restriction;
    # this is a retrospective conditional study, not a roster-wide forecast.
    # -----------------------------------------------------------------------

    # Outcomes are explicitly taken from calendar year t+1, never season t itself.
    outcomes = totals[["playerID", "yearID", "obp", "PA"]].rename(columns={
        "obp": "next_obp", "PA": "next_pa"
    })
    outcomes["yearID"] = outcomes["yearID"] - 1
    rows = totals.merge(outcomes, on=["playerID", "yearID"], how="inner",
                        validate="one_to_one")
    # CRITICAL LIMITATION: next_pa cannot be known when the t+1 forecast is
    # issued. This yields a hindsight-selected evaluation sample, not a
    # deployable forecast for every MLB player on an active roster.
    rows = rows.loc[(rows.PA >= min_pa) & (rows.next_pa >= min_pa)].copy()
    rows["yearID"] = rows.yearID.astype(int)
    rows["next_year"] = rows.yearID + 1
    if rows.empty:
        raise ValueError("No qualifying player-seasons with consecutive-year outcomes.")
    if rows["next_obp"].isna().any():
        raise AssertionError("Missing next-season target encountered.")
    return rows


# ===========================================================================
# SECTION 4: REGULARIZED STATISTICAL FORECASTING MODEL
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: new_model()
#
# Purpose: Define the ridge-regression comparison model consistently.
# Inputs:  None; the ridge penalty (alpha=20) is fixed in this example.
# Returns: scikit-learn Pipeline with median imputation, scaling, and ridge.
# Guard:   Each call creates an unfitted pipeline; fitting occurs exclusively
#          inside the relevant historical training window in score().
# ---------------------------------------------------------------------------

def new_model() -> Pipeline:
    """Build a reproducible pipeline with training-only scaling and imputation."""
    # A fixed regularization strength avoids test-set hyperparameter searches.
    # Preprocessing is fitted separately inside each training window.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=20.0)),
    ])


# ===========================================================================
# SECTION 5: EXPANDING-WINDOW / OUT-OF-TIME PERFORMANCE EVALUATION
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: score(rows, test_seasons)
#
# Purpose: Evaluate genuine forward-season forecasts without label leakage.
# Inputs:  rows -- prepared player-season table; test_seasons -- number of
#          most recent eligible forecast-origin seasons to consider.
# Returns: (row-level prediction table, annual model-comparison metrics).
# Method:  For each origin t, train on records whose target years are <= t;
#          predict outcomes at t+1; compare ridge against shrinkage by
#          mean absolute error (MAE) and root mean squared error (RMSE).
# Guard:   Preprocessing and model fitting use training rows only.
# Note:    The evaluation remains conditional on qualifying future PA.
# ---------------------------------------------------------------------------

def score(rows: pd.DataFrame, test_seasons: int = 5):
    """Evaluate player-season forecasts in a sequence of future target years."""
    # The most recent eligible origin years become evaluation years.  All
    # training outcomes must have occurred by each model's forecast origin.
    if test_seasons < 1:
        raise ValueError("test_seasons must be at least one")
    available_years = sorted(rows.yearID.unique().tolist())
    candidate_years = available_years[-test_seasons:]
    pred_parts = []
    year_metrics = []
    for forecast_origin in candidate_years:
        # At END of origin year t, training data through outcome year t are known,
        # so training observations may have origins <= t-1, not t itself.
        train = rows.loc[rows.yearID <= forecast_origin - 1]
        test = rows.loc[rows.yearID == forecast_origin]
        if len(train) < 100 or len(test) < 20:
            continue
        assert train.next_year.max() <= forecast_origin
        assert test.next_year.min() == forecast_origin + 1
        # Fit the entire preprocessing-and-regression pipeline on training
        # records only; the prediction year's outcomes are never passed to fit.
        fitted = new_model().fit(train[FEATURES], train.next_obp)
        test_pred = np.clip(fitted.predict(test[FEATURES]), 0.0, 1.0)
        out = test[["playerID", "yearID", "next_year", "PA", "next_pa",
                    "next_obp", "baseline_obp"]].copy()
        out["model_pred"] = test_pred
        out["baseline_abs_error"] = (out.baseline_obp - out.next_obp).abs()
        out["model_abs_error"] = (out.model_pred - out.next_obp).abs()
        out["baseline_squared_error"] = (out.baseline_obp - out.next_obp) ** 2
        out["model_squared_error"] = (out.model_pred - out.next_obp) ** 2
        pred_parts.append(out)
        year_metrics.append({
            "forecast_origin": int(forecast_origin),
            "target_year": int(forecast_origin + 1),
            "n_players": len(out),
            "baseline_MAE": float(out.baseline_abs_error.mean()),
            "model_MAE": float(out.model_abs_error.mean()),
            "baseline_RMSE": float(np.sqrt(out.baseline_squared_error.mean())),
            "model_RMSE": float(np.sqrt(out.model_squared_error.mean())),
        })
    if not pred_parts:
        raise ValueError("Insufficient qualifying historical rows for a forward evaluation.")
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(year_metrics)


# ===========================================================================
# SECTION 6: PAIRED UNCERTAINTY ANALYSIS (PLAYER-CLUSTER BOOTSTRAP)
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: bootstrap_improvement(preds, seed, replicates)
#
# Purpose: Quantify the sampling uncertainty in paired MAE improvement.
# Inputs:  preds -- same player-seasons evaluated under both models;
#          seed -- reproducible random seed; replicates -- bootstrap draws.
# Returns: (baseline MAE - model MAE, lower 95%, upper 95%).
# Method:  Resample player clusters rather than independent observations,
#          preserving each sampled player's collection of evaluated seasons.
# Limit:   This interval does not separately represent shared season shocks,
#          model-selection uncertainty, or the future-qualification selection.
# ---------------------------------------------------------------------------

def bootstrap_improvement(preds: pd.DataFrame, seed: int = 47,
                          replicates: int = 1000) -> tuple[float, float, float]:
    """Cluster-bootstrap reduction in MAE, resampling players not individual rows."""
    # Paired differences preserve the same player-season observations for
    # both models. Players are resampled to retain within-player dependence;
    # seasonal common shocks are not separately clustered in this demo.
    by_player = preds.assign(diff=(preds.baseline_abs_error - preds.model_abs_error)) \
        .groupby("playerID").agg(total_diff=("diff", "sum"), n=("diff", "size"))
    contributions = by_player.total_diff.to_numpy(dtype=float)
    n_rows = by_player.n.to_numpy(dtype=float)
    n_players = len(by_player)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for b in range(replicates):
        selection = rng.integers(0, n_players, size=n_players)
        estimates[b] = contributions[selection].sum() / n_rows[selection].sum()
    point = float((preds.baseline_abs_error - preds.model_abs_error).mean())
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi)


# ===========================================================================
# SECTION 7: RESEARCH OUTPUTS, DIAGNOSTICS, AND VISUALIZATIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: save_report(preds, metrics, output)
#
# Purpose: Create reproducible tables, a narrative summary, and a figure.
# Inputs:  preds -- row-level evaluation; metrics -- annual results;
#          output -- destination folder under the project's Results tree.
# Writes:  forward_predictions.csv, metrics_by_year.csv,
#          research_summary.txt, and forward_mae.png.
# Method:  Compute the cluster interval, export raw evaluation details,
#          and graph season-level absolute errors for both approaches.
# Note:    data_provenance.json is saved separately by main().
# ---------------------------------------------------------------------------

def save_report(preds: pd.DataFrame, metrics: pd.DataFrame, output: Path):
    """Export transparent row-level predictions, annual metrics and a figure."""
    output.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output / "forward_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_year.csv", index=False)
    gain, low, high = bootstrap_improvement(preds)
    summary = (
        "Research question: Does a regularized model reduce next-season OBP prediction "
        "error versus a shrinkage baseline?\n"
        "Population: Batters with >=200 APPROXIMATED PA in BOTH seasons.\n"
        "No outcome is imputed for players who do not return or qualify.\n"
        "Evaluation: expanding chronological training; evaluation on recent seasons.\n"
        "Training data for forecast origin t only include known outcomes through year t.\n"
        "Caution: These results are conditional on playing time; not an unconditional "
        "player-value forecast.\n"
        f"Evaluation player-seasons: {len(preds):,}\n"
        f"Distinct players: {preds.playerID.nunique():,}\n"
        f"Baseline MAE: {preds.baseline_abs_error.mean():.5f}\n"
        f"Ridge MAE: {preds.model_abs_error.mean():.5f}\n"
        f"Baseline minus ridge MAE: {gain:+.5f}\n"
        f"95% player-cluster bootstrap interval: [{low:+.5f}, {high:+.5f}]\n"
        "This is a simple demonstration, not a production projection system.\n"
    )
    (output / "research_summary.txt").write_text(summary, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(metrics))
    ax.plot(x, metrics.baseline_MAE, "o-", label="Shrinkage baseline")
    ax.plot(x, metrics.model_MAE, "s-", label="Regularized model")
    ax.set_xticks(x, [str(v) for v in metrics.target_year])
    ax.set_xlabel("Target season")
    ax.set_ylabel("Mean absolute OBP error")
    ax.set_title("Forward-season prediction error (returning, qualifying hitters)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "forward_mae.png", dpi=160)
    plt.close(fig)
    print(summary)


# ===========================================================================
# SECTION 8: SYNTHETIC REGRESSION TEST FIXTURE (NEVER MLB EVIDENCE)
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: synthetic_fixture(n_players, start, end)
#
# Purpose: Build a repeatable artificial dataset for software regression tests.
# Inputs:  n_players -- number of fictional players; start/end -- test years.
# Returns: Lahman-shaped batting records with synthetic counts and histories.
# Warning: Artificial records are NEVER evidence about MLB model performance.
# ---------------------------------------------------------------------------

def synthetic_fixture(n_players: int = 300, start: int = 2007,
                      end: int = 2024) -> pd.DataFrame:
    """TEST FIXTURE ONLY. Synthetic observations are never used as MLB evidence."""
    rng = np.random.default_rng(20261120)
    rows = []
    for i in range(n_players):
        skill = rng.normal(0.320, 0.035)
        for year in range(start, end + 1):
            if rng.uniform() < 0.12:
                continue
            pa = int(rng.integers(180, 651))
            skill = float(np.clip(0.85 * skill + 0.15 * 0.320 + rng.normal(0, 0.015), 0.20, 0.45))
            bb = int(rng.binomial(pa, 0.075))
            hbp = int(rng.binomial(pa, 0.01))
            sf = int(rng.binomial(pa, 0.006))
            ab = max(0, pa - bb - hbp - sf)
            hit_prob = np.clip((skill * pa - bb - hbp) / max(1, ab), 0.15, 0.40)
            hits = int(rng.binomial(ab, hit_prob))
            so = int(rng.binomial(pa, 0.20))
            rows.append(dict(playerID=f"test{i:04d}", yearID=year, AB=ab,
                             H=hits, BB=bb, HBP=hbp, SF=sf, SO=so))
    return pd.DataFrame(rows)


# ===========================================================================
# SECTION 9: AUTOMATED INTEGRITY CHECKS USING ARTIFICIAL RECORDS
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: self_test()
#
# Purpose: Confirm that the main pipeline executes and basic safeguards hold.
# Inputs:  None; all data are generated inside a temporary directory.
# Returns: None; prints explicit PASS/FAIL diagnostic messages.
# Checks:  Missing identifiers, within-year stint aggregation, label-year
#          alignment, valid probability ranges, finite predictions, and
#          successful calculation/export of comparison metrics.
# Limit:   This is a software self-test, NOT independent empirical validation.
# ---------------------------------------------------------------------------

def self_test():
    """Exercise the full pipeline using artificial records only."""
    with tempfile.TemporaryDirectory() as temp_dir:
        p = Path(temp_dir) / "TEST_FIXTURE_ONLY.csv"
        fixture = synthetic_fixture()
        # Regression test for the missing-identifier defect: two rows with
        # no ID must NOT be stitched together into a synthetic "nan" player.
        missing_id_rows = pd.DataFrame([
            dict(playerID=None, yearID=2021, AB=400, H=120,
                 BB=30, HBP=4, SF=3, SO=90),
            dict(playerID=None, yearID=2022, AB=410, H=127,
                 BB=32, HBP=3, SF=2, SO=93),
        ])
        fixture = pd.concat([fixture, missing_id_rows], ignore_index=True)
        # Same-year stints must combine before rates are computed. Artificial
        # duplicate stints below replace one record with two consistent parts.
        original = fixture.iloc[0].copy()
        split = original.copy()
        for c in ["AB", "H", "BB", "HBP", "SF", "SO"]:
            first = int(original[c] // 2)
            split[c] = int(original[c] - first)
            fixture.loc[fixture.index[0], c] = first
        fixture = pd.concat([fixture, pd.DataFrame([split])], ignore_index=True)
        fixture.to_csv(p, index=False)
        rows = load_and_prepare(p)
        assert not rows.playerID.astype(str).str.lower().isin(
            ["nan", "none", "null", ""]
        ).any(), "Rows with missing player identifiers survived cleaning"
        # Sanity-check that the baseline uses only information available at time t.
        assert rows["next_year"].eq(rows["yearID"] + 1).all()
        assert rows["baseline_obp"].between(0, 1).all()
        preds, metrics = score(rows)
        assert len(metrics) >= 3
        assert np.isfinite(preds[["model_pred", "next_obp"]].to_numpy()).all()
        save_report(preds, metrics, Path(temp_dir) / "TEST_OUTPUT")
    print("PASS: synthetic fixture and chronological workflow checks completed.")
    print("IMPORTANT: No synthetic numerical result should be reported as a baseball finding.")


# ===========================================================================
# SECTION 10: COMMAND-LINE INTERFACE, REAL-DATA RUN, AND PROVENANCE EXPORT
# ===========================================================================

# ---------------------------------------------------------------------------
# FUNCTION: main()
#
# Purpose: Provide a single noninteractive entry point for test/real runs.
# Inputs:  Command-line switches for CSV, cache/output paths, time filters,
#          qualifying PA, target evaluation seasons, and self-test mode.
# Returns: None; writes reports and prints diagnostics or raises an error.
# Method:  Parse options, optionally acquire public data, validate raw file
#          coverage/provenance, prepare predictors, evaluate, and report.
# Guard:   --self-test is the ONLY path that manufactures synthetic rows.
# ---------------------------------------------------------------------------

def main():
    """Parse command-line options and execute a synthetic or public-data run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Lahman-style Batting.csv file")
    parser.add_argument("--output", type=Path, default=Path("Results"))
    parser.add_argument("--data-dir", type=Path, default=Path("Data"),
                        help="Cache downloaded public Batting.csv here")
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--min-pa", type=int, default=200)
    parser.add_argument("--test-seasons", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="Automatically fetch actual public Lahman Batting.csv")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    # Use an actual public-data CSV for research mode. Synthetic observations
    # are generated ONLY by --self-test and NEVER enter real-data evaluation.
    source = "User-supplied CSV (origin not independently verified)"
    if args.input is None and args.download:
        args.input, source = acquire_public_data(args.data_dir)
    if args.input is None:
        parser.error("Specify --input Batting.csv, --download, or --self-test.")
    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")

    # Record data coverage AND the exact file hash before model fitting.
    provenance = dataset_provenance(args.input, source)
    # In --download mode the intended input is the FULL public Lahman release,
    # not a synthetic fixture, a truncated download, or an arbitrary season
    # subset already cached under the expected filename. Validate the broad
    # historical coverage before any analysis or conclusions are reported.
    if args.download and (
        provenance["raw_rows"] < 50_000
        or provenance["raw_min_year"] > 1900
        or provenance["raw_max_year"] < 2025
    ):
        raise ValueError(
            "Data/Batting.csv does not resemble the full public Lahman "
            "2025 batting release (historical coverage/row count mismatch). "
            "Inspect, rename, or remove the cached file before rerunning; "
            "do not report these results as public-MLB findings."
        )
    print("Public data audit:")
    print(f"  Raw batting records: {provenance['raw_rows']:,}")
    print(f"  Raw season coverage: {provenance['raw_min_year']}–{provenance['raw_max_year']}")
    print(f"  SHA-256: {provenance['sha256']}")
    rows = load_and_prepare(args.input, min_year=args.min_year, min_pa=args.min_pa)
    preds, metrics = score(rows, test_seasons=args.test_seasons)
    save_report(preds, metrics, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Real-data reports written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
