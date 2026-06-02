"""
GridLock — Traffic Demand Prediction Pipeline (v8: robust ensemble on the
proven full-train baseline)
============================================================================
HISTORY / WHY THIS DESIGN
-------------------------
The full-train LightGBM baseline (preserved as `solution_v1_baseline.py`) scored
**91.22** on the real leaderboard. A later "d49-only + fancy features" rewrite
scored higher on local OOF but **dropped to 88.89 / 88.28 on the real LB** — it
overfit a leaky, night-only signal. Lesson, confirmed by a leakage-aware local
harness (`research_cv.py`): on this dataset, **simple + leak-free generalizes;
high-cardinality per-geohash time encodings (gh×mod, gh×hour) silently leak the
target and collapse on the held-out daytime block** (block-holdout R² 0.88 → 0.43).

So this version does NOT change the winning recipe's feature set. It keeps the
baseline's exact, leak-free features and adds only what is mathematically safe and
almost always helps R²: **variance reduction via seed-bagging + model diversity**
(3-seed LightGBM + HistGradientBoosting + ExtraTrees, simple average). No new
leak-prone features, no training-population change.

Evaluation: score = max(0, 100 * r2_score(actual, predicted))
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/runtime-terror/Desktop/Github/GridLock")
TRAIN_PATH = PROJECT_ROOT / "train.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"
SAMPLE_SUB_PATH = PROJECT_ROOT / "sample_submission.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

N_FOLDS = 5
SEED = 42
LGB_SEEDS = (42, 7, 2024)   # seed-bagging for variance reduction


def hbar(title: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}" if title else line)


# --------------------------------------------------------------------------- #
# Feature engineering — IDENTICAL to the proven baseline (leak-free)
# --------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def decode_geohashes(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    def _dec(gh: str):
        if gh in cache:
            return cache[gh]
        try:
            ll = pgh.decode(gh)
            lat = getattr(ll, "latitude", None)
            lon = getattr(ll, "longitude", None)
            if lat is None or lon is None:
                lat, lon = float(ll[0]), float(ll[1])
        except Exception:
            lat, lon = np.nan, np.nan
        cache[gh] = (lat, lon)
        return lat, lon

    out = df["geohash"].map(_dec)
    df["lat"] = out.map(lambda x: x[0])
    df["lon"] = out.map(lambda x: x[1])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    rad = 2 * np.pi * df["minute_of_day"] / (24 * 60)
    df["tod_sin"], df["tod_cos"] = np.sin(rad), np.cos(rad)
    h_rad = 2 * np.pi * df["hour"] / 24
    df["hour_sin"], df["hour_cos"] = np.sin(h_rad), np.cos(h_rad)
    df["dow"] = df["day"] % 7
    dow_rad = 2 * np.pi * df["dow"] / 7
    df["dow_sin"], df["dow_cos"] = np.sin(dow_rad), np.cos(dow_rad)
    df["tod_bucket"] = pd.cut(df["hour"], bins=[-1, 5, 11, 16, 20, 23],
                              labels=[0, 1, 2, 3, 4]).astype(int)
    df["is_rush"] = (df["hour"].between(7, 10) | df["hour"].between(17, 20)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["global_minute"] = df["day"] * 1440 + df["minute_of_day"]
    return df


def encode_categoricals(train, test):
    cat_cols = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    cat_names: list[str] = []
    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0).fillna("__NA__")
        codes, _ = pd.factorize(combined.astype(str))
        train[col + "_enc"] = codes[: len(train)]
        test[col + "_enc"] = codes[len(train):]
        cat_names.append(col + "_enc")
    combined_gh = pd.concat([train["geohash"], test["geohash"]], axis=0)
    gh_codes, _ = pd.factorize(combined_gh.astype(str))
    train["geohash_enc"] = gh_codes[: len(train)]
    test["geohash_enc"] = gh_codes[len(train):]
    cat_names.append("geohash_enc")
    for plen in (3, 4, 5):
        col = f"geohash_p{plen}"
        combined_p = pd.concat([train["geohash"].str[:plen], test["geohash"].str[:plen]], axis=0)
        p_codes, _ = pd.factorize(combined_p.astype(str))
        train[col + "_enc"] = p_codes[: len(train)]
        test[col + "_enc"] = p_codes[len(train):]
        cat_names.append(col + "_enc")
    return train, test, cat_names


def add_geohash_aggregates(train, test):
    g_stats = train.groupby("geohash").agg(
        gh_lanes_mean=("NumberofLanes", "mean"), gh_temp_mean=("Temperature", "mean")).reset_index()
    g_stats_test = test.groupby("geohash").agg(
        gh_lanes_mean_t=("NumberofLanes", "mean"), gh_temp_mean_t=("Temperature", "mean")).reset_index()
    merged = pd.merge(g_stats, g_stats_test, on="geohash", how="outer")
    merged["gh_lanes_mean"] = merged["gh_lanes_mean"].fillna(merged["gh_lanes_mean_t"])
    merged["gh_temp_mean"] = merged["gh_temp_mean"].fillna(merged["gh_temp_mean_t"])
    merged = merged[["geohash", "gh_lanes_mean", "gh_temp_mean"]]
    return train.merge(merged, on="geohash", how="left"), test.merge(merged, on="geohash", how="left")


def add_interactions(df):
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush"]
    df["lanes_x_landmarks"] = df["NumberofLanes"] * (df["Landmarks_enc"] >= 0).astype(int)
    df["temp_x_rush"] = df["Temperature"].fillna(df["Temperature"].median()) * df["is_rush"]
    df["lat_x_lon"] = df["lat"] * df["lon"]
    return df


def oof_target_encode(train, test, target_col, keys, n_splits=5, smoothing=20.0, seed=SEED):
    """Out-of-fold target encoding — ONLY for high-count keys (geohash + prefixes).
    Deliberately excludes geohash×time keys, which leak (≈1 row/cell)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = train[target_col].mean()
    new_cols: list[str] = []
    for key in keys:
        feat = f"te_{key}"
        train[feat] = np.nan
        for _, (tr_idx, val_idx) in enumerate(kf.split(train)):
            stats = train.iloc[tr_idx].groupby(key)[target_col].agg(["mean", "count"])
            sm = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            train.loc[train.index[val_idx], feat] = train.iloc[val_idx][key].map(sm).values
        train[feat] = train[feat].fillna(global_mean)
        stats_full = train.groupby(key)[target_col].agg(["mean", "count"])
        sm_full = (stats_full["mean"] * stats_full["count"] + global_mean * smoothing) / (stats_full["count"] + smoothing)
        test[feat] = test[key].map(sm_full).fillna(global_mean)
        new_cols.append(feat)
    return train, test, new_cols


# --------------------------------------------------------------------------- #
# Ensemble — KFold, seed-bagged LGB + HistGB + ExtraTrees (variance reduction)
# --------------------------------------------------------------------------- #
def train_ensemble(X, y, X_test, cat_features, n_splits=N_FOLDS, seed=SEED):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_members = (len(LGB_SEEDS) + 2)            # LGB seeds + HistGB + ExtraTrees
    Xf = X.fillna(X.median()); Xtf = X_test.fillna(X.median())

    lgb_base = dict(objective="regression", metric="rmse", num_leaves=127, max_depth=-1,
                    min_data_in_leaf=20, feature_fraction=0.85, bagging_fraction=0.85,
                    bagging_freq=1, lambda_l1=0.0, lambda_l2=0.1, verbose=-1, n_jobs=-1)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        t0 = time.time()
        val_members = np.zeros((len(val_idx), n_members))
        test_members = np.zeros((len(X_test), n_members))
        mi = 0
        # seed-bagged LightGBM
        for sd in LGB_SEEDS:
            p = dict(lgb_base, learning_rate=0.03, seed=sd,
                     feature_fraction=0.8 if sd != 42 else 0.85)
            dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_features)
            dva = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], categorical_feature=cat_features, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
            val_members[:, mi] = m.predict(X.iloc[val_idx], num_iteration=m.best_iteration)
            test_members[:, mi] = m.predict(X_test, num_iteration=m.best_iteration)
            mi += 1
        # HistGradientBoosting
        h = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=0.1, random_state=seed)
        h.fit(Xf.iloc[tr_idx], y[tr_idx])
        val_members[:, mi] = h.predict(Xf.iloc[val_idx]); test_members[:, mi] = h.predict(Xtf); mi += 1
        # ExtraTrees
        e = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        e.fit(Xf.iloc[tr_idx], y[tr_idx])
        val_members[:, mi] = e.predict(Xf.iloc[val_idx]); test_members[:, mi] = e.predict(Xtf); mi += 1

        oof[val_idx] = val_members.mean(axis=1)
        test_pred += test_members.mean(axis=1) / n_splits
        print(f"  Fold {fold}: R²={r2_score(y[val_idx], oof[val_idx]):.5f}  time={time.time()-t0:.1f}s")
    return oof, test_pred


# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.time()
    hbar("Loading data")
    train, test = load_data()
    test_index = test["Index"].copy()
    print(f"train {train.shape}  test {test.shape}")

    hbar("Feature engineering (baseline, leak-free)")
    cache: dict = {}
    train = decode_geohashes(train, cache); test = decode_geohashes(test, cache)
    train = add_time_features(train); test = add_time_features(test)
    train, test, cat_features = encode_categoricals(train, test)
    train, test = add_geohash_aggregates(train, test)
    train = add_interactions(train); test = add_interactions(test)

    te_keys = ["geohash"]
    for p in (3, 4, 5):
        train[f"geohash_p{p}"] = train["geohash"].str[:p]
        test[f"geohash_p{p}"] = test["geohash"].str[:p]
        te_keys.append(f"geohash_p{p}")
    train, test, te_cols = oof_target_encode(train, test, "demand", te_keys, N_FOLDS, smoothing=20.0)
    print(f"  target-encoded (high-count only): {te_cols}")

    feature_cols = [
        "NumberofLanes", "Temperature", "lat", "lon", "lat_x_lon",
        "day", "hour", "minute", "minute_of_day", "global_minute",
        "tod_sin", "tod_cos", "hour_sin", "hour_cos", "dow", "dow_sin", "dow_cos",
        "tod_bucket", "is_rush", "is_night",
        *cat_features,
        "gh_lanes_mean", "gh_temp_mean",
        "lanes_x_rush", "lanes_x_landmarks", "temp_x_rush",
        *te_cols,
    ]
    X = train[feature_cols].copy(); X_test = test[feature_cols].copy()
    y = train["demand"].values.astype(np.float64)
    for col in ["Temperature", "gh_temp_mean", "temp_x_rush"]:
        med = X[col].median(); X[col] = X[col].fillna(med); X_test[col] = X_test[col].fillna(med)
    print(f"  feature matrix: train {X.shape}, test {X_test.shape}  ({len(LGB_SEEDS)}-seed LGB + HistGB + ExtraTrees)")

    hbar(f"Training ensemble ({N_FOLDS}-fold CV)")
    oof, test_pred = train_ensemble(X, y, X_test, cat_features, N_FOLDS, SEED)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    cv_r2 = r2_score(y, oof)
    hbar("CV Results")
    print(f"Overall OOF R²        : {cv_r2:.6f}   (random-KFold — inflated; LB ≈ this minus ~0.04)")
    print(f"Overall score (100·R²): {max(0.0, 100.0*cv_r2):.4f}")
    print(f"  test pred mean={test_pred.mean():.4f} std={test_pred.std():.4f} "
          f"min={test_pred.min():.4f} max={test_pred.max():.4f}")

    hbar("Writing submission")
    sub = pd.DataFrame({"Index": test_index.values, "demand": test_pred})
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    assert list(sub.columns) == list(sample.columns), f"col mismatch {list(sub.columns)}"
    assert sub.shape == (41778, 2), f"bad shape {sub.shape}"
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {SUBMISSION_PATH}  shape={sub.shape}")
    print(sub.head())
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
