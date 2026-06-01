"""
GridLock — Traffic Demand Prediction Pipeline
=============================================
End-to-end ML pipeline:
  1. EDA summary
  2. Feature engineering (geohash decode, time/cyclical, categorical, target encoding,
     geohash-level aggregates, interactions)
  3. K-Fold cross-validated LightGBM regression
  4. Test prediction + submission file

Evaluation: score = max(0, 100 * r2_score(actual, predicted))
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygeohash as pgh
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
PROJECT_ROOT = Path("/home/runtime-terror/Desktop/Github/GridLock")
TRAIN_PATH = PROJECT_ROOT / "train.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"
SAMPLE_SUB_PATH = PROJECT_ROOT / "sample_submission.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

N_FOLDS = 5
SEED = 42

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def hbar(title: str = "") -> None:
    line = "=" * 78
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ----------------------------------------------------------------------------- #
# 1. Load & EDA
# ----------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def print_eda(train: pd.DataFrame, test: pd.DataFrame) -> None:
    hbar("EDA — Exploratory Data Analysis")
    print(f"Train shape: {train.shape}")
    print(f"Test  shape: {test.shape}")

    print("\n-- Dtypes --")
    print(train.dtypes)

    print("\n-- Missing values (train) --")
    print(train.isna().sum())
    print("\n-- Missing values (test) --")
    print(test.isna().sum())

    print("\n-- Target (demand) distribution --")
    print(train["demand"].describe())
    q = train["demand"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    print(f"Quantiles: {q}")

    print("\n-- Categorical value counts (train) --")
    for col in ["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather"]:
        print(f"\n{col}:")
        print(train[col].value_counts(dropna=False).to_string())

    print("\n-- Day / timestamp coverage --")
    print(f"Train days   : {sorted(train['day'].unique())}")
    print(f"Test  days   : {sorted(test['day'].unique())}")
    print(f"# distinct timestamps in train: {train['timestamp'].nunique()}")
    print(f"# distinct timestamps in test : {test['timestamp'].nunique()}")

    print("\n-- Geohash coverage --")
    train_gh = set(train["geohash"].unique())
    test_gh = set(test["geohash"].unique())
    print(f"Unique geohashes (train): {len(train_gh)}")
    print(f"Unique geohashes (test) : {len(test_gh)}")
    print(f"Overlap                 : {len(train_gh & test_gh)}")
    print(f"Geohash lengths         : {sorted(train['geohash'].str.len().unique())}")

    # Numeric correlations with target (using numeric encodings of cats)
    print("\n-- Numeric correlations with demand --")
    num_corr = (
        train[["NumberofLanes", "Temperature", "demand"]].corr()["demand"].drop("demand")
    )
    print(num_corr.to_string())

    # Group means against target
    print("\n-- Mean demand by categorical --")
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather", "NumberofLanes"]:
        gm = train.groupby(col, dropna=False)["demand"].agg(["mean", "count"])
        print(f"\n{col}:")
        print(gm.to_string())


# ----------------------------------------------------------------------------- #
# 2. Feature engineering
# ----------------------------------------------------------------------------- #
def decode_geohashes(df: pd.DataFrame, cache: dict | None = None) -> pd.DataFrame:
    """Decode geohash to lat/lon. Cache decode() to avoid re-decoding duplicates."""
    cache = cache if cache is not None else {}

    def _dec(gh: str) -> tuple[float, float]:
        if gh in cache:
            return cache[gh]
        try:
            ll = pgh.decode(gh)
            # pygeohash 3.x returns a LatLong namedtuple; older versions a tuple.
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
    """Parse 'H:M' timestamp -> minute-of-day + cyclical features.

    Also build a global minute index (day * 1440 + minute_of_day) and treat 'day'
    as day-of-week proxy.
    """
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Cyclical: time of day
    radians = 2 * np.pi * df["minute_of_day"] / (24 * 60)
    df["tod_sin"] = np.sin(radians)
    df["tod_cos"] = np.cos(radians)

    # Cyclical: hour
    h_rad = 2 * np.pi * df["hour"] / 24
    df["hour_sin"] = np.sin(h_rad)
    df["hour_cos"] = np.cos(h_rad)

    # Day-of-week proxy from `day` (mod 7 gives weekday)
    df["dow"] = df["day"] % 7
    dow_rad = 2 * np.pi * df["dow"] / 7
    df["dow_sin"] = np.sin(dow_rad)
    df["dow_cos"] = np.cos(dow_rad)

    # Coarse buckets: morning / mid / evening / night
    df["tod_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    df["is_rush"] = (
        df["hour"].between(7, 10) | df["hour"].between(17, 20)
    ).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Global minute timeline (helps with two-day train, single-day test)
    df["global_minute"] = df["day"] * 1440 + df["minute_of_day"]

    return df


def encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Label-encode (factorize) categorical columns consistently across train/test.

    LightGBM handles label-encoded categoricals natively when listed as
    `categorical_feature`. We treat NaN as its own category by filling with a
    sentinel string before factorizing.
    """
    cat_cols = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    cat_feature_names: list[str] = []

    for col in cat_cols:
        combined = pd.concat([train[col], test[col]], axis=0).fillna("__NA__")
        codes, _ = pd.factorize(combined.astype(str))
        train[col + "_enc"] = codes[: len(train)]
        test[col + "_enc"] = codes[len(train):]
        cat_feature_names.append(col + "_enc")

    # geohash itself is high-cardinality but we encode it too (LGB handles it).
    combined_gh = pd.concat([train["geohash"], test["geohash"]], axis=0)
    gh_codes, _ = pd.factorize(combined_gh.astype(str))
    train["geohash_enc"] = gh_codes[: len(train)]
    test["geohash_enc"] = gh_codes[len(train):]
    cat_feature_names.append("geohash_enc")

    # Geohash prefixes (coarser cells, less sparse)
    for prefix_len in (3, 4, 5):
        col = f"geohash_p{prefix_len}"
        combined_p = pd.concat(
            [train["geohash"].str[:prefix_len], test["geohash"].str[:prefix_len]], axis=0
        )
        p_codes, _ = pd.factorize(combined_p.astype(str))
        train[col + "_enc"] = p_codes[: len(train)]
        test[col + "_enc"] = p_codes[len(train):]
        cat_feature_names.append(col + "_enc")

    return train, test, cat_feature_names


def add_geohash_aggregates(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-geohash static features (use train target to build aggregates with
    out-of-fold logic later; here we only add static structural features that
    do not leak: feature mode/mean of NumberofLanes is fine but mean(demand) is not
    used directly here — we'll do target encoding inside CV)."""
    # Per-geohash structural means (no target leakage)
    g_stats = (
        train.groupby("geohash")
        .agg(
            gh_lanes_mean=("NumberofLanes", "mean"),
            gh_temp_mean=("Temperature", "mean"),
        )
        .reset_index()
    )
    # Append test-only geohashes by also computing stats from test for non-target cols
    g_stats_test = (
        test.groupby("geohash")
        .agg(
            gh_lanes_mean_t=("NumberofLanes", "mean"),
            gh_temp_mean_t=("Temperature", "mean"),
        )
        .reset_index()
    )
    merged = pd.merge(g_stats, g_stats_test, on="geohash", how="outer")
    merged["gh_lanes_mean"] = merged["gh_lanes_mean"].fillna(merged["gh_lanes_mean_t"])
    merged["gh_temp_mean"] = merged["gh_temp_mean"].fillna(merged["gh_temp_mean_t"])
    merged = merged[["geohash", "gh_lanes_mean", "gh_temp_mean"]]

    train = train.merge(merged, on="geohash", how="left")
    test = test.merge(merged, on="geohash", how="left")
    return train, test


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap interaction features."""
    df["lanes_x_rush"] = df["NumberofLanes"] * df["is_rush"]
    df["lanes_x_landmarks"] = df["NumberofLanes"] * (df["Landmarks_enc"] >= 0).astype(int)
    df["temp_x_rush"] = df["Temperature"].fillna(df["Temperature"].median()) * df["is_rush"]
    df["lat_x_lon"] = df["lat"] * df["lon"]
    return df


# ----------------------------------------------------------------------------- #
# 3. CV target encoding for geohash (out-of-fold to prevent leakage)
# ----------------------------------------------------------------------------- #
def oof_target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    keys: list[str],
    n_splits: int = 5,
    smoothing: float = 20.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Out-of-fold target encoding for one or more group keys."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = train[target_col].mean()
    new_cols: list[str] = []

    for key in keys:
        feat_name = f"te_{key}"
        train[feat_name] = np.nan
        for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
            tr_part = train.iloc[tr_idx]
            stats = tr_part.groupby(key)[target_col].agg(["mean", "count"])
            smoothed = (
                stats["mean"] * stats["count"] + global_mean * smoothing
            ) / (stats["count"] + smoothing)
            train.loc[train.index[val_idx], feat_name] = (
                train.iloc[val_idx][key].map(smoothed).values
            )
        train[feat_name] = train[feat_name].fillna(global_mean)

        # For test: use full-train stats
        stats_full = train.groupby(key)[target_col].agg(["mean", "count"])
        smoothed_full = (
            stats_full["mean"] * stats_full["count"] + global_mean * smoothing
        ) / (stats_full["count"] + smoothing)
        test[feat_name] = test[key].map(smoothed_full).fillna(global_mean)
        new_cols.append(feat_name)

    return train, test, new_cols


# ----------------------------------------------------------------------------- #
# 4. Train LightGBM with K-Fold CV
# ----------------------------------------------------------------------------- #
def train_lgbm_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    cat_features: list[str],
    n_splits: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, list[pd.DataFrame]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fi_frames: list[pd.DataFrame] = []

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=20,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=0.1,
        verbose=-1,
        n_jobs=-1,
        seed=seed,
    )

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        t0 = time.time()
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        dvalid = lgb.Dataset(
            X_val, label=y_val, categorical_feature=cat_features, reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = val_pred
        test_pred += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

        fi = pd.DataFrame(
            {
                "feature": X.columns,
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
                "fold": fold,
            }
        )
        fi_frames.append(fi)

        fold_r2 = r2_score(y_val, val_pred)
        print(
            f"  Fold {fold}: best_iter={model.best_iteration:>4d}  "
            f"R²={fold_r2:.5f}  time={time.time() - t0:.1f}s"
        )

    return oof, test_pred, fi_frames


# ----------------------------------------------------------------------------- #
# 5. Main pipeline
# ----------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.time()
    hbar("Loading data")
    train, test = load_data()
    print(f"Loaded train {train.shape}, test {test.shape}")

    print_eda(train, test)

    hbar("Feature engineering")
    # Save originals before mutation
    test_index = test["Index"].copy()

    # Geohash decode (lat/lon)
    cache: dict = {}
    train = decode_geohashes(train, cache)
    test = decode_geohashes(test, cache)
    print("  geohash decoded to lat/lon (cache size:", len(cache), ")")

    # Time features
    train = add_time_features(train)
    test = add_time_features(test)
    print("  time features added")

    # Categorical encodings (geohash + cats)
    train, test, cat_features = encode_categoricals(train, test)
    print(f"  categorical features encoded: {cat_features}")

    # Static per-geohash aggregates (non-target)
    train, test = add_geohash_aggregates(train, test)
    print("  geohash structural aggregates added")

    # Interactions
    train = add_interactions(train)
    test = add_interactions(test)
    print("  interaction features added")

    # Out-of-fold target encoding for geohash + geohash prefixes
    te_keys = ["geohash", "geohash"]  # placeholder
    te_keys = ["geohash"]
    # also encode geohash prefixes by adding string columns
    for p in (3, 4, 5):
        train[f"geohash_p{p}"] = train["geohash"].str[:p]
        test[f"geohash_p{p}"] = test["geohash"].str[:p]
        te_keys.append(f"geohash_p{p}")

    train, test, te_cols = oof_target_encode(
        train, test, target_col="demand", keys=te_keys, n_splits=N_FOLDS, smoothing=20.0
    )
    print(f"  out-of-fold target-encoded: {te_cols}")

    # Final feature list
    feature_cols = [
        # raw / structural
        "NumberofLanes",
        "Temperature",
        "lat",
        "lon",
        "lat_x_lon",
        # time
        "day",
        "hour",
        "minute",
        "minute_of_day",
        "global_minute",
        "tod_sin",
        "tod_cos",
        "hour_sin",
        "hour_cos",
        "dow",
        "dow_sin",
        "dow_cos",
        "tod_bucket",
        "is_rush",
        "is_night",
        # categorical encodings
        *cat_features,
        # geohash aggregates
        "gh_lanes_mean",
        "gh_temp_mean",
        # interactions
        "lanes_x_rush",
        "lanes_x_landmarks",
        "temp_x_rush",
        # target encodings
        *te_cols,
    ]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y = train["demand"].values.astype(np.float64)

    # Fill NaNs in numeric features with median (LGB also handles NaN, but be explicit
    # for the few engineered numeric cols)
    for col in ["Temperature", "gh_temp_mean", "temp_x_rush"]:
        med = X[col].median()
        X[col] = X[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    print(f"\nFinal feature matrix: train {X.shape}, test {X_test.shape}")
    print(f"Categorical features (for LGB): {cat_features}")

    hbar(f"Training LightGBM ({N_FOLDS}-fold CV)")
    oof, test_pred, fi_frames = train_lgbm_cv(
        X, y, X_test, cat_features=cat_features, n_splits=N_FOLDS, seed=SEED
    )

    # Predictions can occasionally go slightly outside [0, 1]; clip.
    test_pred = np.clip(test_pred, 0.0, 1.0)

    cv_r2 = r2_score(y, oof)
    cv_score = max(0.0, 100.0 * cv_r2)
    hbar("CV Results")
    print(f"Overall OOF R²      : {cv_r2:.6f}")
    print(f"Overall score (100*R²): {cv_score:.4f}")

    hbar("Top feature importances (mean gain across folds)")
    fi_all = pd.concat(fi_frames, ignore_index=True)
    fi_mean = (
        fi_all.groupby("feature")[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    print(fi_mean.head(25).to_string())

    hbar("Writing submission")
    sub = pd.DataFrame({"Index": test_index.values, "demand": test_pred})
    # Match sample_submission.csv column order/names exactly
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    assert list(sub.columns) == list(sample.columns), (
        f"Column mismatch: got {list(sub.columns)} expected {list(sample.columns)}"
    )
    assert sub.shape == (41778, 2), f"Bad shape: {sub.shape}"
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH}  shape={sub.shape}")
    print(sub.head())

    hbar("Summary")
    print(f"CV OOF R²            : {cv_r2:.6f}")
    print(f"Final score (max 0, 100*R²): {cv_score:.4f}")
    print(f"Submission file       : {SUBMISSION_PATH}")
    print(f"Total runtime         : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
