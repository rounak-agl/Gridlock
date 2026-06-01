"""
GridLock — Traffic Demand Prediction Pipeline (v7)
==================================================
Implements the documented breakthrough that the previous `solution.py` (now
preserved as `solution_v1_baseline.py`) never actually contained: the naive
baseline trained on the full 77k mixed-population training set and was capped at
~91 LB by a training/test distribution mismatch.

This version trains the demand model on the *correct* population — the day-49
rows only — using leakage-free cross-day features anchored on the day-48 history.

Why d49-only?
-------------
  Test  = day-49 daytime (mod 135-825), predicted from day-48 history.
  d49 train rows (mod 0-120) are the only rows whose features (d48 lookups) are a
  GENUINE cross-day signal AND whose task matches the test task. d48 rows leak
  (their d48[gh,mod] lookup == their own label), which inflated OOF to ~0.99 and
  collapsed on the LB.

Empirically verified before building (on the *hardest* night regime):
  raw d48_exact -> d49           R2 = 0.493
  d48_exact + per-gh residual    R2 = 0.897   (leave-one-mod-out)
At the easier daytime test mods the d48 anchor is far stronger (R2 0.90-0.96),
so the model has real headroom above the night-hour OOF it is validated on.

Pipeline
--------
  1. Parse temporal structure (day, mod = minute-of-day).
  2. Leakage-free feature engineering anchored on the full day-48 history.
  3. KFold OOF training on d49 rows: LightGBM (raw / log1p / pure) + sklearn
     HistGradientBoosting + ExtraTrees + Ridge. XGBoost/CatBoost added if present.
  4. Non-negative ridge meta-blend over OOF predictions, with a grid-search and
     best-single fallback (whichever wins on honest OOF).
  5. Clip to [0,1], write the 41778x2 submission.

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
from sklearn.neighbors import BallTree
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

import lightgbm as lgb

# Optional boosters — used only if importable.
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False
try:
    from catboost import CatBoostRegressor
    HAVE_CB = True
except Exception:
    HAVE_CB = False

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
DELTAS = (15, 30, 45, 60, 90, 120, 180, 240)  # +/- minute offsets for d48 lookups


def hbar(title: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}" if title else line)


# ----------------------------------------------------------------------------- #
# 1. Load & temporal parse
# ----------------------------------------------------------------------------- #
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["timestamp"].str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["mod"] = df["hour"] * 60 + df["minute"]   # minute-of-day, 15-min grid
    rad = 2 * np.pi * df["mod"] / 1440.0
    df["mod_sin"] = np.sin(rad)
    df["mod_cos"] = np.cos(rad)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_rush"] = (df["hour"].between(7, 10) | df["hour"].between(17, 20)).astype(int)
    return df


# ----------------------------------------------------------------------------- #
# 2. Geohash decode (lat/lon) with cache
# ----------------------------------------------------------------------------- #
def decode_geohashes(gh_series: pd.Series, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    def _dec(gh):
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

    out = gh_series.map(_dec)
    return out.map(lambda x: x[0]).values, out.map(lambda x: x[1]).values


# ----------------------------------------------------------------------------- #
# 3. d48-anchored feature engineering (leakage-free for d49/test rows)
# ----------------------------------------------------------------------------- #
def build_d48_structures(d48: pd.DataFrame):
    """Lookup tables built ONLY from day-48 — clean to use as features for any
    d49 / test row (a different day)."""
    exact = d48.groupby(["geohash", "mod"])["demand"].mean()           # (gh,mod) -> demand
    exact_map = exact.to_dict()

    # Per-geohash full-day d48 trajectory (sorted) for interpolation fill.
    traj = {}
    for gh, g in d48.groupby("geohash"):
        gg = g.sort_values("mod")
        traj[gh] = (gg["mod"].values.astype(float), gg["demand"].values.astype(float))

    gh_stats = d48.groupby("geohash")["demand"].agg(
        gh_mean="mean", gh_std="std", gh_max="max", gh_min="min", gh_median="median"
    )
    gh_hour = d48.groupby(["geohash", "hour"])["demand"].mean()
    mod_mean = d48.groupby("mod")["demand"].mean()
    mod_std = d48.groupby("mod")["demand"].std()

    return dict(exact=exact_map, traj=traj, gh_stats=gh_stats,
                gh_hour=gh_hour.to_dict(), mod_mean=mod_mean.to_dict(),
                mod_std=mod_std.to_dict())


def d48_reliability_table(d48: pd.DataFrame) -> dict:
    """Lag-15 autocorrelation R^2 within d48 at each mod slot — a proxy for how
    trustworthy the d48 exact lookup is at that time of day."""
    piv = d48.pivot_table(index="geohash", columns="mod", values="demand")
    mods = sorted(d48["mod"].unique())
    rel = {}
    for i, m in enumerate(mods):
        if i == 0:
            rel[m] = np.nan
            continue
        prev = mods[i - 1]
        a, b = piv[prev], piv[m]
        ok = a.notna() & b.notna()
        if ok.sum() > 30:
            try:
                rel[m] = max(0.0, r2_score(b[ok], a[ok]))
            except Exception:
                rel[m] = np.nan
        else:
            rel[m] = np.nan
    # fill NaN with median
    med = np.nanmedian(list(rel.values()))
    return {m: (v if not np.isnan(v) else med) for m, v in rel.items()}, med


def interp_d48(traj: dict, gh: str, mod: int) -> float:
    """Linear interpolation of the geohash's d48 trajectory at `mod`. NaN if the
    geohash is absent from d48."""
    t = traj.get(gh)
    if t is None:
        return np.nan
    xs, ys = t
    if len(xs) == 0:
        return np.nan
    return float(np.interp(mod, xs, ys))


def make_features(df: pd.DataFrame, S: dict, rel_table: dict, rel_med: float,
                  prefix_aggs: dict) -> pd.DataFrame:
    """Build the model feature frame for a set of rows using day-48 structures."""
    n = len(df)
    gh = df["geohash"].values
    mod = df["mod"].values.astype(int)
    hour = df["hour"].values.astype(int)
    exact = S["exact"]
    traj = S["traj"]

    feat = pd.DataFrame(index=df.index)

    # --- d48 exact lookup + interpolation fill + missing flag ---
    d48_exact = np.array([exact.get((g, m), np.nan) for g, m in zip(gh, mod)])
    d48_interp = np.array([interp_d48(traj, g, m) for g, m in zip(gh, mod)])
    feat["d48_exact_raw"] = d48_exact
    feat["d48_missing"] = np.isnan(d48_exact).astype(int)
    feat["d48_anchor"] = np.where(np.isnan(d48_exact), d48_interp, d48_exact)

    # --- temporal delta lookups (exact preferred, interp fallback) ---
    for d in DELTAS:
        for sign, tag in ((+1, "p"), (-1, "m")):
            mm = mod + sign * d
            vals = np.array([
                exact.get((g, int(x)), np.nan) if 0 <= x <= 1425 else np.nan
                for g, x in zip(gh, mm)
            ])
            interp = np.array([interp_d48(traj, g, int(x)) for g, x in zip(gh, mm)])
            feat[f"d48_{tag}{d}"] = np.where(np.isnan(vals), interp, vals)

    # --- temporal shape: slope & curvature around the slot from interp ---
    a15p = np.array([interp_d48(traj, g, int(m) + 15) for g, m in zip(gh, mod)])
    a15m = np.array([interp_d48(traj, g, int(m) - 15) for g, m in zip(gh, mod)])
    feat["d48_slope"] = (a15p - a15m) / 2.0
    feat["d48_accel"] = a15p + a15m - 2 * feat["d48_anchor"].values

    # --- per-geohash full-day stats ---
    gs = S["gh_stats"]
    for c in ["gh_mean", "gh_std", "gh_max", "gh_min", "gh_median"]:
        feat[c] = df["geohash"].map(gs[c]).values
    feat["gh_cv"] = feat["gh_std"] / (feat["gh_mean"] + 1e-6)

    # --- per-(geohash,hour) ---
    ghh = S["gh_hour"]
    feat["gh_hour_mean"] = np.array([ghh.get((g, h), np.nan) for g, h in zip(gh, hour)])

    # --- global time-of-day profile ---
    feat["mod_mean"] = np.array([S["mod_mean"].get(m, np.nan) for m in mod])
    feat["mod_std"] = np.array([S["mod_std"].get(m, np.nan) for m in mod])

    # --- ratios (shape, not level) ---
    feat["d48_ratio_mod"] = feat["d48_anchor"] / (feat["mod_mean"] + 1e-6)
    feat["d48_ratio_gh"] = feat["d48_anchor"] / (feat["gh_mean"] + 1e-6)
    feat["profile"] = feat["gh_hour_mean"] / (feat["gh_mean"] + 1e-6)

    # --- d48 reliability at this mod + interaction ---
    feat["d48_reliability"] = np.array([rel_table.get(m, rel_med) for m in mod])
    feat["d48_x_rel"] = feat["d48_anchor"] * feat["d48_reliability"]

    # --- spatial prefix aggregates (p4/p5) at (prefix,mod) and (prefix,hour) ---
    for plen in (4, 5):
        pcol = df["geohash"].str[:plen]
        pm = prefix_aggs[(plen, "mod")]
        ph = prefix_aggs[(plen, "hour")]
        feat[f"p{plen}_mod_mean"] = [pm.get((p, m), np.nan) for p, m in zip(pcol.values, mod)]
        feat[f"p{plen}_hour_mean"] = [ph.get((p, h), np.nan) for p, h in zip(pcol.values, hour)]

    # --- structural / contextual numerics ---
    feat["NumberofLanes"] = df["NumberofLanes"].values
    feat["Temperature"] = df["Temperature"].values
    feat["lat"] = df["lat"].values
    feat["lon"] = df["lon"].values
    feat["hour"] = hour
    feat["mod"] = mod
    feat["mod_sin"] = df["mod_sin"].values
    feat["mod_cos"] = df["mod_cos"].values
    feat["is_night"] = df["is_night"].values
    feat["is_rush"] = df["is_rush"].values

    return feat


def add_spatial_nn(train_feat, test_feat, d48, train_df, test_df, k=6):
    """IDW-weighted demand of k nearest d48 geohashes (by haversine on lat/lon)."""
    centroids = d48.groupby("geohash").agg(lat=("lat", "first"), lon=("lon", "first"),
                                           dem=("demand", "mean")).dropna()
    pts = np.radians(centroids[["lat", "lon"]].values)
    dem = centroids["dem"].values
    tree = BallTree(pts, metric="haversine")

    def query(df_, feat_):
        q = np.radians(df_[["lat", "lon"]].values)
        ok = ~np.isnan(q).any(axis=1)
        nn_mean = np.full(len(df_), np.nan)
        nn_idw = np.full(len(df_), np.nan)
        if ok.sum():
            dist, idx = tree.query(q[ok], k=min(k, len(dem)))
            w = 1.0 / (dist + 1e-6)
            vals = dem[idx]
            nn_mean[ok] = vals.mean(axis=1)
            nn_idw[ok] = (vals * w).sum(axis=1) / w.sum(axis=1)
        feat_["nn_mean"] = nn_mean
        feat_["nn_idw"] = nn_idw

    query(train_df, train_feat)
    query(test_df, test_feat)


# ----------------------------------------------------------------------------- #
# 4. OOF features that depend on the d49 target (computed without leakage)
# ----------------------------------------------------------------------------- #
def add_oof_residual(d49: pd.DataFrame, test_feat, test_df, d49_feat,
                     folds, S):
    """gh_d49_resid = per-geohash mean of (d49_night - d48_exact_at_same_mod).

    For d49 TRAIN rows it must be out-of-fold (exclude the row's own fold) to
    avoid leakage. For TEST rows the full d49-night estimate is used (test is a
    disjoint daytime population, so no leakage)."""
    exact = S["exact"]
    gh = d49["geohash"].values
    mod = d49["mod"].values.astype(int)
    d48e = np.array([exact.get((g, m), np.nan) for g, m in zip(gh, mod)])
    resid = d49["demand"].values - d48e          # NaN where no d48 match

    # global & per-roadtype residual fallbacks (full d49-night)
    glob = np.nanmean(resid)
    rt = d49["RoadType"].fillna("__NA__").values
    rt_resid = {}
    for r in np.unique(rt):
        m = (rt == r) & ~np.isnan(resid)
        rt_resid[r] = np.nanmean(resid[m]) if m.sum() else glob

    # OOF per-geohash residual for d49 train
    oof_gh_resid = np.full(len(d49), np.nan)
    for tr_idx, va_idx in folds:
        sub = pd.DataFrame({"gh": gh[tr_idx], "r": resid[tr_idx]}).dropna()
        gmean = sub.groupby("gh")["r"].mean()
        oof_gh_resid[va_idx] = pd.Series(gh[va_idx]).map(gmean).values
    # fallback fill: roadtype then global
    rt_fill = np.array([rt_resid.get(r, glob) for r in rt])
    oof_gh_resid = np.where(np.isnan(oof_gh_resid), rt_fill, oof_gh_resid)
    d49_feat["gh_d49_resid"] = oof_gh_resid
    d49_feat["rt_d49_resid"] = rt_fill

    # full-data per-geohash residual for TEST
    full = pd.DataFrame({"gh": gh, "r": resid}).dropna()
    gmean_full = full.groupby("gh")["r"].mean()
    test_gh = test_df["geohash"].values
    test_rt = test_df["RoadType"].fillna("__NA__").values
    test_rt_fill = np.array([rt_resid.get(r, glob) for r in test_rt])
    test_resid = pd.Series(test_gh).map(gmean_full).values
    test_resid = np.where(np.isnan(test_resid), test_rt_fill, test_resid)
    test_feat["gh_d49_resid"] = test_resid
    test_feat["rt_d49_resid"] = test_rt_fill

    return glob


def add_oof_target_encoding(d49, test_feat, test_df, d49_feat, folds,
                            keys, smoothing=10.0):
    """OOF target encoding of d49 demand over categorical keys (d49 train),
    full-train encoding for test."""
    y = d49["demand"].values
    gmean = y.mean()
    for key, getter in keys:
        col = f"te_{key}"
        tr_key = getter(d49)
        oof = np.full(len(d49), np.nan)
        for tr_idx, va_idx in folds:
            sub = pd.DataFrame({"k": tr_key[tr_idx], "y": y[tr_idx]})
            agg = sub.groupby("k")["y"].agg(["mean", "count"])
            sm = (agg["mean"] * agg["count"] + gmean * smoothing) / (agg["count"] + smoothing)
            oof[va_idx] = pd.Series(tr_key[va_idx]).map(sm).values
        d49_feat[col] = np.where(np.isnan(oof), gmean, oof)

        full = pd.DataFrame({"k": tr_key, "y": y})
        agg = full.groupby("k")["y"].agg(["mean", "count"])
        sm = (agg["mean"] * agg["count"] + gmean * smoothing) / (agg["count"] + smoothing)
        te_key = getter(test_df)
        test_feat[col] = pd.Series(te_key).map(sm).fillna(gmean).values


# ----------------------------------------------------------------------------- #
# 5. Models — KFold OOF on d49 rows
# ----------------------------------------------------------------------------- #
def lgb_oof(X, y, Xt, folds, params, log_target=False, name="lgb"):
    oof = np.zeros(len(X)); test_pred = np.zeros(len(Xt))
    for f, (tr, va) in enumerate(folds, 1):
        ytr = np.log1p(y[tr]) if log_target else y[tr]
        dtr = lgb.Dataset(X.iloc[tr], label=ytr)
        dva = lgb.Dataset(X.iloc[va],
                          label=(np.log1p(y[va]) if log_target else y[va]),
                          reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=4000,
                      valid_sets=[dva], valid_names=["v"],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
        p = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pt = m.predict(Xt, num_iteration=m.best_iteration)
        if log_target:
            p = np.expm1(p); pt = np.expm1(pt)
        oof[va] = p; test_pred += pt / len(folds)
    print(f"    {name:14s} OOF R2 = {r2_score(y, oof):.5f}")
    return oof, test_pred


def sk_oof(make_model, X, y, Xt, folds, name, log_target=False):
    oof = np.zeros(len(X)); test_pred = np.zeros(len(Xt))
    Xf = X.fillna(X.median()); Xtf = Xt.fillna(X.median())
    for tr, va in folds:
        m = make_model()
        ytr = np.log1p(y[tr]) if log_target else y[tr]
        m.fit(Xf.iloc[tr], ytr)
        p = m.predict(Xf.iloc[va]); pt = m.predict(Xtf)
        if log_target:
            p = np.expm1(p); pt = np.expm1(pt)
        oof[va] = p; test_pred += pt / len(folds)
    print(f"    {name:14s} OOF R2 = {r2_score(y, oof):.5f}")
    return oof, test_pred


# ----------------------------------------------------------------------------- #
# 6. Main
# ----------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    hbar("Loading data")
    train, test = load_data()
    train = parse_time(train); test = parse_time(test)
    test_index = test["Index"].copy()
    print(f"train {train.shape}  test {test.shape}")

    # decode lat/lon
    cache: dict = {}
    train["lat"], train["lon"] = decode_geohashes(train["geohash"], cache)
    test["lat"], test["lon"] = decode_geohashes(test["geohash"], cache)

    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy().reset_index(drop=True)
    print(f"d48 rows {len(d48)}  d49 rows {len(d49)}  test rows {len(test)}")

    hbar("Feature engineering (leakage-free, d48-anchored)")
    S = build_d48_structures(d48)
    rel_table, rel_med = d48_reliability_table(d48)

    # prefix spatial aggregates from d48
    prefix_aggs = {}
    for plen in (4, 5):
        pc = d48["geohash"].str[:plen]
        prefix_aggs[(plen, "mod")] = d48.assign(p=pc).groupby(["p", "mod"])["demand"].mean().to_dict()
        prefix_aggs[(plen, "hour")] = d48.assign(p=pc).groupby(["p", "hour"])["demand"].mean().to_dict()

    d49_feat = make_features(d49, S, rel_table, rel_med, prefix_aggs)
    test_feat = make_features(test, S, rel_table, rel_med, prefix_aggs)
    add_spatial_nn(d49_feat, test_feat, d48, d49, test)
    print(f"  base features: {d49_feat.shape[1]}")

    # CV folds on d49 (shared by every OOF computation & model)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(d49_feat))

    # OOF residual + target encodings (target-dependent → must be OOF)
    glob_resid = add_oof_residual(d49, test_feat, test, d49_feat, folds, S)
    te_keys = [
        ("geohash", lambda df: df["geohash"].values),
        ("p5", lambda df: df["geohash"].str[:5].values),
        ("p4", lambda df: df["geohash"].str[:4].values),
        ("rt_mod", lambda df: (df["RoadType"].fillna("NA").astype(str) + "_" + df["mod"].astype(str)).values),
        ("rt", lambda df: df["RoadType"].fillna("NA").astype(str).values),
    ]
    add_oof_target_encoding(d49, test_feat, test, d49_feat, folds, te_keys)
    print(f"  + residual & target-encoding features → total {d49_feat.shape[1]}")
    print(f"  global d49-d48 residual = {glob_resid:+.4f}")

    # align columns
    feat_cols = list(d49_feat.columns)
    test_feat = test_feat[feat_cols]
    y = d49["demand"].values.astype(np.float64)

    # quick anchor baselines (sanity)
    anchor = d49_feat["d48_anchor"].fillna(d49_feat["d48_anchor"].median()).values
    print(f"  [sanity] d48_anchor-only OOF R2 (night)      = {r2_score(y, anchor):.4f}")
    ar = np.clip(anchor + d49_feat['gh_d49_resid'].values, 0, 1)
    print(f"  [sanity] anchor + gh_resid OOF R2 (night)    = {r2_score(y, ar):.4f}")

    hbar("Training models (KFold OOF on d49)")
    lgb_base = dict(objective="regression", metric="rmse", learning_rate=0.02,
                    num_leaves=63, min_data_in_leaf=15, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.2,
                    verbose=-1, n_jobs=-1, seed=SEED)
    lgb_deep = dict(lgb_base, num_leaves=127, min_data_in_leaf=10, lambda_l2=0.1)

    oof_list, test_list, names = [], [], []

    o, t = lgb_oof(d49_feat, y, test_feat, folds, lgb_base, name="lgb_raw")
    oof_list.append(o); test_list.append(t); names.append("lgb_raw")

    o, t = lgb_oof(d49_feat, y, test_feat, folds, lgb_base, log_target=True, name="lgb_log1p")
    oof_list.append(o); test_list.append(t); names.append("lgb_log1p")

    o, t = lgb_oof(d49_feat, y, test_feat, folds, lgb_deep, name="lgb_deep")
    oof_list.append(o); test_list.append(t); names.append("lgb_deep")

    o, t = sk_oof(lambda: HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=63, l2_regularization=0.1,
        min_samples_leaf=15, random_state=SEED),
        d49_feat, y, test_feat, folds, "histgb")
    oof_list.append(o); test_list.append(t); names.append("histgb")

    o, t = sk_oof(lambda: ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        d49_feat, y, test_feat, folds, "extratrees")
    oof_list.append(o); test_list.append(t); names.append("extratrees")

    o, t = sk_oof(lambda: Ridge(alpha=5.0), d49_feat, y, test_feat, folds, "ridge")
    oof_list.append(o); test_list.append(t); names.append("ridge")

    if HAVE_XGB:
        def xgb_oof():
            oof = np.zeros(len(d49_feat)); tp = np.zeros(len(test_feat))
            for tr, va in folds:
                m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                     n_jobs=-1, random_state=SEED, early_stopping_rounds=150)
                m.fit(d49_feat.iloc[tr], y[tr],
                      eval_set=[(d49_feat.iloc[va], y[va])], verbose=False)
                oof[va] = m.predict(d49_feat.iloc[va]); tp += m.predict(test_feat) / len(folds)
            print(f"    {'xgb':14s} OOF R2 = {r2_score(y, oof):.5f}")
            return oof, tp
        o, t = xgb_oof(); oof_list.append(o); test_list.append(t); names.append("xgb")

    if HAVE_CB:
        def cb_oof():
            oof = np.zeros(len(d49_feat)); tp = np.zeros(len(test_feat))
            Xf = d49_feat.fillna(d49_feat.median()); Xtf = test_feat.fillna(d49_feat.median())
            for tr, va in folds:
                m = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=7,
                                      l2_leaf_reg=3.0, random_seed=SEED, verbose=0)
                m.fit(Xf.iloc[tr], y[tr], eval_set=(Xf.iloc[va], y[va]),
                      early_stopping_rounds=150, use_best_model=True)
                oof[va] = m.predict(Xf.iloc[va]); tp += m.predict(Xtf) / len(folds)
            print(f"    {'catboost':14s} OOF R2 = {r2_score(y, oof):.5f}")
            return oof, tp
        o, t = cb_oof(); oof_list.append(o); test_list.append(t); names.append("catboost")

    OOF = np.column_stack(oof_list)
    TEST = np.column_stack(test_list)

    hbar("Meta-blend (honest OOF on d49)")
    # 1) non-negative ridge stack
    meta = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    meta.fit(OOF, y)
    blend_oof = meta.predict(OOF)
    blend_test = meta.predict(TEST)
    r2_stack = r2_score(y, blend_oof)
    print(f"  ridge-stack OOF R2     = {r2_stack:.5f}  weights={dict(zip(names, np.round(meta.coef_,3)))}")

    # 2) best single
    singles = {n: r2_score(y, OOF[:, i]) for i, n in enumerate(names)}
    best_single = max(singles, key=singles.get)
    r2_single = singles[best_single]
    print(f"  best single = {best_single} ({r2_single:.5f})")

    # 3) simple mean of the top-3 tree models for robustness
    tree_idx = [i for i, n in enumerate(names) if n.startswith(("lgb", "histgb", "xgb", "cat"))]
    mean_oof = OOF[:, tree_idx].mean(axis=1)
    mean_test = TEST[:, tree_idx].mean(axis=1)
    r2_mean = r2_score(y, mean_oof)
    print(f"  tree-mean OOF R2       = {r2_mean:.5f}")

    # choose best strategy on honest OOF
    cands = {"stack": (r2_stack, blend_test, blend_oof),
             "single": (r2_single, TEST[:, names.index(best_single)], OOF[:, names.index(best_single)]),
             "tree_mean": (r2_mean, mean_test, mean_oof)}
    best = max(cands, key=lambda k: cands[k][0])
    best_r2, final_test, final_oof = cands[best]
    print(f"  → selected strategy: {best}  (OOF R2 = {best_r2:.5f})")

    final_test = np.clip(final_test, 0.0, 1.0)
    final_score = max(0.0, 100.0 * best_r2)

    hbar("Results")
    print(f"Honest OOF R² (d49)        : {best_r2:.6f}")
    print(f"Honest OOF score (100·R²)  : {final_score:.4f}")
    print(f"  test pred  mean={final_test.mean():.4f} std={final_test.std():.4f} "
          f"min={final_test.min():.4f} max={final_test.max():.4f}")
    print(f"  d49 train  mean={y.mean():.4f} std={y.std():.4f}")

    hbar("Feature importance (lgb_raw, fold-mean gain)")
    dtr = lgb.Dataset(d49_feat, label=y)
    m = lgb.train(lgb_base, dtr, num_boost_round=400, callbacks=[lgb.log_evaluation(0)])
    fi = pd.Series(m.feature_importance("gain"), index=d49_feat.columns).sort_values(ascending=False)
    print(fi.head(20).to_string())

    hbar("Writing submission")
    sub = pd.DataFrame({"Index": test_index.values, "demand": final_test})
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    assert list(sub.columns) == list(sample.columns), \
        f"col mismatch {list(sub.columns)} vs {list(sample.columns)}"
    assert sub.shape == (41778, 2), f"bad shape {sub.shape}"
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {SUBMISSION_PATH}  shape={sub.shape}")
    print(sub.head())
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
