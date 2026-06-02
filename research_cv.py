"""
LB-faithful local validation harness.
======================================
Random-KFold OOF on the full train is inflated (≈0.95 local vs 91.2 LB) because
every geohash/timeslot leaks across folds. We need a local metric that tracks the
REAL task: predict day-49 *daytime* demand (mod 135-825) for geohashes we only
know from *other* times.

Pseudo-test  = day-48 rows at the test mod range (135-825).
Pseudo-train = everything else (day-48 at mods 0-120 & 840-1425, + all day-49).
Train the candidate pipeline on pseudo-train, predict pseudo-test, score R².
This mirrors the LB setup (same time-of-day structure, geohash seen at other mods).

Run candidate feature/model variants through `evaluate()` and compare.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
import pygeohash as pgh
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
warnings.filterwarnings("ignore")

TRAIN = "/home/runtime-terror/Desktop/Github/GridLock/train.csv"
TEST = "/home/runtime-terror/Desktop/Github/GridLock/test.csv"
SEED = 42
TEST_MOD_LO, TEST_MOD_HI = 135, 825


def mod_of(df):
    p = df["timestamp"].str.split(":", expand=True)
    return p[0].astype(int) * 60 + p[1].astype(int)


def base_features(df, cache):
    df = df.copy()
    df["mod"] = mod_of(df)
    df["hour"] = df["mod"] // 60
    df["minute"] = df["mod"] % 60
    rad = 2 * np.pi * df["mod"] / 1440.0
    df["tod_sin"] = np.sin(rad); df["tod_cos"] = np.cos(rad)
    df["is_rush"] = (df["hour"].between(7, 10) | df["hour"].between(17, 20)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["tod_bucket"] = pd.cut(df["hour"], [-1, 5, 11, 16, 20, 23], labels=[0,1,2,3,4]).astype(int)

    def dec(gh):
        if gh in cache: return cache[gh]
        try:
            ll = pgh.decode(gh); lat = getattr(ll,"latitude",ll[0]); lon = getattr(ll,"longitude",ll[1])
        except Exception: lat, lon = np.nan, np.nan
        cache[gh] = (lat, lon); return lat, lon
    o = df["geohash"].map(dec)
    df["lat"] = o.map(lambda x: x[0]); df["lon"] = o.map(lambda x: x[1])
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        df[col] = df[col].fillna("__NA__").astype(str)
    for p in (3,4,5):
        df[f"gh_p{p}"] = df["geohash"].str[:p]
    # composite keys for time-resolved target encoding
    g = df["geohash"].astype(str)
    df["gh_hour"] = g + "_" + df["hour"].astype(str)
    df["gh_tod"] = g + "_" + df["tod_bucket"].astype(str)
    df["gh_mod"] = g + "_" + df["mod"].astype(str)
    df["rt_mod"] = df["RoadType"].astype(str) + "_" + df["mod"].astype(str)
    df["rt_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
    return df


def factorize_cats(tr, te, cols):
    names = []
    for c in cols:
        comb = pd.concat([tr[c], te[c]]).astype(str)
        codes, _ = pd.factorize(comb)
        tr[c+"_enc"] = codes[:len(tr)]; te[c+"_enc"] = codes[len(tr):]
        names.append(c+"_enc")
    return names


def te_fit_transform(tr, te, key, target="demand", smoothing=20.0, nfold=5):
    """OOF target-encode tr, full-fit transform te."""
    gm = tr[target].mean()
    oof = np.full(len(tr), np.nan)
    kf = KFold(nfold, shuffle=True, random_state=SEED)
    for ti, vi in kf.split(tr):
        s = tr.iloc[ti].groupby(key)[target].agg(["mean","count"])
        sm = (s["mean"]*s["count"]+gm*smoothing)/(s["count"]+smoothing)
        oof[vi] = tr.iloc[vi][key].map(sm).values
    tr_enc = pd.Series(oof, index=tr.index).fillna(gm)
    s = tr.groupby(key)[target].agg(["mean","count"])
    sm = (s["mean"]*s["count"]+gm*smoothing)/(s["count"]+smoothing)
    te_enc = te[key].map(sm).fillna(gm)
    return tr_enc.values, te_enc.values


def build_xy(tr, te, te_keys, smoothing, extra_num=None):
    cat_cols = ["RoadType","LargeVehicles","Landmarks","Weather","geohash","gh_p3","gh_p4","gh_p5"]
    cat_names = factorize_cats(tr, te, cat_cols)
    num = ["NumberofLanes","Temperature","lat","lon","hour","minute","mod",
           "tod_sin","tod_cos","is_rush","is_night","tod_bucket","day"]
    if extra_num: num = num + extra_num
    feats = list(num) + cat_names
    for key, sm in te_keys:
        col = f"te_{key}"; smv = sm if sm else smoothing
        tr[col], te[col] = te_fit_transform(tr, te, key, smoothing=smv)
        feats.append(col)
    return tr[feats].copy(), te[feats].copy(), feats, cat_names


def train_predict(Xtr, ytr, Xte, cat_names, params, rounds=1500, nfold=3):
    """KFold-bagged LightGBM; returns mean test prediction."""
    kf = KFold(nfold, shuffle=True, random_state=SEED)
    pred = np.zeros(len(Xte))
    for ti, vi in kf.split(Xtr):
        dtr = lgb.Dataset(Xtr.iloc[ti], label=ytr[ti], categorical_feature=cat_names)
        dva = lgb.Dataset(Xtr.iloc[vi], label=ytr[vi], categorical_feature=cat_names, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred += m.predict(Xte, num_iteration=m.best_iteration) / nfold
    return np.clip(pred, 0, 1)


def train_predict_ensemble(Xtr, ytr, Xte, cat_names, nfold=3):
    """Seed-bagged LightGBM + HistGB + ExtraTrees, simple average (variance reduction)."""
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
    kf = KFold(nfold, shuffle=True, random_state=SEED)
    preds = []
    base = dict(objective="regression", metric="rmse", num_leaves=127, min_data_in_leaf=20,
                feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, lambda_l2=0.1,
                verbose=-1, n_jobs=-1)
    for ti, vi in kf.split(Xtr):
        # seed-bagged LGB
        for sd in (42, 7, 2024):
            p = dict(base, learning_rate=0.03, seed=sd, feature_fraction=0.8 if sd != 42 else 0.85)
            dtr = lgb.Dataset(Xtr.iloc[ti], label=ytr[ti], categorical_feature=cat_names)
            dva = lgb.Dataset(Xtr.iloc[vi], label=ytr[vi], categorical_feature=cat_names, reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=2500, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
            preds.append(m.predict(Xte, num_iteration=m.best_iteration))
        Xf, Xtf = Xtr.fillna(Xtr.median()), Xte.fillna(Xtr.median())
        h = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=0.1, random_state=SEED)
        h.fit(Xf.iloc[ti], ytr[ti]); preds.append(h.predict(Xtf))
        e = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
        e.fit(Xf.iloc[ti], ytr[ti]); preds.append(e.predict(Xtf))
    return np.clip(np.mean(preds, axis=0), 0, 1)


def evaluate(name, te_keys, smoothing=20.0, params=None, extra_num=None, rounds=1500,
             holdout="block", ensemble=False):
    raw = pd.read_csv(TRAIN)
    cache = {}
    df = base_features(raw, cache)
    if holdout == "block":
        # contiguous daytime block of d48 held out (mirrors test's contiguous range)
        is_ptest = (df["day"] == 48) & df["mod"].between(TEST_MOD_LO, TEST_MOD_HI)
    else:
        # scattered: random 20% of d48 daytime cells held out (keeps time-of-day support
        # so per-geohash/roadtype time features can be validated)
        rng = np.random.RandomState(SEED)
        day_mask = (df["day"] == 48) & df["mod"].between(TEST_MOD_LO, TEST_MOD_HI)
        pick = rng.rand(len(df)) < 0.20
        is_ptest = day_mask & pick
    ptrain = df[~is_ptest].reset_index(drop=True)
    ptest = df[is_ptest].reset_index(drop=True)
    Xtr, Xte, feats, cats = build_xy(ptrain, ptest, te_keys, smoothing, extra_num)
    ytr = ptrain["demand"].values
    if params is None:
        params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                      num_leaves=127, min_data_in_leaf=20, feature_fraction=0.85,
                      bagging_fraction=0.85, bagging_freq=1, lambda_l2=0.1,
                      verbose=-1, n_jobs=-1, seed=SEED)
    if ensemble:
        pred = train_predict_ensemble(Xtr, ytr, Xte, cats)
    else:
        pred = train_predict(Xtr, ytr, Xte, cats, params, rounds=rounds)
    yte = ptest["demand"].values
    r2 = r2_score(yte, pred)
    # by roadtype
    rt = ptest["RoadType"].values
    parts = {r: r2_score(yte[rt==r], pred[rt==r]) for r in ["Highway","Street","Residential"] if (rt==r).sum()>5}
    print(f"{name:46s} R2={r2:.4f}  " +
          " ".join(f"{k[:4]}={v:.3f}" for k,v in parts.items()))
    return r2


REG = dict(objective="regression", metric="rmse", learning_rate=0.03,
           num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.3,
           verbose=-1, n_jobs=-1, seed=SEED)

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "block"
    print(f"=== holdout={mode} ===  (block=contiguous daytime; scatter=20% daytime cells)\n")
    base_keys = [("geohash", None), ("gh_p3", None), ("gh_p4", None), ("gh_p5", None)]
    evaluate("BASELINE single-LGB", base_keys, smoothing=20.0, holdout=mode)
    evaluate("BASELINE smooth=10", base_keys, smoothing=10.0, holdout=mode)
    evaluate("ENSEMBLE (3-seed LGB+HistGB+ExtraTrees)", base_keys, smoothing=20.0,
             holdout=mode, ensemble=True)
