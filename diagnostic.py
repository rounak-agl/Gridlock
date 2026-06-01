"""Stricter validation diagnostics for the d49-only model.

1. Leave-mods-out CV (GroupKFold by mod) — simulates predicting UNSEEN time slots,
   the actual test task — vs the shuffle-KFold used for the submission.
2. Test/anchor correlation — confirms the submission tracks the strong daytime
   d48 anchor rather than ignoring it.
"""
import warnings, numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
warnings.filterwarnings("ignore")
import solution as S

train, test = S.load_data()
train = S.parse_time(train); test = S.parse_time(test)
cache = {}
train["lat"], train["lon"] = S.decode_geohashes(train["geohash"], cache)
test["lat"], test["lon"] = S.decode_geohashes(test["geohash"], cache)
d48 = train[train.day == 48].copy()
d49 = train[train.day == 49].copy().reset_index(drop=True)

St = S.build_d48_structures(d48)
rel, relmed = S.d48_reliability_table(d48)
pa = {}
for plen in (4, 5):
    pc = d48["geohash"].str[:plen]
    pa[(plen, "mod")] = d48.assign(p=pc).groupby(["p", "mod"])["demand"].mean().to_dict()
    pa[(plen, "hour")] = d48.assign(p=pc).groupby(["p", "hour"])["demand"].mean().to_dict()

f49 = S.make_features(d49, St, rel, relmed, pa)
ft = S.make_features(test, St, rel, relmed, pa)
S.add_spatial_nn(f49, ft, d48, d49, test)

# ---- leave-mods-out CV (group by mod) ----
groups = d49["mod"].values
gkf = GroupKFold(n_splits=min(5, d49["mod"].nunique()))
folds = list(gkf.split(f49, d49["demand"].values, groups))
S.add_oof_residual(d49, ft, test, f49, folds, St)
te_keys = [("geohash", lambda df: df["geohash"].values),
           ("p5", lambda df: df["geohash"].str[:5].values),
           ("rt_mod", lambda df: (df["RoadType"].fillna("NA").astype(str)+"_"+df["mod"].astype(str)).values),
           ("rt", lambda df: df["RoadType"].fillna("NA").astype(str).values)]
S.add_oof_target_encoding(d49, ft, test, f49, folds, te_keys)
ft = ft[list(f49.columns)]
y = d49["demand"].values.astype(float)

Xf = f49.fillna(f49.median())
oof = np.zeros(len(Xf))
for tr, va in folds:
    m = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42)
    m.fit(Xf.iloc[tr], y[tr]); oof[va] = m.predict(Xf.iloc[va])
print(f"Leave-mods-out (GroupKFold by mod) ExtraTrees OOF R2 = {r2_score(y, oof):.4f}")

# anchor-only and anchor+resid under same regime (already computed standalone ~0.897)
anc = f49["d48_anchor"].fillna(f49["d48_anchor"].median()).values
print(f"  anchor-only R2 (night)            = {r2_score(y, anc):.4f}")

# ---- test prediction vs daytime d48 anchor ----
sub = pd.read_csv(S.SUBMISSION_PATH).sort_values("Index").reset_index(drop=True)
test_sorted = test.sort_values("Index").reset_index(drop=True)
ft_sorted = ft.copy(); ft_sorted["Index"] = test["Index"].values
ft_sorted = ft_sorted.sort_values("Index").reset_index(drop=True)
anchor_test = ft_sorted["d48_anchor"].values
pred = sub["demand"].values
ok = ~np.isnan(anchor_test)
print(f"\nTest submission vs daytime d48 anchor:")
print(f"  corr(pred, d48_anchor)            = {np.corrcoef(pred[ok], anchor_test[ok])[0,1]:.4f}")
print(f"  pred  mean={pred.mean():.4f} std={pred.std():.4f}")
print(f"  anchor mean={np.nanmean(anchor_test):.4f} std={np.nanstd(anchor_test):.4f}")
print(f"  rows with no d48 anchor (interp/fallback) = {(~ok).sum()} ({100*(~ok).sum()/len(pred):.1f}%)")
