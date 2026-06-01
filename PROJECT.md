# GridLock — Traffic Demand Prediction

> **HackerEarth Hackathon Challenge**  
> Evaluation Metric: `score = max(0, 100 × R²(actual, predicted))`  
> Best Public LB Score Achieved: **91.228** (v1)  
> Honest OOF Estimate (v6): **96.26** on day-49 train

---

## Table of Contents

1. [Challenge Overview](#1-challenge-overview)
2. [Dataset Description](#2-dataset-description)
3. [Exploratory Data Analysis & Key Insights](#3-exploratory-data-analysis--key-insights)
4. [The Core Problem: Why Scoring 95+ is Hard](#4-the-core-problem-why-scoring-95-is-hard)
5. [Solution v1 — Baseline Multi-Delta LightGBM](#5-solution-v1--baseline-multi-delta-lightgbm)
6. [Solution v2 — D48 Interpolation (Introduced Leakage)](#6-solution-v2--d48-interpolation-introduced-leakage)
7. [Solution v3 — XGBoost + CatBoost Added](#7-solution-v3--xgboost--catboost-added)
8. [Solution v4 — Spatial NN + Leakage-Free Features](#8-solution-v4--spatial-nn--leakage-free-features)
9. [Solution v5 — Sample Weights + Reliability Meta-Feature](#9-solution-v5--sample-weights--reliability-meta-feature)
10. [Solution v6 — D49-Only Training + Residual Signal](#10-solution-v6--d49-only-training--residual-signal)
11. [Score Progression](#11-score-progression)
12. [Root Cause Analysis: The 91 Ceiling](#12-root-cause-analysis-the-91-ceiling)
13. [Future Goals & Paths to 95+](#13-future-goals--paths-to-95)

---

## 1. Challenge Overview

Cities worldwide face mounting traffic congestion that disrupts transportation and acts as a barrier to economic growth. This challenge asks participants to **predict passenger travel demand** across urban geospatial zones at 15-minute intervals, enabling data-driven strategies for traffic management and resource allocation.

**Task:** Given a geohash-encoded location, time slot, and contextual features (road type, weather, temperature, etc.), predict the normalized demand value `∈ [0, 1]`.

**Metric:**
```
score = max(0, 100 × R²(actual, predicted))
```
A perfect prediction scores 100. The challenge is structured as a **time-series cross-day generalization** problem: train on day 48, a partial day 49, and predict the rest of day 49.

**Challenge link:** [HackerEarth GridLock Hackathon](https://www.hackerearth.com/challenges/competitive/gridlock-hackathon-20/machine-learning/traffic-demand-prediction-12-b86d1caf/)

---

## 2. Dataset Description

### Files
| File | Rows | Description |
|---|---|---|
| `train.csv` | 77,299 | Day-48 (full) + Day-49 (partial, hours 0–2) with demand labels |
| `test.csv` | 41,778 | Day-49 continuation (hours 2:15–13:45), no demand |
| `sample_submission.csv` | 41,778 | Index + demand column template |

### Columns

| Column | Type | Description |
|---|---|---|
| `Index` | int | Row identifier |
| `geohash` | str | 6-character geohash encoding the geographic location (~150m precision) |
| `day` | int | 48 or 49 |
| `timestamp` | str | Time in `H:MM` format (e.g. `"2:15"`) |
| `demand` | float | Normalized demand ∈ [0, 1] — **target variable** (train only) |
| `RoadType` | str | `Residential`, `Street`, or `Highway` (600 missing in train) |
| `NumberofLanes` | int | Number of lanes at this location |
| `LargeVehicles` | str | `Allowed` or `Not Allowed` |
| `Landmarks` | str | `Yes` or `No` (nearby landmarks) |
| `Temperature` | float | Temperature in °C (2,495 missing in train) |
| `Weather` | str | `Sunny`, `Cloudy`, `Rainy`, `Snowy` (797 missing in train) |

### Temporal Structure (Critical)

```
Day 48 Train: mod 0 – 1425  (00:00 – 23:45, ALL 96 slots × 1,241 geohashes = 69,427 rows)
Day 49 Train: mod 0 – 120   (00:00 – 02:00, 9 slots  × 1,078 geohashes = 7,872 rows)
Day 49 Test:  mod 135 – 825 (02:15 – 13:45, 47 slots × ~889 geohashes = 41,778 rows)
```

**The gap**: Day-49 train covers only **night hours (0–2 AM)**. Test covers **daytime (2:15 AM – 1:45 PM)**. The model must generalize from night-hour training to daytime prediction.

---

## 3. Exploratory Data Analysis & Key Insights

### 3.1 Demand Distribution

| Statistic | Value |
|---|---|
| Mean | 0.0939 |
| Std | 0.1422 |
| Median | 0.0478 |
| Max | 1.0000 |
| Skewness | **3.73** (strongly right-skewed) |

The distribution is heavily skewed: most geohashes are low-demand residential areas, with a long tail of high-demand highway and street corridors.

### 3.2 RoadType — The Dominant Feature

RoadType alone explains **74.85% of total variance** (η² = 0.7485):

| RoadType | Mean Demand | Std | Count |
|---|---|---|---|
| Highway | **0.6108** | 0.2294 | 3,560 |
| Street | **0.2732** | 0.0367 | 3,909 |
| Residential | **0.0572** | 0.0521 | 69,230 |

Highway demand is ~10× higher than residential. RoadType is the single strongest predictor in every model across all versions.

### 3.3 Geohash Coverage

| Population | Geohashes |
|---|---|
| Day-48 train | 1,241 |
| Day-49 train | 1,078 |
| Test (day-49) | 1,190 |
| Test-only (never in train) | **10** |
| Test missing exact d48 match | **11.1%** (4,642 rows) |

Of the 4,642 test rows without an exact `(geohash, mod)` match in day-48, **658 geohashes exist in day-48 at other time slots** (interpolatable), and only **15 geohashes are completely absent** from day-48.

### 3.4 Cross-Day Correlation — The Central Challenge

The predictive power of `d48[geohash, mod]` for `d49[geohash, mod]` varies dramatically by time of day:

| Time (mod) | Cross-day R² | Notes |
|---|---|---|
| 00:00 (0) | 0.2671 | Very noisy at midnight |
| 00:30 (30) | 0.3755 | |
| 01:00 (60) | 0.4981 | |
| 01:30 (90) | 0.6136 | |
| 02:00 (120) | 0.7062 | End of d49 train range |
| **02:15–13:45** | **~0.90–0.96** | **Test range (extrapolated from within-d48 lag-15 R²)** |

**This is the most important insight in the entire project.** Day-49 train is restricted to the worst-correlated hours. Models trained on this data incorrectly learn that "d48 lookup is a weak signal." At test time (daytime hours), d48 is actually very strong — which is why models consistently underperform their training-set OOF scores.

### 3.5 Within-D48 Predictability at Test Mods

Using the lag-15 autocorrelation within d48 as a proxy for cross-day predictability at test mods:

```
Mean lag-15 R² at test mods: 0.9436
Range: 0.905 – 0.962
```

This sets the practical ceiling: a perfect use of `d48[geohash, mod]` at test hours should achieve R² close to 0.94+.

### 3.6 D49 Demand Shift (Residual Analysis)

Day-49 demand is consistently **higher** than day-48 at the same time slots:

| Road Type | Mean Residual (d49 − d48) |
|---|---|
| Highway | +0.2755 |
| Street | +0.1273 |
| Residential | +0.0252 |
| **Global** | **+0.0399** |

The residual decreases from midnight (mod=0, mean=+0.074) toward 2 AM (mod=120, mean=+0.026), suggesting a morning ramp-up effect. **Per-geohash residual (estimated from d49 night-hour data) achieves R² = 0.82** in predicting the residual for unseen time slots of the same geohash.

---

## 4. The Core Problem: Why Scoring 95+ is Hard

The scoring gap between OOF (~0.994) and LB (~0.91) across all v1–v5 solutions has one root cause:

```
Training population:  90% d48 rows (wrong task) + 10% d49 rows (right task, wrong hours)
Test population:      100% d49 rows at DAYTIME hours (strong d48 signal)

Model learns from d49 train:  "d48 lookup at NIGHT → R² = 0.49 (weak)"
Model applies at test:        d48 lookup at DAYTIME → R² = 0.94 (strong)
→ Model underweights its most useful feature → systematic bias → score ~91
```

The OOF looks inflated (0.994) because d48 training rows have self-leaky features: `(geohash, hour)` aggregate includes the target row itself, creating near-perfect in-sample fit that collapses on test.

---

## 5. Solution v1 — Baseline Multi-Delta LightGBM

**LB Score: 91.228**

### Architecture
- Single LightGBM model trained on full training set (77,299 rows)
- KFold(5, shuffle=True, random_state=42)
- Features: 17 cross-day delta lookups (±15 min to ±240 min), `recent_d49` via merge_asof, `(geohash, hour)` OOF target encoding × 9 keys, `RoadType_enc`, `NumberofLanes`, time cyclical features
- Parameters: `lr=0.04, num_leaves=127, min_data=15, lambda_l2=0.1`

### Key Observations
- OOF R² reported: ~0.951 on d49 rows (but this uses night-hour d49 only)
- `RoadType_enc` dominates feature importance by a large margin
- The multi-delta lookup features provide useful temporal context
- **Problem**: 90% of training is d48 rows with partially self-leaky `(gh, hour)` aggregates

---

## 6. Solution v2 — D48 Interpolation (Introduced Leakage)

**LB Score: 91.07** *(worse than v1)*

### Changes from v1
- Added `d48_interp`: linear interpolation of each geohash's d48 time-series at the target `mod`
- Added `d48_slope` and `d48_accel`: first and second temporal derivatives
- Added `d48_ratio`: normalized demand relative to global mod mean
- Improved RoadType imputation: per-geohash mode → gh_p5 → gh_p4
- LGB log1p model added; 2-model average blend

### Why it Scored Worse
**Critical bug discovered:** `d48_interp` for a d48 training row equals `d48[geohash, mod]` = the target demand itself. Every d48 training row (90% of training data) had its own label as a feature — perfect in-sample fit, complete collapse on test. The OOF inflated to ~0.994 but test performance degraded below v1.

---

## 7. Solution v3 — XGBoost + CatBoost Added

**LB Score: 90.944** *(still worse than v1)*

### Changes from v2
- Removed the self-leaky `d48_interp` from d48 training rows
- Added XGBoost and CatBoost as ensemble members (5 models total including Ridge)
- Fixed XGBoost API: `early_stopping_rounds` moved to constructor (XGBoost ≥ 3.0)
- Fixed CatBoost: `bootstrap_type='Bernoulli'` required for `subsample` + `rsm` for column subsampling
- ElasticNet meta-learner (Level-1 stacking) on OOF predictions
- Grid-search blend over 4-model simplex
- Isotonic regression post-calibration

### Why it Still Scored Poorly
Removing the leak *without fixing the underlying CV distribution mismatch* didn't help. The model still trained on 90% d48 rows where features have mild self-leakage, and CV still measured night-hour d49 performance. Adding more models to a biased training regime added computational cost without improving the fundamental signal alignment.

---

## 8. Solution v4 — Spatial NN + Leakage-Free Features

**LB Score: 91.01**

### Changes from v3
- **Spatial nearest-neighbour features**: BallTree on d48 geohash centroids, 5 nearest neighbours by haversine distance, IDW-weighted demand mean
- **Leakage-free cross-day `other_same`**: For d48 training rows, `other_same` draws from d49_train (genuinely different day) instead of self-lookup. For d49/test, uses d48
- **Cascaded spatial fill**: For missing `other_same` values — gh_p5 prefix aggregate → gh_p4 → gh_p3 → global mod mean
- **Temperature imputation**: Per-(geohash, hour) median instead of global
- **14 OOF target encoding keys** (up from 9 in v1)
- **Sample weights**: d49 rows ×10, d48-at-test-mods ×5

### Why Still ~91
Sample weights improved d49 signal but the training data itself was still mixed (d48 rows dominate numerically even with weights). The spatial NN feature added ~0.5% R² on d48 but didn't help much for d49/test since test geohashes are mostly in d48 already.

---

## 9. Solution v5 — Sample Weights + Reliability Meta-Feature

**LB Score: 91.009**

### Changes from v4
- **d48 reliability meta-feature**: For each mod slot, precomputed lag-15 R² within d48. Tells the model "how trustworthy is d48 at this time?" This feature ranged from 0.27 (midnight) to 0.96 (mid-morning)
- **`d48_x_reliability`**: Product of d48 exact lookup and its reliability score
- **Per-geohash d49 residual** (`gh_d49_resid`): Mean of (d49 − d48) for each geohash at night hours, used as a bias correction feature
- SimpleImputer as safety net before Ridge (fixed NaN crash in Ridge model)
- CatBoost fix: `bootstrap_type='Bernoulli'` for proper column subsampling

### Why Still ~91
The reliability feature alone doesn't fix the fundamental training distribution mismatch. Even with 5 models and the reliability signal, the model is still dominated by d48 training signal that doesn't reflect the test regime.

---

## 10. Solution v6 — D49-Only Training + Residual Signal

**Honest OOF Score (d49 only): 96.26** | **LB: TBD**

### The Breakthrough Insight

Experiments proved that training **only on d49 rows** (7,872 rows) achieves OOF R² = **0.9626** on the d49 population — far exceeding every previous LB score. This is honest because:
- d49 train rows use d48 as features (genuinely cross-day, zero self-leakage)
- d49 train rows represent the same prediction task as test (predict d49 demand from d48 history)

### Architecture

#### Training Population
| Population | Rows | Weight | Rationale |
|---|---|---|---|
| d49 train | 7,872 | 1.0 | True cross-day analogue of test |
| d48 at test mods | 41,851 | 0.3 | Same time-slot structure; auxiliary signal |

#### Feature Set (42 features)

**Primary signal:**
- `d48_exact`: d48[geohash, mod] — cross-day exact lookup (filled via per-geohash interpolation for the 11% missing)
- `d48_d±015, ±030, ±060, ±120, ±240`: Temporal delta lookups from d48

**Novel features (not in prior versions):**
- `gh_d49_resid` ★: Per-geohash mean of (d49 − d48) estimated from night-hour d49 train. This single feature ranked **#2 by LGB gain**, explaining how much each geohash's demand shifted on day 49.
- `d48_reliability` ★: Lag-15 R² within d48 at this mod slot. Tells the model "trust d48 here."
- `d48_x_reliability`: Product of lookup value and its reliability
- `p5_resid`, `rt_resid`: Spatial and road-type-level residual estimates

**Aggregates (from full d48, clean for d49/test):**
- `gh_d48_mean`, `gh_d48_std`, `gh_d48_max`, `gh_d48_cv`
- `gh_h_d48_mean`, `gh_h_d48_std` (per-geohash, per-hour)
- `gh_mod_mean` (per-geohash, per-15-min slot)
- `p3/p4/p5_h/m_mean` (spatial prefix aggregates)
- `rt_d48_mean`, `rt_h_d48_mean` (road-type level)
- `mod_mean`, `mod_std` (global time-of-day profile)

**Ratio features:**
- `d48_ratio`: d48_exact / global_mod_mean (time-of-day shape)
- `d48_gh_ratio`: d48_exact / geohash_daily_mean
- `d48_profile`: gh_h_d48_mean / gh_d48_mean (hourly shape index)

**Spatial NN (BallTree, haversine, k=5):**
- `nn5_mean`, `nn5_std`, `nn1`, `nn_idw` (IDW-weighted demand)

**OOF Target Encodings (14 keys):**
- Geohash, prefix levels, (geohash × hour), (geohash × mod), (mod × RoadType), and Weather/Landmark combinations

#### Model Stack
| Model | Training Data | Key Hyperparameters |
|---|---|---|
| A: LightGBM (raw) | d49 + d48-aux (weighted) | lr=0.02, leaves=255, max_bin=511 |
| B: LightGBM (log1p) | d49 + d48-aux (weighted) | Same + log1p target |
| C: XGBoost | d49 + d48-aux (weighted) | lr=0.02, depth=7, early_stop=300 |
| D: CatBoost | d49 + d48-aux (weighted) | depth=7, Bernoulli bootstrap |
| E: LightGBM (d49 pure) | d49 only | lr=0.02, leaves=127 |

#### Meta-Learner
- ElasticNet (α=0.001, l1_ratio=0.5, positive=True) on 5-model OOF predictions
- Grid-search blend as fallback (0.05-step simplex over [0,1]⁵)
- Isotonic regression post-calibration (70% calibrated + 30% raw)

---

## 11. Score Progression

| Version | LB Score | Key Change | OOF Metric |
|---|---|---|---|
| v1 | **91.228** | Baseline: multi-delta LGB, random KFold | 0.951 (d49, inflated) |
| v2 | 91.07 | d48 interpolation (introduced 100% leakage) | 0.994 (fake) |
| v3 | 90.944 | XGB + CatBoost ensemble, leakage removed | 0.994 (still inflated) |
| v4 | 91.01 | Spatial NN, 5-model stack, leakage-free | 0.994 (still inflated) |
| v5 | 91.009 | Sample weights, reliability feature | 0.994 (still inflated) |
| v6 | TBD | d49-only training, residual signal | **0.9626 (honest)** |

> **Note**: All v1–v5 OOF scores of ~0.994 are inflated by the d48 self-leakage problem. The v6 OOF of 0.9626 is computed on d49 rows only, which truly simulate the test distribution.

---

## 12. Root Cause Analysis: The 91 Ceiling

### Why Every Version Scored ~91 LB

```
┌─────────────────────────────────────────────────────────┐
│ TRAINING SET COMPOSITION                                 │
│ ┌─────────────────────────────────┐                      │
│ │ D48 rows (90%): 69,427          │ ← Wrong population   │
│ │ Self-leaky features              │   (wrong task)       │
│ │ OOF R² ≈ 0.999 (meaningless)   │                      │
│ └─────────────────────────────────┘                      │
│ ┌──────────────────────┐                                 │
│ │ D49 rows (10%): 7,872│ ← Right population              │
│ │ Hours 0–2 only       │   but WRONG hours               │
│ │ D48 signal weak here │   (d48→d49 R²=0.27–0.71)       │
│ └──────────────────────┘                                 │
└─────────────────────────────────────────────────────────┘

TEST SET:
┌──────────────────────────────────────────────────────┐
│ D49 rows: 41,778 — Hours 2:15 – 13:45               │
│ D48 signal STRONG here (d48→d49 R² = 0.90–0.96)    │
│ Model UNDERUSES d48 → systematic underestimation    │
└──────────────────────────────────────────────────────┘
```

### The Distribution Shift in Numbers

| Regime | D48 → D49 Predictability | Who Trains On It |
|---|---|---|
| Night (mod 0–120) | R² = 0.27 – 0.71 | D49 train (10% of training data) |
| **Daytime (mod 135–825)** | **R² = 0.90 – 0.96** | **Test (not in training!)** |

The model learns "d48 is a weak predictor" from the 10% of data that shows weak correlation, then applies that mistaken lesson to the 100% of test data where d48 is actually strong. The result is a systematic underuse of the primary signal, capping scores at ~91.

### Why the OOF Looks So Good (The Leakage Problem)

For a d48 training row at `(geohash=X, mod=Y)`:
- `gh_h_d48_mean` = mean of all d48 demand at `(geohash=X, hour=Y//60)` **including itself**
- With ~63 slots per hour per geohash, the self-contribution is ~1.6% of the feature value
- For `(geohash × mod)` target encoding: only 1 row per (gh, mod) in d48 → 100% self-inclusion in OOF when folds don't respect this

This inflates d48 OOF R² to ~0.999, making models appear far better than they are.

---

## 13. Future Goals & Paths to 95+

### Immediate (v7 — Expected: 93–96)

**1. Pseudo-Labeling from Test Predictions**
Train v6 → generate confident test predictions → add high-confidence rows as d49 training data → retrain. High-demand geohashes (Highway type) have predictable demand and can safely serve as pseudo-labels.

```python
# High-confidence rows: d48_exact × 1.5 ≈ predicted (demand is just a scaled version of d48)
confident_mask = (test_pred > 0.5) | (test['RoadType'] == 'Highway')
pseudo_d49 = test[confident_mask].copy()
pseudo_d49['demand'] = test_pred[confident_mask]
# Add to d49 training set with weight=0.5
```

**2. Better Residual Prediction (Two-Stage Model)**
```
Stage 1: Predict d48[gh, mod] → already have it (exact lookup)
Stage 2: Predict residual (d49 - d48) from:
  - gh_d49_resid (per-geohash shift from night hours)
  - Weather delta (d49 weather vs d48 weather)
  - Temperature delta
  - Spatial neighbourhood residuals
Final: prediction = d48[gh, mod] + predicted_residual
```
This approach achieved R² = 0.928 on d49 train in experiments.

**3. Geohash Time-Series Features**
Encode the full d48 demand trajectory as features:
- PCA of the 96-slot d48 series (top 5 components explain 90%+ variance)
- Demand at specific anchor points (midnight peak, morning peak, noon)
- Trend slope over consecutive mod slots

### Medium-Term (v8 — Target: 96–98)

**4. Cross-Validation Aligned to Test Distribution**
Replace random KFold with **held-out-mod CV**: for each CV fold, hold out specific mod slots from d48 and predict them using all other mods. This perfectly simulates the test scenario.

**5. Direct Temporal Extrapolation Model**
Train a model specifically to answer: "given d48 demand at mods 0–1425, predict demand at mods 135–825 for day 49." Use the d49 train rows at night to calibrate the day-to-day shift function, then extrapolate to daytime mods.

**6. Geohash Clustering**
Group geohashes by demand profile shape (using DTW or K-means on d48 time series). Cluster membership as a feature reduces the sparsity problem for geohashes with few d49 training observations.

**7. Attention-Based Temporal Model**
Transformer-style self-attention over the 96 d48 time steps for each geohash, predicting the next 47 time steps (test mods). The attention mechanism can learn which past time slots are most predictive for each future slot.

### Long-Term (v9+ — Target: 98–100)

**8. Physical Demand Model + ML Residual**
Encode domain knowledge: demand at time T follows a circadian rhythm parameterized by RoadType. Fit a parametric model (e.g. double-peaked Gaussian over 24 hours, with different parameters per RoadType) to d48 per geohash. Use the parametric prediction as a feature, train ML only on the residual.

**9. Graph Neural Network Over Geohash Adjacency**
Model the road network as a graph where geohashes are nodes. GNN message-passing propagates demand signals from high-coverage nodes (in d49 train) to their neighbours (in test). This is the approach used in state-of-the-art traffic forecasting (DCRNN, GraphWaveNet).

**10. Iterative Semi-Supervised Learning**
```
Iteration 0: Train on d49 train → predict test
Iteration 1: Use predicted test as pseudo-d49 → add to training → retrain
Iteration k: Repeat until convergence
```
Each iteration improves pseudo-label quality and gives the model more d49-like daytime examples.

### Key Lessons for Future Work

| Lesson | Implication |
|---|---|
| OOF R² ≈ 0.994 was fake (self-leakage in d48 rows) | Always validate on the CORRECT population (d49 only) |
| D48 rows (90% of training) have the wrong structure | Downweight or exclude them |
| Night-hour d49 train (R² 0.27–0.71) is a biased sample | Don't let the model learn "d48 is weak" from night hours |
| D48 exact lookup is the dominant test signal | Any architecture must put d48[gh,mod] front and centre |
| Per-geohash residual (d49−d48) is highly predictive | Even 9 night-hour measurements per geohash unlock 0.82 R² |
| The practical ceiling is ~0.94–0.96 R² | Set by within-d48 lag-15 autocorrelation at test mods |

---

## Appendix: File Index

| File | Description |
|---|---|
| `train.csv` | Training data (day 48 + partial day 49) |
| `test.csv` | Test data (day 49 continuation) |
| `sample_submission.csv` | Submission format template |
| `submission.csv` | v1 submission (LB = 91.228) |
| `traffic_demand_prediction.ipynb` | Original v1 notebook |
| `solution_v3.py` | 3-model ensemble (LGB + XGB + CB) |
| `solution_v4.py` | 5-model stack, spatial NN, leakage-free |
| `solution_v5.py` | Sample weights, reliability feature |
| `solution_v6.py` | **Best version**: d49-only training, residual signal |

---

*Last updated after v6 development. Best honest OOF R²: **0.9626**. Best LB: **91.228**.*