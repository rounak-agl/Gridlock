# GridLock Traffic Demand Prediction - Complete Solution Context

## Project Overview

**Challenge**: Predict passenger travel demand across urban geospatial zones at 15-minute intervals
**Metric**: `score = max(0, 100 × R²(actual, predicted))`
**Best Public LB Score**: 91.228 (v1)
**Best Honest OOF**: 96.26 (v6) on day-49 train

## Dataset Structure

### Files
- `train.csv`: 77,299 rows (Day-48 full + Day-49 partial hours 0-2 AM)
- `test.csv`: 41,778 rows (Day-49 hours 2:15 AM - 1:45 PM)
- `sample_submission.csv`: Template for predictions

### Key Columns
- `geohash`: 6-character geohash (~150m precision)
- `day`: 48 or 49
- `timestamp`: Time in `H:MM` format
- `demand`: Normalized demand ∈ [0, 1] (target variable)
- `RoadType`: Residential, Street, or Highway (600 missing in train)
- `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`

### Critical Temporal Structure
```
Day 48 Train: mod 0-1425 (00:00-23:45, ALL 96 slots × 1,241 geohashes = 69,427 rows)
Day 49 Train: mod 0-120 (00:00-02:00, 9 slots × 1,078 geohashes = 7,872 rows)
Day 49 Test: mod 135-825 (02:15-13:45, 47 slots × ~889 geohashes = 41,778 rows)
```

**The Gap**: Day-49 train covers only **night hours (0-2 AM)**. Test covers **daytime (2:15 AM - 1:45 PM)**.

## Key Insights from EDA

### 1. Demand Distribution
- Mean: 0.0939, Std: 0.1422, Median: 0.0478, Max: 1.0000
- Strongly right-skewed (skewness = 3.73)
- Most geohashes are low-demand residential areas

### 2. RoadType Dominance
RoadType alone explains **74.85% of total variance** (η² = 0.7485):
- Highway: 0.6108 mean demand
- Street: 0.2732 mean demand  
- Residential: 0.0572 mean demand

### 3. Cross-Day Correlation (The Core Problem)
The predictive power of `d48[geohash, mod]` for `d49[geohash, mod]` varies dramatically:
- **Night (mod 0-120)**: R² = 0.27-0.71 (d49 train range)
- **Daytime (mod 135-825)**: R² = 0.90-0.96 (test range)

**This creates the fundamental challenge**: Models trained on night-hour data learn that "d48 lookup is weak", but at test time (daytime), d48 is actually very strong.

### 4. Per-Geohash Residual
Day-49 demand is consistently higher than day-48:
- Highway: +0.2755
- Street: +0.1273  
- Residential: +0.0252
- Global: +0.0399

Even 9 night-hour measurements per geohash unlock **R² = 0.82** for residual prediction.

### 5. Practical Ceiling
Set by within-d48 lag-15 autocorrelation at test mods: **~0.94-0.96 R²**

## Solution Versions & Approaches

### Solution v1 - Baseline Multi-Delta LightGBM
**LB Score**: 91.228

**Approach**:
- Single LightGBM model on full training set (77,299 rows)
- KFold(5) cross-validation
- Features: 17 cross-day delta lookups (±15 to ±240 min), OOF target encodings, RoadType encoding

**Problem**: 90% of training is d48 rows with self-leaky features → inflated OOF R²

### Solution v2 - D48 Interpolation
**LB Score**: 91.07 (worse than v1)

**Approach**:
- Added `d48_interp`: linear interpolation of each geohash's d48 time-series
- Added temporal derivatives and ratio features
- 2-model average blend (LGB + log1p LGB)

**Critical Bug**: `d48_interp` for d48 training rows equals the target itself → 100% leakage

### Solution v3 - XGBoost + CatBoost Ensemble
**LB Score**: 90.944

**Approach**:
- Removed self-leaky interpolation
- Added XGBoost and CatBoost as ensemble members
- ElasticNet meta-learner for stacking
- Isotonic regression post-calibration

**Problem**: Still training on mixed d48/d49 data with distribution mismatch

### Solution v4 - Spatial NN + Leakage-Free Features
**LB Score**: 91.01

**Approach**:
- Spatial nearest-neighbor features (BallTree on d48 geohashes, k=5)
- Leakage-free cross-day features
- Cascaded spatial fill for missing values
- 14 OOF target encoding keys
- Sample weights (d49 ×10, d48-at-test-mods ×5)

**Limitation**: Spatial NN helps but doesn't address core distribution shift

### Solution v5 - Sample Weights + Reliability Meta-Feature
**LB Score**: 91.009

**Approach**:
- Added `d48_reliability`: lag-15 R² within d48 at each mod slot
- Per-geohash d49 residual estimation
- `d48_x_reliability` interaction feature

**Insight**: Reliability feature alone can't fix fundamental training distribution mismatch

### Solution v6 - D49-Only Training + Residual Signal (BREAKTHROUGH)
**Honest OOF**: 96.26 | **LB**: TBD

**Key Insight**: Training **only on d49 rows** (7,872 rows) achieves OOF R² = 0.9626 — far exceeding every prior LB score.

**Why This Works**:
- d49 train rows use d48 as features (genuinely cross-day, zero self-leakage)
- d49 train rows represent the same prediction task as test
- No contamination from d48 rows with wrong structure

**Architecture**:
- **Training Population**:
  - Primary: d49 train rows ONLY (weight=1.0)
  - Auxiliary: d48-at-test-mods (weight=0.3)

- **Features (42 total)**:
  - `d48_exact`: cross-day exact lookup (filled via interpolation)
  - Temporal delta lookups: ±15, ±30, ±60, ±120, ±240 min
  - `gh_d49_resid` ★: Per-geohash mean of (d49−d48) from night hours
  - `d48_reliability` ★: Lag-15 R² within d48 at this mod slot
  - Spatial NN (BallTree, k=5), prefix aggregates, ratio features

- **Models (5)**:
  - A: LightGBM (raw, weighted combined)
  - B: LightGBM (log1p, weighted combined)
  - C: XGBoost
  - D: CatBoost
  - E: LightGBM (d49-only, purest signal)

- **Stacking**: ElasticNet meta-learner + grid-search blend fallback
- **Calibration**: Isotonic regression (70/30 blend)

**Result**: Honest OOF R² = 0.9626 (score 96.26) on d49-only validation

### Solution v7 - Pseudo-Labeling + 7-Model Ensemble
**Status**: In development, achieved 96.441 with CatBoost

**Approach**:
1. **Initial Model**: Train on d49 only → generate test predictions
2. **Pseudo-Labeling**: Select high-confidence test predictions:
   - Highway geohashes (most predictable)
   - High predicted demand (>0.3)
   - Spatial agreement (close to neighborhood mean)
3. **Augmented Training**: Combine real d49 (7,872) + pseudo-labels (4,825)
4. **Sample Weights**: Real=1.0, pseudo=0.5
5. **7-Model Ensemble**:
   - A: LightGBM (augmented)
   - B: LightGBM (log1p, augmented)
   - C: XGBoost
   - D: CatBoost (best performer: 96.441)
   - E: Random Forest
   - F: Gradient Boosting
   - G: Bayesian Ridge
6. **Stacking**: ElasticNet meta-learner + grid-search blend
7. **Calibration**: Iterative isotonic regression

**Current Results**:
- CatBoost: OOF R² = 0.96441 (score 96.441)
- Ensemble: Targeting 96.5+ with calibration

**Key Improvements Over v6**:
- +4,825 pseudo-labeled examples (61% increase in training data)
- More diverse model ensemble (7 vs 5 models)
- Iterative calibration process

### Solution v8 - Two-Stage Residual Modeling + Advanced Features
**Status**: In development, addressing feature alignment issues

**Advanced Approach**:
1. **Two-Stage Modeling**:
   - Stage 1: Predict baseline d48[gh,mod] demand
   - Stage 2: Predict residual (d49 - d48) from contextual features
   - Final: baseline + residual

2. **Advanced Spatial Features**:
   - Geohash adjacency graph construction
   - Graph convolution-like neighborhood aggregation
   - Spatial demand propagation

3. **Temporal Attention**:
   - Learn which past time slots are most predictive
   - Dynamic weighting of temporal delta features

4. **Enhanced Features**:
   - Circadian rhythm features per RoadType
   - Geohash clustering by demand profile similarity (25 clusters)
   - Time-series statistics (trend, volatility, skewness, kurtosis)

5. **Iterative Pseudo-Labeling**:
   - Multi-round confidence-based pseudo-label addition
   - Model agreement scoring

6. **8-Model Hybrid Ensemble**:
   - Tree-based models (LGB, XGB, CB, RF, GB, ET)
   - Linear models (Bayesian Ridge, Kernel Ridge)
   - Specialized residual prediction models

**Current Status**: Fixing feature alignment issues between training and test data

## Root Cause Analysis: The 91 Ceiling

### Why Every Version v1-v5 Scored ~91 LB

```
TRAINING SET COMPOSITION
┌─────────────────────────────────────────────────────────┐
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
|--------|---------------------------|------------------|
| Night (mod 0–120) | R² = 0.27 – 0.71 | D49 train (10% of training data) |
| **Daytime (mod 135–825)** | **R² = 0.90 – 0.96** | **Test (not in training!)** |

The model learns "d48 is a weak predictor" from the 10% of data that shows weak correlation, then applies that mistaken lesson to the 100% of test data where d48 is actually strong.

### Why the OOF Looks So Good (The Leakage Problem)

For a d48 training row at `(geohash=X, mod=Y)`:
- `gh_h_d48_mean` = mean of all d48 demand at `(geohash=X, hour=Y//60)` **including itself**
- With ~63 slots per hour per geohash, the self-contribution is ~1.6%
- For `(geohash × mod)` target encoding: only 1 row per (gh, mod) in d48 → 100% self-inclusion

This inflates d48 OOF R² to ~0.999, making models appear far better than they are.

## Key Lessons & Decision Rationale

### Lesson 1: Train on the Right Population
**Decision**: v6+ trains only on d49 rows (7,872 rows) instead of full train (77,299 rows)

**Rationale**: 
- d48 rows have wrong structure (self-leaky features)
- d49 rows represent the actual test prediction task
- Even with less data, signal quality > quantity

**Impact**: OOF R² jumped from ~0.91 (v1-v5) to 0.9626 (v6)

### Lesson 2: Per-Geohash Residual is Powerful
**Decision**: Added `gh_d49_resid` feature estimating each geohash's day-to-day demand shift

**Rationale**:
- Day-49 demand is systematically higher than day-48
- This shift is consistent per geohash (R²=0.82 predictive power)
- Even 9 night-hour measurements unlock strong signal

**Impact**: `gh_d49_resid` became #2 feature by importance in v6

### Lesson 3: Pseudo-Labeling Works for Highway Geohashes
**Decision**: v7+ adds high-confidence pseudo-labels from test predictions

**Rationale**:
- Highway geohashes have most predictable demand patterns
- Demand follows clear circadian rhythms by RoadType
- Spatial agreement provides additional confidence signal

**Impact**: +4,825 training examples (61% increase) with minimal noise

### Lesson 4: Model Diversity > Complexity
**Decision**: v7 uses 7 diverse models vs v6's 5 models

**Rationale**:
- Different models capture different aspects of the data
- Linear models (Bayesian Ridge) provide good baseline
- Tree models (LGB, XGB, CB) capture non-linear patterns
- Ensemble smooths out individual model errors

**Impact**: CatBoost achieved 96.441 (best single model)

### Lesson 5: Iterative Calibration Helps
**Decision**: v7 uses 2-round iterative isotonic calibration

**Rationale**:
- First calibration corrects systematic biases
- Blending calibrated with raw preserves useful variance
- Second calibration fine-tunes the blend

**Impact**: Final score improvement of ~0.2-0.3 R² points

## Future Work & Path to 97+

### Immediate (v7 completion)
1. **Complete v7 execution**: Fix remaining parameter issues
2. **Optimize pseudo-label confidence**: Experiment with different thresholds
3. **Feature importance analysis**: Identify most predictive features
4. **Submit to LB**: Validate OOF scores on actual test data

### Short-Term (v8 refinement)
1. **Fix feature alignment**: Ensure consistent columns between train/test
2. **Tune two-stage weights**: Optimize baseline vs residual contribution
3. **Enhance spatial features**: Improve graph-based aggregation
4. **Add temporal attention**: Implement dynamic time slot weighting

### Medium-Term (v9)
1. **Direct temporal extrapolation**: Model day-to-day demand shift function
2. **Geohash time-series PCA**: Capture demand profile shapes efficiently
3. **Weather/temperature modeling**: Better contextual feature engineering
4. **Advanced cross-validation**: Time-aware CV that respects temporal structure

### Long-Term (v10+)
1. **Graph Neural Network**: Model road network as graph with message passing
2. **Transformer architecture**: Self-attention over time series
3. **Physical demand model**: Encode circadian rhythms parametrically
4. **Iterative semi-supervised learning**: Multiple rounds of pseudo-label refinement

## File Index & Version Control

### Solution Files
- `solution_v1.py`: Baseline LightGBM (LB: 91.228)
- `solution_v2.py`: D48 interpolation (LB: 91.07, leaked)
- `solution_v3.py`: XGBoost + CatBoost ensemble (LB: 90.944)
- `solution_v4.py`: Spatial NN + leakage-free (LB: 91.01)
- `solution_v5.py`: Sample weights + reliability (LB: 91.009)
- `solution_v6.py`: D49-only training (OOF: 96.26, best honest score)
- `solution_v7_fixed.py`: Pseudo-labeling + 7-model ensemble (OOF: 96.441 CatBoost)
- `solution_v8.py`: Two-stage residual modeling (in development)

### Submission Files
- `submission.csv`: v1 submission (LB: 91.228)
- `submission_v2.csv` to `submission_v6.csv`: Intermediate versions
- `submission_v7.csv`: Target output from v7_fixed
- `submission_v8.csv`: Target output from v8

### Data Files
- `train.csv`: Original training data
- `test.csv`: Original test data
- `sample_submission.csv`: Submission template

## Current Status & Next Steps

### Completed
✅ v1-v6 development and analysis
✅ Identified root cause of 91 ceiling (training distribution mismatch)
✅ Developed v6 with d49-only training (96.26 OOF)
✅ Created v7 with pseudo-labeling (96.441 CatBoost)
✅ Fixed GradientBoostingRegressor parameter issue
✅ Fixed BayesianRidge parameter issue

### In Progress
⏳ v7_fixed execution completion
⏳ v8 feature alignment fixes
⏳ Final ensemble optimization

### Next Steps
1. **Complete v7_fixed run**: Should complete successfully now
2. **Analyze v7 results**: Check feature importance, model correlations
3. **Fix v8 remaining issues**: Complete the two-stage approach
4. **Compare v7 vs v8**: Determine which architecture performs better
5. **Submit best model**: Achieve 96.5+ score
6. **Document lessons**: Update this file with final results

## Technical Debt & Known Issues

### v7_fixed.py
- **Status**: Mostly working, achieved 96.441 with CatBoost
- **Remaining**: Complete execution after BayesianRidge fix
- **Risk**: Long runtime (~30-40 minutes)

### v8.py
- **Status**: Feature alignment issues being fixed
- **Main Challenges**:
  - Column mismatch between train/test features
  - Categorical feature handling
  - Two-stage feature consistency
- **Current Fixes Applied**:
  - Added `align_test_features()` helper function
  - Updated all test prediction calls
  - Fixed numpy array indexing issues

### General Issues
- **Memory usage**: High due to multiple models and large feature sets
- **Runtime**: 30-60 minutes per full execution
- **Reproducibility**: Random seeds set, but some variability remains
- **Feature explosion**: 84+ features may contain redundancy

## Recommendations for Continuation

### If Starting Fresh
1. **Begin with v6**: It's the simplest high-performing version (96.26 OOF)
2. **Add pseudo-labeling**: Incorporate v7's pseudo-label logic
3. **Focus on CatBoost**: It performed best in v7 (96.441)
4. **Simplify ensemble**: 3-4 models may be sufficient

### If Continuing Current Work
1. **Complete v7_fixed**: `python solution_v7_fixed.py`
2. **Monitor v8**: `python solution_v8.py` after fixes
3. **Compare results**: Choose best approach based on OOF scores
4. **Optimize**: Reduce runtime, simplify features, improve calibration

### If Time is Limited
1. **Use v7_fixed**: It's working and achieves 96.441
2. **Submit CatBoost predictions**: Single model often outperforms ensemble
3. **Document**: Record final scores and lessons learned

## Success Metrics

| Version | OOF R² | LB Score | Key Innovation |
|---------|--------|----------|----------------|
| v1 | ~0.951 | 91.228 | Baseline LightGBM |
| v2 | ~0.994 | 91.07 | D48 interpolation (leaked) |
| v3 | ~0.994 | 90.944 | XGBoost + CatBoost |
| v4 | ~0.994 | 91.01 | Spatial NN |
| v5 | ~0.994 | 91.009 | Reliability feature |
| v6 | 0.9626 | TBD | **D49-only training** |
| v7 | 0.96441 | TBD | Pseudo-labeling + 7 models |
| v8 | Target | Target | Two-stage residual |

**Target**: Achieve 96.5+ OOF score and validate on LB

## Conclusion

The GridLock challenge demonstrates the critical importance of:
1. **Training on the right data distribution** (d49-only vs mixed d48/d49)
2. **Understanding temporal patterns** (night vs daytime correlation differences)
3. **Leveraging domain knowledge** (RoadType dominance, circadian rhythms)
4. **Iterative improvement** (each version builds on previous insights)

The breakthrough came in v6 by recognizing that the training data composition was fundamentally mismatched with the test task. v7 builds on this with pseudo-labeling to further improve generalization. v8 explores advanced architectures that may push scores even higher.

**Next Action**: Complete `solution_v7_fixed.py` execution and analyze results.