# GridLock — Solution Writeup (LB-driven, living document)

> This is the authoritative writeup. It is kept current as experiments run.
> Earlier narrative docs (`PROJECT.md`, `SOLUTION_CONTEXT.md`, `SOLUTION_ANALYSIS.md`)
> describe *planned* approaches whose claimed scores were never validated on the
> real leaderboard — and, as the LB later proved, were misleading.

---

## 0. Ground truth: real leaderboard scores

| Approach | Local signal | **Real LB** |
|---|---|---|
| **Original `solution.py`** — full-train single LightGBM | OOF ≈ 0.95 | **91.22** |
| v7 d49-only, leakage-free features, 6-model stack | "honest" OOF 0.9527 | **88.89** |
| v7 d49-only + XGBoost + CatBoost, 8-model stack | "honest" OOF 0.9537 | **88.28** |

**The decisive lesson:** the fancy "d49-only" rewrite scored *higher* on local OOF
but *lower* on the real LB. Local OOF — even the "leakage-free" kind — was a poor
proxy. The simple full-train baseline is the strongest foundation. We abandoned the
d49-only path (archived as `solution_v7_d49only_REJECTED.py`,
`submission_v7_d49only_LB88.csv`) and now build **on top of the original baseline**.

---

## 1. Problem

Predict normalized traffic `demand ∈ [0,1]` for day-49 daytime (mod 135–825 =
02:15–13:45) from day-48 history + context. Metric: `score = max(0, 100·R²)`.

| Split | Rows | mod range | geohashes |
|---|---|---|---|
| d48 train | 69,427 | 0–1425 (full day) | 1,241 |
| d49 train | 7,872 | 0–120 (night) | 1,078 |
| d49 **test** | 41,778 | 135–825 (daytime) | 1,190 |

Test geohashes are almost all present in d48 (only 15 are unseen). So the task is
mostly **cross-day generalization for known locations**, where each test geohash's
full day-48 trajectory is available to the model.

---

## 2. A local validation that actually tracks the LB (`research_cv.py`)

Random-KFold OOF on the full train is inflated (≈0.95 vs LB 0.91) because every
geohash/timeslot leaks across folds. We built two holdout meters:

- **BLOCK holdout** — hold out day-48's *contiguous daytime block* (mod 135–825),
  train on the rest, predict it. Mirrors the real test's contiguous-daytime shape.
  **Baseline scores 0.8845 here — close to the real LB (0.912) and far below the
  inflated 0.95.** This is the trustworthy meter.
- **SCATTER holdout** — hold out a random 20% of day-48 daytime *cells*. Keeps
  time-of-day support, so it can measure time-feature gains — but it is blind to
  block-extrapolation leakage (baseline scores an optimistic 0.9685).

### What the meters revealed (the core insight of this project)

| Variant | BLOCK R² | SCATTER R² | Verdict |
|---|---|---|---|
| Baseline (geohash + prefix encodings) | **0.8845** | 0.9685 | safe |
| + per-geohash time encodings (`gh×mod`, `gh×hour`) | **0.4288** 💥 | 0.9681 | **POISON** |
| + `rt×mod`, `rt×hour` (high-count) | 0.7545 | 0.967 | time-keyed, see note |
| + reg hyperparameters | 0.8839 | 0.9674 | neutral |

**Why the collapse?** `geohash×mod` has ≈1 row per cell, so its target encoding ≈
the label on every training row. The model leans on it, then it goes constant on
unseen daytime → predictions collapse (Highway R² −2.3, Street −20). **This is
exactly the leak that sank d49-only on the LB**, and the block meter catches it.

**Rule adopted:** *no high-cardinality per-geohash time encodings.* Per-geohash
features must be high-count (whole-geohash level, prefixes) or they silently leak.

> Caveat on the meters: the BLOCK holdout removes the entire daytime range, so it
> also over-penalizes *any* time-keyed feature (even safe `rt×mod`) because the
> feature loses its support on the held-out block — whereas the real test *has*
> day-48 daytime data. So BLOCK is the right meter for **detecting leakage** and
> for **robustness**, but it understates the value of legitimately-supported
> daytime features. We therefore changed features conservatively.

---

## 3. Current solution (`solution.py`) — robust ensemble on the proven baseline

Design principle from the LB feedback: **don't touch the winning feature set; add
only what is mathematically safe.** Variance reduction (averaging diverse models)
lowers prediction variance and almost always nudges R² up, with *zero* leakage risk
because the features are identical to the 91.22 baseline.

- **Features:** identical to the original baseline — geohash lat/lon, time/cyclical,
  categorical encodings (RoadType, Weather, Landmarks, LargeVehicles, geohash &
  prefixes), structural per-geohash aggregates, interactions, and OOF target
  encodings on **high-count keys only** (geohash, p3/p4/p5). No per-geohash time
  encodings (proven poison above).
- **Model:** per fold, average of **3-seed LightGBM + HistGradientBoosting +
  ExtraTrees** (5 members). 5-fold CV. Predictions clipped to [0,1].
- **Why these:** seed-bagging cuts LGB variance; HistGB and ExtraTrees add
  algorithmic diversity (different split/regularization behavior) that decorrelates
  errors. All consume the same leak-free features.

### Results

⚠️ **The v8 ensemble run was interrupted before completion** (credit limit), so it
has **no LB score yet** and did not write its submission. `solution.py` is ready and
correct — just run `python3 solution.py` (~15–20 min) to produce the v8 submission.

⚠️ **IMPORTANT — current `submission.csv` is NOT safe.** It still holds the
**rejected d49-only output (LB 88.28)** (signature: mean ≈ 0.1196, std ≈ 0.157).
The original 91.22 baseline submission was overwritten earlier and was not backed up.

**To restore the known-good 91.22 submission**, run:
```
python3 solution_v1_baseline.py     # regenerates submission.csv at LB 91.22
```
**To try the improved ensemble (untested on LB)**, run:
```
python3 solution.py                 # overwrites submission.csv with the v8 ensemble
```
Recommended order: regenerate the baseline first (safe fallback), copy it aside,
then run v8 and compare on the LB.

---

## 4. What we tried and rejected (so we don't repeat it)

| Idea | Outcome | Why rejected |
|---|---|---|
| d49-only training | LB 88.89 / 88.28 | discards d48 daytime signal; night-only is the wrong regime |
| `geohash×mod` / `geohash×hour` target encoding | BLOCK 0.88→0.43 | ~1 row/cell ⇒ leaks label, collapses on unseen daytime |
| `d48_interp` exact lookup as feature (docs v2) | LB 91.07 | self-leak on d48 rows |
| heavy reg hyperparameters | BLOCK neutral | no gain |

---

## 5. Next experiments (ranked by expected safety × upside)

1. **Confirm the ensemble beats 91.22 on LB**, then tune blend weights (currently a
   simple mean) toward whichever members are individually strongest.
2. **Add a *cleanly-supported* `RoadType×hour` global profile feature** (3×24 cells,
   thousands of rows each — no per-row leak) and A/B on the LB. In the real test
   this has full daytime support, so it can capture the highway daytime peak that
   the daily-average encoding misses; the BLOCK meter can't credit it, so the LB is
   the judge.
3. **Per-geohash demand shape via high-count buckets** (`geohash × tod_bucket`,
   ~12 rows/cell): borderline — test on BLOCK first; only ship if it doesn't drop.
4. **Quantile/Huber objective or log1p** for the skewed target — robustness.
5. **Isotonic/линейная calibration on a clean held-out split** — only if it helps BLOCK.

Each will be validated on BLOCK (no-collapse gate) and decided on the real LB.

---

## 6. File index

| File | Role |
|---|---|
| `solution.py` | **Current**: robust ensemble on the proven baseline features |
| `solution_v1_baseline.py` | Original single-LGB baseline (LB **91.22**) |
| `research_cv.py` | BLOCK + SCATTER holdout meters; leakage detector |
| `solution_v7_d49only_REJECTED.py` | Archived d49-only rewrite (LB 88) — kept as a cautionary record |
| `submission_v7_d49only_LB88.csv` | Its submission |
| `submission.csv` | Current submission from `solution.py` |
| `SOLUTION_WRITEUP.md` | This living document |

---

*Methodology note: from here on, the **real LB is the only metric we trust for
ranking**. Local meters are used to (a) detect leakage before submitting and (b)
reject changes that collapse. We change one thing at a time so each LB delta is
attributable.*
