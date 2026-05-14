# Intoxication Detection from Smartphone Accelerometer

Binary classification of 10-second accelerometer windows as **sober** or **intoxicated** (TAC ≥ 0.08) using the UCI BarCrawl dataset. Three model classes are compared on the same patient-level split: logistic regression and random forest on hand-crafted features, and a 1D CNN on the raw signal.

## What we're testing

Whether smartphone accelerometer signal alone is enough to detect intoxication from gait, and whether the choice of model class (linear / nonlinear tabular / deep) meaningfully changes performance.

## Files

- **`preprocess.py`** -- loads raw accelerometer + TAC data, splits each recording into continuous segments, resamples to 40 Hz, subtracts each subject's per-axis mean (kills orientation as a subject-identity cue), builds 10-second windows, labels them via TAC interpolation, and saves two parallel feature sets:
  - `project/processed/tab/{X,y,pids}.npy` -- engineered features (mean / std / min / max / range, zero-crossing rate, dominant frequency, per axis; plus phone type) _(not included in repo)_
  - `project/processed/seq/{X,y,pids}.npy` -- raw windows downsampled to 20 Hz, shape `(N, 200, 3)` _(not included in repo)_
- **`logistic_regression.py`** —- LR on the tabular features
- **`rf.py`** -- Random Forest on the tabular features
- **`cnn.py`** -- 1D CNN on the raw windows
- **`main.py`** — quick dataset summary: per-participant counts, drunk/sober balance, and feature-level averages for both the tabular and sequential `.npy` files.
- **`feature_analysis.py`** — ranks the tabular features, picks the three features most associated with sober behavior, and plots a comparison to `project/results/feature_analysis.png`.

Each model script applies the same patient-level split, tunes its decision threshold on the validation patient via Youden's J, then reports AUC, sensitivity, and specificity on the held-out test patient, and saves ROC + confusion-matrix plots to `project/results/`.

## Running

```bash
pip install -r requirements.txt

# 1) build the feature files -- must run first
python preprocess.py

# 2) run any of the models, in any order
python logistic_regression.py
python rf.py
python cnn.py
```

`preprocess.py` also writes per-subject raw-signal plots to `project/plots/`.

## Patient-level split

To avoid same-subject leakage, the patient whose drunk-window rate is closest to 50% is held out as the test subject; the second-closest is the validation subject; the remaining patients form the training set. The same split is used by all three model scripts.
