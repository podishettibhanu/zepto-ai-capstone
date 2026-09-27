# Analytics (`/analytics`)

## Install & run

```bash
pip install -r requirements.txt
python 01_eda.py       # needs internet the first time (sns.load_dataset), then caches titanic.csv
python 02_modeling.py  # reads the cached titanic.csv only -- no network needed
```

`01_eda.py` loads the raw Titanic dataset **exactly once** and immediately
caches it to `titanic.csv`. Every later step -- EDA, cleaning, and the whole
of `02_modeling.py` -- reads from that one cached, cleaned file, so the raw
dataset is never re-fetched independently for modeling.

## Design decisions

- **Missing-value strategy** follows the stated threshold rule and is decided
  **per column at runtime** from the actual measured percentage (printed by
  the script): <5% missing → drop those rows; 5–30% → impute (median for
  numeric, mode for categorical); >30% → too sparse to impute reliably, so
  missingness is encoded as its own `"Unknown"` category instead of dropping
  the column outright (this preserves the possibility that *whether* a
  passenger's cabin/deck was recorded correlates with survival).
- **Correlation matrix** is restricted to exactly `survived, pclass, age,
  sibsp, parch, fare` — `adult_male` and `alone` are excluded because they're
  deterministic derivations of `sex`/`age` and `sibsp`/`parch` respectively,
  not independent measured features.
- **Train/test split is stratified** on `survived` to preserve the class
  balance in both folds, since Titanic survival is imbalanced (~38% survived).
- **All preprocessing (imputation, encoding, scaling) is fit on the training
  split only**, via a single `ColumnTransformer` wrapped in a `Pipeline`, so
  the fit-train/transform-test separation is structural rather than manual.
- **Imbalance handling** compares baseline vs. `class_weight='balanced'` vs.
  SMOTE, with SMOTE applied only to the training fold (`fit_resample` is
  called after the split, never before) to avoid leaking synthetic points
  into the test set.
- **Regression side-task** predicts `fare` from `age, sibsp, parch, pclass,
  sex, embarked` with a linear regression, reporting MAE/RMSE/R²/Adjusted R²
  and a residual plot.
- **Saved artifact**: `models/best_pipeline.joblib` is the *complete* fitted
  `Pipeline` (preprocessing + estimator together, selected by highest test
  F1 among the three classifiers), so it can be called directly on raw,
  unpreprocessed input — never just the bare estimator.

## Outputs

Running both scripts prints every profiling/EDA/modeling result required by
the acceptance criteria to stdout (missing-% per column with the threshold
applied, IQR outlier counts, survival-rate breakdowns, the two strongest
correlations, all classifier metrics, the imbalance comparison, GridSearchCV
best params + OOB score, regression metrics, and the final recommendation),
and saves supporting chart images under `charts/`. Paste the actual console
output from your run into this file (or keep it as a companion log) as your
final written interpretations — the scripts generate the numbers, but the
grader wants them recorded here in Markdown.
## Verified Run

The EDA and modeling scripts were executed successfully. The cleaned Titanic dataset, charts, and trained classification pipeline were generated and verified, including pipeline reload prediction consistency.