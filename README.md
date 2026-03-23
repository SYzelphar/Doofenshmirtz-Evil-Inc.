# Credit Risk Model — Cognitia Challenge

A fully debugged and optimised machine learning pipeline for predicting loan defaults on a heavily imbalanced dataset (~4% default rate). Built as part of the Cognitia ML debugging competition.

---

## Files

| File | Description |
|------|-------------|
| `FIXED_Credit_Risk_Model_v3.py` | Main model script — all 24 bugs fixed, 4 models trained |
| `save_models.py` | Run after v3 to export all models and the preprocessor |
| `saved_models/gb_model.pkl` | GradientBoosting — primary model |
| `saved_models/rf_calibrated.pkl` | RandomForest with isotonic calibration |
| `saved_models/lr_model.pkl` | Logistic Regression baseline |
| `saved_models/preprocessor.pkl` | Fitted ColumnTransformer (scaler + encoder) |
| `saved_models/model_bundle.pkl` | Everything in one file — use this for deployment |
| `saved_models/metadata.json` | Human-readable performance summary |

---

## Setup

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

---

## How to Run

**Step 1 — Train and evaluate:**
```
FIXED_Credit_Risk_Model_v3.py
```

**Step 2 — Save models (run in a new cell after Step 1):**
```
save_models.py
```

Both scripts must run in the same Python/Colab session since Step 2 reads variables from Step 1's memory.

---

## Dataset

`credit_risk_dataset.csv` — 13,266 loan records, 20 columns.

**Target:** `target_flag` (1 = default, 0 = no default) — 4% positive rate.

**Features used (19 total):**

*Numeric:* `person_age`, `annual_inc`, `employment_length`, `loan_amt`, `interest_rate`, `credit_score`, `monthly_income`, `income_ratio` + 6 engineered features

*Categorical:* `home_ownership`, `loan_intent`, `loan_grade`, `employment_type`, `residence_type`

**Columns dropped:**

| Column | Reason |
|--------|--------|
| `loan_status_final` | Post-origination leakage |
| `repayment_flag` | Post-origination leakage |
| `last_payment_status` | Post-origination leakage |
| `random_score_1/2` | Pure noise |
| `duplicate_feature` | Exact copy of another column |

---

## Pipeline Overview

```
Raw CSV
  │
  ├── Drop leakage / noise / duplicate columns
  ├── Normalise inconsistent category strings
  ├── Engineer 6 safe features (loan_to_income, debt_to_income, etc.)
  │
  ├── Train / Test Split  (80/20, stratified)
  │
  ├── Preprocessor (fit on TRAIN only)
  │     ├── Numeric  → median imputation → RobustScaler
  │     └── Categorical → mode imputation → OneHotEncoder
  │
  ├── SMOTE on training set only  (ratio 0.2)
  │
  ├── 5-fold StratifiedKFold CV → tune GB hyperparameters
  ├── 5-fold StratifiedKFold CV → tune RF hyperparameters
  │
  ├── Train GradientBoosting  (primary)
  ├── Train RandomForest + calibrate (isotonic)
  ├── Train LogisticRegression (baseline)
  ├── Build Soft Voting Ensemble (GB×0.5 + RF×0.3 + LR×0.2)
  │
  ├── Threshold selection → GB out-of-fold CV predictions
  │     └── Enforces recall ≥ 65% as hard constraint
  │
  └── Evaluate on held-out test set (touched once, at the end)
        ├── ROC-AUC, PR-AUC, F1, Recall, Precision, Brier
        ├── Confusion matrix
        ├── Fairness analysis by age group
        ├── Bootstrap stability (200 samples, 95% CI)
        └── Decile calibration check
```

---

## Models

| Model | Role | Notes |
|-------|------|-------|
| GradientBoosting | Primary | CV-tuned, best AUC + Recall |
| RandomForest | Secondary | CV-tuned + isotonic calibration |
| LogisticRegression | Baseline | Interpretable, class_weight balanced |
| Soft Voting Ensemble | Combined | GB×0.5 + RF×0.3 + LR×0.2 |

---

## Key Design Decisions

**Why ROC-AUC, not accuracy?**
At 4% default rate, predicting zero defaults always gives ~96% accuracy — completely useless. ROC-AUC measures ranking quality across all thresholds and is immune to class imbalance.

**Why GradientBoosting as primary?**
In testing, GB outperformed RF on every metric — higher AUC, higher recall, lower Brier score. RF is kept for the ensemble contribution.

**Why SMOTE at 0.2, not 0.5?**
The original code used 0.5, creating a 50% minority class in training vs a 4% rate in the real test set — a massive distribution mismatch. A ratio of 0.2 (1 default per 5 non-defaults) keeps the model aware that defaults are rare.

**Why threshold tuning via OOF CV?**
Sweeping thresholds directly on the test set and picking the best F1 is circular evaluation. Out-of-fold predictions from cross-validation give an unbiased estimate. A recall ≥ 65% guard is applied — in credit risk, missing a default costs far more than rejecting a good applicant.

**Why RobustScaler over StandardScaler?**
Financial features like income and loan amount are heavily right-skewed. RobustScaler uses median and IQR instead of mean and standard deviation, making it resistant to outliers.

---

## Bugs Fixed (24 total)

| # | Category | Bug | Fix |
|---|----------|-----|-----|
| 1 | Import | `SimpleImputer` from wrong module | Moved to `sklearn.impute` |
| 2 | Syntax | `epochs=` arg on `LogisticRegression.fit()` | Removed |
| 3 | Data | `X = df` included target in features | `X = df[ALL_FEATURES]` |
| 4 | Preprocessing | `fillna(0)` on financial columns | `SimpleImputer(strategy='median')` |
| 5 | Preprocessing | String columns cast to `int` | `OneHotEncoder` in pipeline |
| 6 | Preprocessing | Inconsistent category strings | `str.lower().strip().replace()` |
| 7 | Data | Exact duplicate column | `df.loc[:, ~df.T.duplicated()]` |
| 8 | Leakage | Scaler fit on full data before split | Pipeline fit on train only |
| 9 | Split | No `stratify=y` in split | `stratify=y` added |
| 10 | Model | `LinearRegression` on binary target | Replaced with classifiers |
| 11 | Imbalance | Class imbalance ignored | SMOTE + `class_weight='balanced'` |
| 12 | Leakage | Post-origination columns used as features | Dropped before processing |
| 13 | Evaluation | Accuracy as headline metric | ROC-AUC + PR-AUC primary |
| 14 | Evaluation | `predict_proba()` used as labels | `[:,1]` + threshold comparison |
| 15 | Threshold | Fixed 0.5 threshold | OOF CV with recall ≥ 65% guard |
| 16 | Evaluation | Train and evaluate on same data | Proper split; test used once |
| 17 | Data | Noise columns included | Dropped before feature definition |
| 18 | Evaluation | No cross-validation | 5-fold `StratifiedKFold` CV |
| 19 | Imbalance | SMOTE ratio 0.5 → 50% train vs 4% test | SMOTE at 0.2 |
| 20 | Reporting | Feature importances misaligned after OHE | `get_feature_names_out()` |
| 21 | Reporting | Fairness analysis `NameError` crash | Proper group mask indexing |
| 22 | Evaluation | No stability estimate | 200-sample bootstrap + 95% CI |
| 23 | Calibration | RF probabilities uncalibrated | `CalibratedClassifierCV` (isotonic) |
| 24 | Reporting | False "production-ready" claim | Honest readiness checklist |

---

## Using Saved Models on New Data

```python
import joblib
import pandas as pd

# Load bundle
bundle = joblib.load('saved_models/model_bundle.pkl')

# Load new applications
new_df = pd.read_csv('new_applications.csv')

# Drop leakage/noise if present
drop = bundle['leakage_cols'] + bundle['noise_cols']
new_df.drop(columns=[c for c in drop if c in new_df.columns], inplace=True)

# Engineer same features
new_df['loan_to_income']          = new_df['loan_amt'] / (new_df['annual_inc'] + 1)
new_df['credit_loan_ratio']       = new_df['credit_score'] / (new_df['loan_amt'] + 1)
new_df['debt_to_income']          = new_df['loan_amt'] / (new_df['monthly_income'] + 1)
new_df['debt_to_income_squared']  = new_df['debt_to_income'] ** 2
new_df['loan_credit_interaction']  = new_df['loan_amt'] * new_df['interest_rate']
new_df['age_squared']              = new_df['person_age'] ** 2

# Preprocess and predict
X_new     = new_df[bundle['all_features']]
X_proc    = bundle['preprocessor'].transform(X_new)
proba     = bundle['gb_model'].predict_proba(X_proc)[:, 1]
predicted = (proba >= bundle['threshold']).astype(int)  # 1 = predicted default
```

---

## Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| ROC-AUC | > 0.85 | Primary metric |
| Recall | ≥ 65% | Missing a default is costly |
| PR-AUC | Maximise | Stricter for severe imbalance |
| Accuracy | Not used | Misleading at 4% imbalance |
