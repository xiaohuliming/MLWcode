# MLWcode - Spaceship Titanic Ensemble Pipeline

AI3023 Machine Learning Workshop course project. Team `closeclaw@MLW`.

Trains five gradient-boosting / tree-based base models, then compares three
ensemble strategies on the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition. The final submission is a hard-voting ensemble of three diverse
single-CatBoost pipelines. Public LB: **0.81529** (global top 1%).

## Files

| File | Description |
| --- | --- |
| `1_main_ensemble.ipynb` | End-to-end pipeline. EDA, FE, 5-model tuning + 5-fold CV, weighted blend, stacking, 3-voter hard voting. Run-all reproduces the final submission. |
| `0_spaceship_titanic_5models.py` | Part 1 standalone script (5 base models + weighted blend + stacking). |
| `2_voter1_Lean_CatBoost.py` | Voter 1 source. Minimal FE + Optuna TPE, 5 seeds x 5 folds. Solo LB 0.81365. |
| `3_voter2_Rich_CatBoost.ipynb` | Voter 2 source. Rich FE with Name encoding + top-27 selection. Solo LB 0.81458. |
| `4_voter3_Reference_CatBoost.ipynb` | Voter 3 source. Kaggle reference baseline. Solo LB 0.81248. |

## Environment

Python 3.9 or later. CPU only (about 3 GB RAM). Tested on Python 3.11 / macOS.

```
pip install pandas numpy scikit-learn matplotlib scipy catboost lightgbm xgboost optuna jupyter
```

## Data

Download `train.csv` and `test.csv` from
https://www.kaggle.com/competitions/spaceship-titanic/data
and place them in the repo root (same directory as the notebooks):

```
MLWcode/
├── 1_main_ensemble.ipynb
├── train.csv    (you provide)
├── test.csv     (you provide)
└── ...
```

## How to run

Open the notebook and run all cells top to bottom:

```
jupyter notebook 1_main_ensemble.ipynb
```

Section-2 flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `RUN_TUNING` | True | Random-search the five base models. False to load cached params. |
| `RUN_FINAL_TRAINING` | True | Train all five base models with 5-fold CV. |
| `RUN_VOTERS_FROM_SCRATCH` | False | False reuses voter CSVs if present. True for a clean reproduction. |

Total runtime is roughly 30-50 minutes on a modern laptop CPU, depending on the
tuning flag.

## Outputs

| File | Strategy | Public LB |
| --- | --- | --- |
| `submission_catboost.csv` | Single CatBoost (tuned) | - |
| `submission_lightgbm.csv` | Single LightGBM (tuned) | - |
| `submission_xgboost.csv` | Single XGBoost (tuned) | - |
| `submission_extratrees.csv` | Single ExtraTrees (tuned) | - |
| `submission_histgradientboosting.csv` | Single HistGradientBoosting (tuned) | - |
| `submission_weighted_blend.csv` | Weighted Blend (soft voting) | 0.80874 |
| `submission_stacking.csv` | Stacking (LR meta-learner) | 0.80500 |
| `submission_v10_majority.csv` | **Hard Voting (3-way majority)** | **0.81529** |

The recommended Kaggle submission is `submission_v10_majority.csv`.

## Method summary

### Part 1 - Five base models

CatBoost, LightGBM, XGBoost, ExtraTrees, HistGradientBoosting, all fed the
same engineered feature matrix (~43 columns) with 5-fold StratifiedKFold.
Random-search tuning (3-fold CV, 10 trials per model).

Two ensemble strategies are then applied on the OOF probabilities:

- Weighted Blend - 1000 Dirichlet random weight samples, pick the highest OOF accuracy.
- Stacking - OOF probabilities as meta-features, LogisticRegression meta-learner with 5-fold CV.

Both methods inflate OOF accuracy but underperform on the leaderboard.

### Part 2 - Hard voting of three single-CatBoost pipelines

Same algorithm (CatBoost), three different feature-engineering pipelines:

| Voter | FE | Tuner | Training | Solo LB |
| --- | --- | --- | --- | --- |
| Lean (v9) | minimal, drops Name | Optuna TPE, 40 trials | 5 seeds x 5 folds = 25 fits | 0.81365 |
| Rich (p5t6) | rich, keeps Name, top-27 selection | RandomSearch | 85/15 single split | 0.81458 |
| Reference (baseline) | basic, Kaggle reference | hand-picked defaults | 70/30 single split | 0.81248 |

For each test row, the final label is the binary majority of the three voters
(at least 2 of 3 predict True).

### Why hard voting wins on this dataset

- The three voters share an algorithm but differ in features, hyperparameter
  direction, and training protocol. Pairwise binary disagreement is about
  5.1% on the test set.
- The Part 1 five-model ensemble had OOF probability correlation rho = 0.977
  across models. They all make the same mistakes, so the ensemble cannot help.
- Soft voting and stacking also inherit probability pollution: rich-FE models
  produce over-confident bimodal probabilities that look reasonable on OOF
  but do not generalise to the leaderboard. Hard voting thresholds at 0.5
  and discards the magnitude.

## Team

closeclaw@MLW (AI3023 Machine Learning Workshop, Spring 2026).
