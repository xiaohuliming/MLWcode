import json
import os
import pandas as pd
import numpy as np
from scipy import sparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# =========================
# Global settings
# =========================

SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
TARGET_COL = 'Transported'
CAT_FEATURES = [
    'HomePlanet',
    'CryoSleep',
    'Destination',
    'VIP',
    'Deck',
    'Side',
    'Is_Solo',
    'AgeBin',
    'CabinNumBin',
    'HomePlanet_Destination',
    'Deck_Side',
]

DROP_COLS = ['PassengerId', 'Cabin', 'Name', 'Group', 'Surname']


RUN_TUNING = True

RUN_FINAL_TRAINING = True

TUNING_N_ITER = 10
TUNING_N_SPLITS = 3

FINAL_N_SPLITS = 5

# =========================
# Visualization settings
# =========================

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BLEND_N_TRIALS = 1000


# =========================
# Feature engineering
# =========================

def split_cabin(value):
    """Split Cabin into Deck, Num and Side."""
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    parts = str(value).split('/')
    if len(parts) != 3:
        return np.nan, np.nan, np.nan

    deck, num, side = parts
    try:
        num = int(num)
    except ValueError:
        num = np.nan

    return deck, num, side


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic feature engineering for Spaceship Titanic."""
    df = df.copy()

    cabin_parts = df['Cabin'].apply(split_cabin)
    df['Deck'] = cabin_parts.apply(lambda x: x[0])
    df['Num'] = cabin_parts.apply(lambda x: x[1])
    df['Side'] = cabin_parts.apply(lambda x: x[2])

    df['Group'] = df['PassengerId'].astype(str).str.split('_').str[0]
    df['Group_Size'] = df['Group'].map(df['Group'].value_counts())
    df['Is_Solo'] = (df['Group_Size'] == 1).astype(int)

    df['Total_Spend'] = df[SPEND_COLS].fillna(0).sum(axis=1)
    df['No_Spend'] = (df['Total_Spend'] == 0).astype(int)

    df['Luxury_Spend'] = df[['Spa', 'VRDeck']].fillna(0).sum(axis=1)
    df['Basic_Spend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall']].fillna(0).sum(axis=1)
    df['Log_Total_Spend'] = np.log1p(df['Total_Spend'])
    df['Spend_Ratio'] = df['Luxury_Spend'] / (df['Basic_Spend'] + 1)
    df['Spend_Per_Age'] = df['Total_Spend'] / (df['Age'] + 1)

    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[-1, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'],
    )

    df['CabinNumBin'] = pd.qcut(
        df['Num'],
        q=5,
        duplicates='drop',
    ).astype(str)

    name_parts = df['Name'].astype(str).str.split(' ')
    df['Surname'] = name_parts.apply(
        lambda x: x[-1] if len(x) > 1 and x[-1] != 'nan' else np.nan
    )
    df['Family_Size'] = df['Surname'].map(df['Surname'].value_counts())

    df['HomePlanet_Destination'] = df['HomePlanet'].astype(str) + '_' + df['Destination'].astype(str)
    df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

    return df


def fill_with_mode(series: pd.Series) -> pd.Series:
    mode = series.mode(dropna=True)
    if mode.empty:
        return series
    fill_value = mode.iloc[0]
    return series.where(series.notna(), fill_value)


def logical_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Logical missing-value imputation based on Spaceship Titanic domain rules."""
    df = df.copy()

    for col in SPEND_COLS:
        df.loc[(df['CryoSleep'] == True) & (df[col].isna()), col] = 0

    df['Total_Spend'] = df[SPEND_COLS].fillna(0).sum(axis=1)
    df['No_Spend'] = (df['Total_Spend'] == 0).astype(int)

    df.loc[(df['Total_Spend'] > 0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    df.loc[(df['No_Spend'] == 1) & (df['CryoSleep'].isna()), 'CryoSleep'] = True

    group_planet = df.groupby('Group')['HomePlanet'].transform(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )
    df['HomePlanet'] = df['HomePlanet'].fillna(group_planet)

    group_destination = df.groupby('Group')['Destination'].transform(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )
    df['Destination'] = df['Destination'].fillna(group_destination)

    group_vip = df.groupby('Group')['VIP'].transform(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )
    df['VIP'] = df['VIP'].fillna(group_vip)

    for col in ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Side']:
        df[col] = fill_with_mode(df[col])

    for col in SPEND_COLS + [
        'Age',
        'Num',
        'Group_Size',
        'Family_Size',
        'Total_Spend',
        'Luxury_Spend',
        'Basic_Spend',
        'Log_Total_Spend',
        'Spend_Ratio',
        'Spend_Per_Age',
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df['Total_Spend'] = df[SPEND_COLS].fillna(0).sum(axis=1)
    df['No_Spend'] = (df['Total_Spend'] == 0).astype(int)
    df['Luxury_Spend'] = df[['Spa', 'VRDeck']].fillna(0).sum(axis=1)
    df['Basic_Spend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall']].fillna(0).sum(axis=1)
    df['Log_Total_Spend'] = np.log1p(df['Total_Spend'])
    df['Spend_Ratio'] = df['Luxury_Spend'] / (df['Basic_Spend'] + 1)
    df['Spend_Per_Age'] = df['Total_Spend'] / (df['Age'] + 1)

    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[-1, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'],
    )

    df['CabinNumBin'] = pd.qcut(
        df['Num'].rank(method='first'),
        q=5,
        duplicates='drop',
    ).astype(str)

    df['HomePlanet_Destination'] = df['HomePlanet'].astype(str) + '_' + df['Destination'].astype(str)
    df['Deck_Side'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

    return df


def prepare_datasets(train_path='train.csv', test_path='test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    test[TARGET_COL] = np.nan
    combined = pd.concat([train, test], ignore_index=True)

    combined = build_features(combined)
    combined = logical_impute(combined)
    combined = combined.drop(columns=DROP_COLS)

    train_clean = combined[combined[TARGET_COL].notna()].copy()
    test_clean = combined[combined[TARGET_COL].isna()].copy()

    y = train_clean[TARGET_COL].astype(int)
    X = train_clean.drop(columns=[TARGET_COL])
    X_test = test_clean.drop(columns=[TARGET_COL])

    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    return X, y, X_test, test


# =========================
# Encoding utilities
# =========================

def build_encoded_matrices(X_train_fold, X_val_fold, X_test, categorical_cols):
    """One-hot encode categorical variables for non-CatBoost models."""
    categorical_cols = [c for c in categorical_cols if c in X_train_fold.columns]
    numeric_cols = [col for col in X_train_fold.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                categorical_cols,
            ),
            (
                'numeric',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                    ]
                ),
                numeric_cols,
            ),
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train_fold)
    X_val_encoded = preprocessor.transform(X_val_fold)
    X_test_encoded = preprocessor.transform(X_test)

    return X_train_encoded, X_val_encoded, X_test_encoded


def to_dense_float32(matrix):
    """HistGradientBoosting requires dense numeric input."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def search_best_threshold(y_true, prob):
    """Search the best threshold on OOF probabilities."""
    best_threshold = 0.5
    best_acc = 0.0

    for threshold in np.linspace(0.35, 0.65, 1201):
        pred = (prob >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold, best_acc


def get_threshold_curve(y_true, prob, thresholds=None):
    """
    Calculate accuracy under different classification thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.35, 0.65, 1201)

    acc_list = []

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        acc_list.append(acc)

    curve_df = pd.DataFrame({
        "threshold": thresholds,
        "accuracy": acc_list
    })

    best_idx = curve_df["accuracy"].idxmax()
    best_threshold = curve_df.loc[best_idx, "threshold"]
    best_acc = curve_df.loc[best_idx, "accuracy"]

    return curve_df, best_threshold, best_acc


# =========================
# Visualization functions
# =========================


def plot_model_accuracy_comparison(model_summary_df):
    """
    Plot OOF best accuracy comparison for five base models.
    """
    plt.figure(figsize=(9, 5))

    plt.bar(
        model_summary_df["model"],
        model_summary_df["oof_best_acc"]
    )

    plt.ylabel("OOF Best Accuracy")
    plt.xlabel("Model")
    plt.title("OOF Best Accuracy Comparison of Base Models")
    plt.xticks(rotation=25, ha="right")

    for i, value in enumerate(model_summary_df["oof_best_acc"]):
        plt.text(
            i,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cv_accuracy_comparison.png"), dpi=300)
    plt.close()


def plot_fold_accuracy_curves(fold_scores_df):
    """
    Plot validation accuracy across folds for all five base models.
    """
    plt.figure(figsize=(9, 5.5))

    for model_name in fold_scores_df["model"].unique():
        model_df = fold_scores_df[fold_scores_df["model"] == model_name]
        best_idx = model_df["validation_accuracy"].idxmax()
        best_row = model_df.loc[best_idx]
        legend_label = (
            f"{model_name} "
            f"(best: Fold {int(best_row['fold'])}, Acc: {best_row['validation_accuracy']:.4f})"
        )

        plt.plot(
            model_df["fold"],
            model_df["validation_accuracy"],
            marker="o",
            linewidth=2,
            label=legend_label
        )

    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy across 5 Folds")
    plt.xticks([1, 2, 3, 4, 5], ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"])
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "five_fold_validation_accuracy_curve.png"), dpi=300)
    plt.close()


def plot_boosting_training_curves(training_curves_df):
    """
    Plot validation logloss curves for boosting models.
    """
    plt.figure(figsize=(9, 5.5))

    for model_name in training_curves_df["model"].unique():
        model_df = training_curves_df[training_curves_df["model"] == model_name]
        plt.plot(
            model_df["iteration"],
            model_df["validation_logloss"],
            linewidth=2,
            label=model_name
        )

    plt.xlabel("Boosting Iteration / Round")
    plt.ylabel("Validation Logloss")
    plt.title("Boosting Model Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boosting_training_curves.png"), dpi=300)
    plt.close()


def plot_threshold_curve(curve_df, best_threshold, best_acc, model_name):
    """
    Plot threshold optimization curve.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        curve_df["threshold"],
        curve_df["accuracy"],
        linewidth=2
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Best threshold = {best_threshold:.4f}"
    )

    plt.axvline(
        0.5,
        linestyle=":",
        linewidth=1.5,
        label="Default threshold = 0.5000"
    )

    plt.scatter(
        [best_threshold],
        [best_acc],
        s=60
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("OOF Accuracy")
    plt.title(f"Threshold Optimization Curve: {model_name}")
    plt.legend()
    plt.tight_layout()

    file_name = f"threshold_optimization_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(FIG_DIR, file_name), dpi=300)
    plt.close()


def plot_all_model_threshold_curves(threshold_curves, best_threshold_map):
    """
    Plot threshold-accuracy curves for all five base models in one figure.
    """
    plt.figure(figsize=(9, 5.5))

    for model_name, curve_df in threshold_curves.items():
        best_threshold = best_threshold_map[model_name]["threshold"]
        best_acc = best_threshold_map[model_name]["accuracy"]
        legend_label = (
            f"{model_name} "
            f"(best: {best_threshold:.3f})"
        )

        plt.plot(
            curve_df["threshold"],
            curve_df["accuracy"],
            linewidth=2,
            label=legend_label
        )

    plt.axvline(
        0.5,
        linestyle=":",
        linewidth=1.5,
        color="black",
        label="Default threshold = 0.5000"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("OOF Accuracy")
    plt.title("OOF Accuracy vs Classification Threshold for Five Base Models")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "threshold_optimization_all_models.png"), dpi=300)
    plt.close()


def plot_blend_weights(weights):
    """
    Plot optimized model weights in weighted blending.
    """
    weight_df = pd.DataFrame({
        "model": list(weights.keys()),
        "weight": list(weights.values())
    }).sort_values("weight", ascending=False)

    plt.figure(figsize=(8, 5))

    plt.bar(
        weight_df["model"],
        weight_df["weight"]
    )

    plt.ylabel("Optimized Weight")
    plt.xlabel("Model")
    plt.title("Optimized Weights in Weighted Blend")
    plt.xticks(rotation=25, ha="right")

    for i, value in enumerate(weight_df["weight"]):
        plt.text(
            i,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "weighted_blend_weights.png"), dpi=300)
    plt.close()


def plot_stacking_coefficients(model_names, coefficients):
    """
    Plot logistic regression coefficients in stacking.
    """
    coef_df = pd.DataFrame({
        "model": model_names,
        "coefficient": coefficients
    }).sort_values("coefficient", ascending=False)

    plt.figure(figsize=(8, 5))

    plt.bar(
        coef_df["model"],
        coef_df["coefficient"]
    )

    plt.ylabel("Logistic Regression Coefficient")
    plt.xlabel("Base Model")
    plt.title("Meta-Model Coefficients in Stacking")
    plt.xticks(rotation=25, ha="right")

    for i, value in enumerate(coef_df["coefficient"]):
        plt.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stacking_coefficients.png"), dpi=300)
    plt.close()


def plot_final_performance_comparison(performance_df):
    """
    Plot individual models and ensemble methods.
    """
    plt.figure(figsize=(9, 5))

    plt.bar(
        performance_df["method"],
        performance_df["oof_accuracy"]
    )

    plt.ylabel("OOF Accuracy")
    plt.xlabel("Method")
    plt.title("Individual Models vs Ensemble Methods")
    plt.xticks(rotation=25, ha="right")

    for i, value in enumerate(performance_df["oof_accuracy"]):
        plt.text(
            i,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ensemble_performance_comparison.png"), dpi=300)
    plt.close()


def plot_random_search_oof_accuracy(tuning_history_df):
    """
    Plot OOF Accuracy of random search trials for all models.
    X-axis: trial number
    Y-axis: OOF accuracy
    One line for each model
    """
    plt.figure(figsize=(9, 5.5))

    for model_name in tuning_history_df["model"].unique():
        model_df = tuning_history_df[tuning_history_df["model"] == model_name]

        plt.plot(
            model_df["trial"],
            model_df["oof_accuracy"],
            marker="o",
            linewidth=2,
            label=model_name
        )

    plt.xlabel("Random Search Trial")
    plt.ylabel("OOF Accuracy")
    plt.title("OOF Accuracy across Random Search Trials")
    plt.xticks(sorted(tuning_history_df["trial"].unique()))
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "random_search_oof_accuracy_curve.png"),
        dpi=300
    )
    plt.close()


# =========================
# Hyperparameter tuning
# =========================

PARAM_SPACES = {
    "CatBoost": {
        "iterations": [1200, 1500, 2000, 2500],
        "learning_rate": [0.02, 0.025, 0.03, 0.035, 0.04],
        "depth": [5, 6, 7],
        "l2_leaf_reg": [3.0, 4.0, 5.0, 7.0],
        "random_strength": [0.5, 1.0, 2.0],
        "bagging_temperature": [0.3, 0.6, 1.0],
    },

    "LightGBM": {
        "n_estimators": [1200, 1500, 2000, 2500],
        "learning_rate": [0.02, 0.025, 0.03, 0.035, 0.04],
        "num_leaves": [15, 31, 47, 63],
        "min_child_samples": [10, 15, 20, 30, 40],
        "subsample": [0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.75, 0.85, 0.95, 1.0],
        "reg_alpha": [0.0, 0.05, 0.1, 0.3],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    },

    "XGBoost": {
        "n_estimators": [1000, 1500, 2000],
        "learning_rate": [0.02, 0.03, 0.035, 0.04],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 2, 3, 5],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "gamma": [0.0, 0.03, 0.05, 0.1],
        "reg_alpha": [0.0, 0.03, 0.05, 0.1],
        "reg_lambda": [0.8, 1.0, 1.2, 1.5],
    },

    "ExtraTrees": {
        "n_estimators": [500, 800, 1200],
        "max_depth": [None, 8, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
        "bootstrap": [False, True],
    },

    "HistGradientBoosting": {
        "max_iter": [300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.035, 0.05],
        "max_leaf_nodes": [15, 31, 45, 63],
        "min_samples_leaf": [10, 20, 30, 40],
        "l2_regularization": [0.0, 0.03, 0.05, 0.1, 0.2],
    },
}


def evaluate_model_params(model_name, params, X, y, n_splits=3, random_state=42):
    """
    Fast CV evaluation for one model and one parameter set.
    Uses 3-fold CV during tuning to save time.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    oof_prob = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        if model_name == "CatBoost":
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Accuracy",
                cat_features=CAT_FEATURES,
                bootstrap_type="Bayesian",
                verbose=0,
                random_seed=random_state,
                allow_writing_files=False,
                **params
            )

            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=100,
                use_best_model=True,
            )

            val_prob = model.predict_proba(X_val_fold)[:, 1]

        else:
            X_train_encoded, X_val_encoded, _ = build_encoded_matrices(
                X_train_fold,
                X_val_fold,
                X_val_fold,
                CAT_FEATURES,
            )

            if model_name == "LightGBM":
                model = lgb.LGBMClassifier(
                    objective="binary",
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                    **params
                )

                model.fit(
                    X_train_encoded,
                    y_train_fold,
                    eval_set=[(X_val_encoded, y_val_fold)],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=100,
                            verbose=False
                        )
                    ],
                )

                val_prob = model.predict_proba(X_val_encoded)[:, 1]

            elif model_name == "XGBoost":
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                    **params
                )

                model.fit(
                    X_train_encoded,
                    y_train_fold,
                    eval_set=[(X_val_encoded, y_val_fold)],
                    verbose=False,
                )

                val_prob = model.predict_proba(X_val_encoded)[:, 1]

            elif model_name == "ExtraTrees":
                model = ExtraTreesClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    **params
                )

                model.fit(X_train_encoded, y_train_fold)
                val_prob = model.predict_proba(X_val_encoded)[:, 1]

            elif model_name == "HistGradientBoosting":
                X_train_dense = to_dense_float32(X_train_encoded)
                X_val_dense = to_dense_float32(X_val_encoded)

                model = HistGradientBoostingClassifier(
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=random_state,
                    **params
                )

                model.fit(X_train_dense, y_train_fold)
                val_prob = model.predict_proba(X_val_dense)[:, 1]

            else:
                raise ValueError(f"Unknown model name: {model_name}")

        oof_prob[val_idx] = val_prob

    best_threshold, best_acc = search_best_threshold(y.values, oof_prob)

    return best_acc, best_threshold


def tune_one_model(model_name, X, y, n_iter=10, n_splits=3, random_state=42):
    """
    Random search tuning for one model.
    """
    print(f"\n========== TUNING {model_name} ==========")

    sampler = list(
        ParameterSampler(
            PARAM_SPACES[model_name],
            n_iter=n_iter,
            random_state=random_state
        )
    )

    best_score = -1
    best_threshold = 0.5
    best_params = None
    tuning_history = []

    for i, params in enumerate(sampler, start=1):
        score, threshold = evaluate_model_params(
            model_name=model_name,
            params=params,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
        )

        print(f"[{i:02d}/{n_iter}] {model_name} OOF Acc = {score:.4f}, threshold = {threshold:.4f}")
        print(f"Params: {params}")

        tuning_history.append({
            "model": model_name,
            "trial": i,
            "oof_accuracy": score,
            "best_threshold": threshold,
            "params": json.dumps(params)
        })

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_params = params
            print(f"New best {model_name}: {best_score:.4f}")

    print(f"\nBest {model_name} score: {best_score:.4f}")
    print(f"Best {model_name} threshold: {best_threshold:.4f}")
    print(f"Best {model_name} params: {best_params}")
    print("=====================================\n")

    history_df = pd.DataFrame(tuning_history)
    history_df.to_csv(f"tuning_history_{model_name}.csv", index=False)

    return best_params, best_score, best_threshold, history_df


def tune_all_models(X, y, n_iter=10, n_splits=3):
    """
    Tune all five models.
    """
    model_order = [
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "ExtraTrees",
        "HistGradientBoosting",
    ]

    best_results = {}
    all_history = []

    for model_name in model_order:
        best_params, best_score, best_threshold, history_df = tune_one_model(
            model_name=model_name,
            X=X,
            y=y,
            n_iter=n_iter,
            n_splits=n_splits,
            random_state=42,
        )

        all_history.append(history_df)

        best_results[model_name] = {
            "best_params": best_params,
            "best_score": best_score,
            "best_threshold": best_threshold,
        }

    with open("best_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)

    if all_history:
        tuning_history_all_df = pd.concat(all_history, ignore_index=True)
        tuning_history_all_df.to_csv("tuning_history_all_models.csv", index=False)
        plot_random_search_oof_accuracy(tuning_history_all_df)
        print("Saved random search OOF accuracy curve.")

    print("\n========== ALL TUNING RESULTS ==========")
    for model_name, result in best_results.items():
        print(f"\n{model_name}")
        print(f"Best score     : {result['best_score']:.4f}")
        print(f"Best threshold : {result['best_threshold']:.4f}")
        print(f"Best params    : {result['best_params']}")

    print("\nSaved best tuning results to best_tuning_results.json")

    return best_results


# =========================
# Final training
# =========================

def train_and_predict(X, y, X_test, n_splits=5, random_state=42, tuned_params=None):
    if tuned_params is None:
        tuned_params = {}

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model_names = [
        'CatBoost',
        'LightGBM',
        'XGBoost',
        'ExtraTrees',
        'HistGradientBoosting',
    ]

    oof_scores = {name: [] for name in model_names}
    oof_probs = {name: np.zeros(len(X)) for name in model_names}
    test_preds = {name: np.zeros(len(X_test)) for name in model_names}
    boosting_training_curves = {}

    print('Starting 5-Model Cross-Validation...')
    print('Models: CatBoost, LightGBM, XGBoost, ExtraTrees, HistGradientBoosting\n')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f'========== FOLD {fold} ==========')

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        X_train_encoded, X_val_encoded, X_test_encoded = build_encoded_matrices(
            X_train_fold,
            X_val_fold,
            X_test,
            CAT_FEATURES,
        )

        X_train_dense = to_dense_float32(X_train_encoded)
        X_val_dense = to_dense_float32(X_val_encoded)
        X_test_dense = to_dense_float32(X_test_encoded)

        # 1. CatBoost
        cb_params = {
            "iterations": 1500,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.6,
        }
        cb_params.update(tuned_params.get("CatBoost", {}))

        cb = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            cat_features=CAT_FEATURES,
            verbose=0,
            random_seed=random_state,
            allow_writing_files=False,
            **cb_params
        )

        cb.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=100,
            use_best_model=True,
        )

        if fold == 1:
            cb_eval_result = cb.get_evals_result()
            boosting_training_curves['CatBoost'] = pd.DataFrame({
                "iteration": np.arange(
                    1,
                    len(cb_eval_result["validation"]["Logloss"]) + 1
                ),
                "validation_logloss": cb_eval_result["validation"]["Logloss"]
            })

        cb_val_prob = cb.predict_proba(X_val_fold)[:, 1]
        cb_val_pred = (cb_val_prob >= 0.5).astype(int)
        cb_acc = accuracy_score(y_val_fold, cb_val_pred)
        oof_scores['CatBoost'].append(cb_acc)
        oof_probs['CatBoost'][val_idx] = cb_val_prob
        test_preds['CatBoost'] += cb.predict_proba(X_test)[:, 1] / n_splits
        print(f'  CatBoost             Acc@0.5: {cb_acc:.4f}')

        # 2. LightGBM
        lgb_params = {
            "n_estimators": 1500,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        lgb_params.update(tuned_params.get("LightGBM", {}))

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **lgb_params
        )

        lgb_model.fit(
            X_train_encoded,
            y_train_fold,
            eval_set=[(X_val_encoded, y_val_fold)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        if fold == 1:
            lgb_eval_result = lgb_model.evals_result_
            boosting_training_curves['LightGBM'] = pd.DataFrame({
                "iteration": np.arange(
                    1,
                    len(lgb_eval_result["valid_0"]["binary_logloss"]) + 1
                ),
                "validation_logloss": lgb_eval_result["valid_0"]["binary_logloss"]
            })

        lgb_val_prob = lgb_model.predict_proba(X_val_encoded)[:, 1]
        lgb_val_pred = (lgb_val_prob >= 0.5).astype(int)
        lgb_acc = accuracy_score(y_val_fold, lgb_val_pred)
        oof_scores['LightGBM'].append(lgb_acc)
        oof_probs['LightGBM'][val_idx] = lgb_val_prob
        test_preds['LightGBM'] += lgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        print(f'  LightGBM             Acc@0.5: {lgb_acc:.4f}')

        # 3. XGBoost
        xgb_params = {
            "n_estimators": 1500,
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        }
        xgb_params.update(tuned_params.get("XGBoost", {}))

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **xgb_params
        )

        xgb_model.fit(
            X_train_encoded,
            y_train_fold,
            eval_set=[(X_val_encoded, y_val_fold)],
            verbose=False,
        )

        if fold == 1:
            xgb_eval_result = xgb_model.evals_result()
            boosting_training_curves['XGBoost'] = pd.DataFrame({
                "iteration": np.arange(
                    1,
                    len(xgb_eval_result["validation_0"]["logloss"]) + 1
                ),
                "validation_logloss": xgb_eval_result["validation_0"]["logloss"]
            })

        xgb_val_prob = xgb_model.predict_proba(X_val_encoded)[:, 1]
        xgb_val_pred = (xgb_val_prob >= 0.5).astype(int)
        xgb_acc = accuracy_score(y_val_fold, xgb_val_pred)
        oof_scores['XGBoost'].append(xgb_acc)
        oof_probs['XGBoost'][val_idx] = xgb_val_prob
        test_preds['XGBoost'] += xgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        print(f'  XGBoost              Acc@0.5: {xgb_acc:.4f}')

        # 4. ExtraTrees
        et_params = {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": False,
        }
        et_params.update(tuned_params.get("ExtraTrees", {}))

        et_model = ExtraTreesClassifier(
            random_state=random_state,
            n_jobs=-1,
            **et_params
        )

        et_model.fit(X_train_encoded, y_train_fold)
        et_val_prob = et_model.predict_proba(X_val_encoded)[:, 1]
        et_val_pred = (et_val_prob >= 0.5).astype(int)
        et_acc = accuracy_score(y_val_fold, et_val_pred)
        oof_scores['ExtraTrees'].append(et_acc)
        oof_probs['ExtraTrees'][val_idx] = et_val_prob
        test_preds['ExtraTrees'] += et_model.predict_proba(X_test_encoded)[:, 1] / n_splits
        print(f'  ExtraTrees           Acc@0.5: {et_acc:.4f}')

        # 5. HistGradientBoosting
        hgb_params = {
            "max_iter": 500,
            "learning_rate": 0.035,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.05,
        }
        hgb_params.update(tuned_params.get("HistGradientBoosting", {}))

        hgb_model = HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=random_state,
            **hgb_params
        )

        hgb_model.fit(X_train_dense, y_train_fold)
        hgb_val_prob = hgb_model.predict_proba(X_val_dense)[:, 1]
        hgb_val_pred = (hgb_val_prob >= 0.5).astype(int)
        hgb_acc = accuracy_score(y_val_fold, hgb_val_pred)
        oof_scores['HistGradientBoosting'].append(hgb_acc)
        oof_probs['HistGradientBoosting'][val_idx] = hgb_val_prob
        test_preds['HistGradientBoosting'] += hgb_model.predict_proba(X_test_dense)[:, 1] / n_splits
        print(f'  HistGradientBoosting Acc@0.5: {hgb_acc:.4f}')

    print('\n========== FINAL MEAN CV ACCURACY ==========')

    model_summary = []

    for name in model_names:
        mean_acc = np.mean(oof_scores[name])
        threshold, threshold_acc = search_best_threshold(y.values, oof_probs[name])

        model_summary.append({
            "model": name,
            "cv_acc_at_0.5": mean_acc,
            "oof_best_threshold": threshold,
            "oof_best_acc": threshold_acc
        })

        print(
            f'{name:>22} : '
            f'CV Acc@0.5 = {mean_acc:.4f} | '
            f'OOF Best Threshold = {threshold:.4f} | '
            f'OOF Best Acc = {threshold_acc:.4f}'
        )

    model_summary_df = pd.DataFrame(model_summary)
    model_summary_df.to_csv("model_cv_summary.csv", index=False)

    fold_scores = []
    for name in model_names:
        for fold_idx, score in enumerate(oof_scores[name], start=1):
            fold_scores.append({
                "model": name,
                "fold": fold_idx,
                "validation_accuracy": score
            })

    fold_scores_df = pd.DataFrame(fold_scores)
    fold_scores_df.to_csv("fold_validation_accuracy.csv", index=False)

    boosting_curve_frames = []
    for model_name, curve_df in boosting_training_curves.items():
        curve_to_save = curve_df.copy()
        curve_to_save.insert(0, "model", model_name)
        boosting_curve_frames.append(curve_to_save)

    if boosting_curve_frames:
        boosting_curves_df = pd.concat(boosting_curve_frames, ignore_index=True)
        boosting_curves_df.to_csv("boosting_training_curves.csv", index=False)
        plot_boosting_training_curves(boosting_curves_df)

    plot_model_accuracy_comparison(model_summary_df)
    plot_fold_accuracy_curves(fold_scores_df)

    threshold_curves = {}
    best_threshold_map = {}
    for name in model_names:
        curve_df, best_threshold, best_acc = get_threshold_curve(y.values, oof_probs[name])
        threshold_curves[name] = curve_df
        best_threshold_map[name] = {
            "threshold": best_threshold,
            "accuracy": best_acc,
        }
        curve_df.to_csv(f"threshold_curve_{name}.csv", index=False)

    plot_all_model_threshold_curves(threshold_curves, best_threshold_map)

    best_model_name = model_summary_df.sort_values(
        "oof_best_acc",
        ascending=False
    ).iloc[0]["model"]
    curve_df = threshold_curves[best_model_name]
    best_threshold = model_summary_df.loc[
        model_summary_df["model"] == best_model_name,
        "oof_best_threshold"
    ].iloc[0]
    best_acc = model_summary_df.loc[
        model_summary_df["model"] == best_model_name,
        "oof_best_acc"
    ].iloc[0]
    plot_threshold_curve(curve_df, best_threshold, best_acc, best_model_name)

    print('============================================\n')
    print("Saved model summary to model_cv_summary.csv")
    print("Saved fold validation accuracy data to fold_validation_accuracy.csv")
    print("Saved boosting training curve data to boosting_training_curves.csv")
    print("Saved boosting model training curves.")
    print("Saved base model accuracy comparison plot.")
    print("Saved five-fold validation accuracy curve.")
    print("Saved combined threshold optimization plot for five base models.")
    print(f"Saved threshold optimization curve for {best_model_name}.")

    return test_preds, oof_probs, model_summary_df


# =========================
# Submissions and blending
# =========================

def save_individual_submissions(test_preds, oof_probs, y, raw_test):
    """Save one submission file for each model."""
    for model_name, preds_prob in test_preds.items():
        best_threshold, _ = search_best_threshold(y.values, oof_probs[model_name])
        final_preds = (preds_prob >= best_threshold).astype(bool)

        submission = pd.DataFrame(
            {
                'PassengerId': raw_test['PassengerId'],
                'Transported': final_preds,
            }
        )

        file_name = f"submission_{model_name.lower()}.csv"
        submission.to_csv(file_name, index=False)
        print(f'Saved {model_name} submission to {file_name}')


def search_best_blend_weights(oof_probs, y, n_trials=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    model_names = list(oof_probs.keys())

    best_acc = 0
    best_weights = None
    best_threshold = 0.5

    print(f'\nSearching blend weights: {n_trials} trials')

    for i in range(n_trials):
        weights_array = rng.dirichlet(np.ones(len(model_names)))

        blend_oof = np.zeros(len(y))
        for name, weight in zip(model_names, weights_array):
            blend_oof += weight * oof_probs[name]

        threshold, acc = search_best_threshold(y.values, blend_oof)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_weights = dict(zip(model_names, weights_array))

        if (i + 1) % 100 == 0:
            print(f'  searched {i + 1}/{n_trials}, best_acc={best_acc:.4f}')

    return best_weights, best_threshold, best_acc


def save_weighted_blend_submission(test_preds, oof_probs, y, raw_test):
    """
    Save one weighted-blend submission.
    Uses Dirichlet random search over OOF probabilities.
    """
    weights, best_threshold, best_acc = search_best_blend_weights(
        oof_probs,
        y,
        n_trials=BLEND_N_TRIALS,
        random_state=42,
    )

    blend_test = np.zeros(len(raw_test))
    for name, weight in weights.items():
        blend_test += weight * test_preds[name]

    final_preds = (blend_test >= best_threshold).astype(bool)

    submission = pd.DataFrame(
        {
            'PassengerId': raw_test['PassengerId'],
            'Transported': final_preds,
        }
    )

    submission.to_csv('submission_weighted_blend.csv', index=False)

    print('\n========== WEIGHTED BLEND ==========')
    print('Best weights:')
    for name, w in weights.items():
        print(f'  {name:>22}: {w:.4f}')
    print(f'Blend best threshold: {best_threshold:.4f}')
    print(f'Blend OOF accuracy  : {best_acc:.4f}')
    print('Saved weighted blend submission to submission_weighted_blend.csv')

    plot_blend_weights(weights)

    pd.DataFrame({
        "model": list(weights.keys()),
        "weight": list(weights.values())
    }).to_csv("weighted_blend_weights.csv", index=False)

    print("Saved weighted blend weights plot.")
    print('====================================\n')


def save_stacking_submission(test_preds, oof_probs, y, raw_test):
    """
    Stacking: use base model OOF predictions as meta-features,
    train LogisticRegression as a second-layer model with CV,
    and generate the final submission.
    """
    model_names = list(oof_probs.keys())

    X_meta = np.column_stack([oof_probs[name] for name in model_names])
    X_meta_test = np.column_stack([test_preds[name] for name in model_names])

    meta_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof_prob = np.zeros(len(y))
    meta_test_prob = np.zeros(len(raw_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y), start=1):
        meta_model.fit(X_meta[train_idx], y.iloc[train_idx])
        meta_oof_prob[val_idx] = meta_model.predict_proba(X_meta[val_idx])[:, 1]
        meta_test_prob += meta_model.predict_proba(X_meta_test)[:, 1] / skf.n_splits

    best_threshold, best_acc = search_best_threshold(y.values, meta_oof_prob)

    final_preds = (meta_test_prob >= best_threshold).astype(bool)

    submission = pd.DataFrame(
        {
            'PassengerId': raw_test['PassengerId'],
            'Transported': final_preds,
        }
    )

    submission.to_csv('submission_stacking.csv', index=False)

    print('\n========== STACKING ==========')
    print(f'Meta-model          : LogisticRegression')
    print(f'Meta-features       : {model_names}')
    print(f'Stacking threshold  : {best_threshold:.4f}')
    print(f'Stacking OOF accuracy: {best_acc:.4f}')

    meta_model_final = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    meta_model_final.fit(X_meta, y)
    coefs = meta_model_final.coef_[0]

    print('Meta-model coefficients (learned weights):')
    for name, coef in zip(model_names, coefs):
        print(f'  {name:>22}: {coef:.4f}')

    plot_stacking_coefficients(model_names, coefs)

    pd.DataFrame({
        "model": model_names,
        "coefficient": coefs
    }).to_csv("stacking_coefficients.csv", index=False)

    print('Saved stacking coefficients plot.')
    print('Saved stacking submission to submission_stacking.csv')
    print('==============================\n')


# =========================
# Main
# =========================

def main():
    X, y, X_test, raw_test = prepare_datasets()

    tuned_params = {}

    if RUN_TUNING:
        best_results = tune_all_models(
            X,
            y,
            n_iter=TUNING_N_ITER,
            n_splits=TUNING_N_SPLITS,
        )

        tuned_params = {
            model_name: result["best_params"]
            for model_name, result in best_results.items()
        }

    if RUN_FINAL_TRAINING:
        test_preds, oof_probs, _model_summary_df = train_and_predict(
            X,
            y,
            X_test,
            n_splits=FINAL_N_SPLITS,
            random_state=42,
            tuned_params=tuned_params,
        )

        save_individual_submissions(test_preds, oof_probs, y, raw_test)
        save_weighted_blend_submission(test_preds, oof_probs, y, raw_test)
        save_stacking_submission(test_preds, oof_probs, y, raw_test)


if __name__ == '__main__':
    main()