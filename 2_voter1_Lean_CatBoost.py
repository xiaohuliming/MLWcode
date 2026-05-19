"""
Spaceship Titanic v9 — Optuna-tuned CatBoost on baseline-faithful FE
====================================================================
History:
  baseline (Kaggle top-3 repro): LB 0.81248
  v4  (CatBoost lr=0.05, iter=1500, fancy FE): LB 0.80547
  v6_1 (multi-seed CB + fancy FE, OOF 0.8148): not on LB
  v7  (CB + LGB blend, fancy FE, OOF higher): LB 0.80734
  v8  (baseline FE + 5-fold × 5-seed, baseline params): LB worse than baseline

Key empirical findings from previous versions (preserve these):
  * Fancy FE (Cabin_Region, Age_Group, Surname-Family aggregates, Group_Cryo_Frac)
    inflates OOF but hurts LB — likely overfit/leakage signals.
  * Threshold tuning on OOF overfits — locked at 0.50.
  * 5-seed × 5-fold avg is more conservative than single 70/30 split but did not
    by itself beat the baseline LB. The single baseline run had hand-picked
    params (lr=0.01, iter=2000, depth=6, l2=3) from a Kaggle reference notebook.

v9 hypothesis:
  Maybe better CatBoost params exist for the 5-fold-CV regime. Tune them.
  Use v8's FE (the proven-best on LB) and Optuna over the standard CatBoost
  hyperparameters. To avoid CV-LB drift, we ALSO evaluate the best params
  with two independent CV schemes (3-fold during search, 5-fold × 5-seed for
  the final fit). If they agree, params are robust.

Outputs:
  best_params_v9.json   tuned CatBoost params (for the report)
  optuna_v9_study.pkl   full Optuna study (importance plots, history)
  submission_v9.csv     recommended upload (multi-seed avg @ thr=0.50)
  oof_v9.npy, test_v9.npy, probs_v9.csv   for future blending
"""

import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
TARGET_COL = 'Transported'

SEEDS = [42, 7, 2024, 1234, 99]
N_SPLITS_FINAL = 5
N_SPLITS_OPT = 3
N_TRIALS = 40
OPT_SEED = 42


# --------------------------------------------------------------------------- #
# Feature engineering — bit-for-bit identical to v8 (the proven-best LB FE).
# --------------------------------------------------------------------------- #
def split_cabin(value):
    if pd.isna(value):
        return 'U', np.nan, 'U'
    parts = str(value).split('/')
    if len(parts) != 3:
        return 'U', np.nan, 'U'
    deck, num, side = parts
    try:
        num = int(num)
    except ValueError:
        num = np.nan
    return deck, num, side


def feature_engineer(df_train, df_test):
    out = []
    for df in [df_train.copy(), df_test.copy()]:
        cab = df['Cabin'].apply(split_cabin)
        df['Deck'] = cab.apply(lambda x: x[0])
        df['Num']  = cab.apply(lambda x: x[1])
        df['Side'] = cab.apply(lambda x: x[2])
        df['Group'] = df['PassengerId'].astype(str).str.split('_').str[0]
        out.append(df)
    df_train, df_test = out

    combined = pd.concat([df_train, df_test], ignore_index=True)
    for col in ['HomePlanet', 'Destination']:
        grp_mode = combined.groupby('Group')[col].transform(
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)
        combined[col] = combined[col].fillna(grp_mode)
        combined[col] = combined[col].fillna(combined[col].mode()[0])
    n_train = len(df_train)
    df_train[['HomePlanet','Destination']] = combined.iloc[:n_train][['HomePlanet','Destination']].values
    df_test[['HomePlanet','Destination']]  = combined.iloc[n_train:][['HomePlanet','Destination']].values

    for df in [df_train, df_test]:
        df[SPEND_COLS] = df[SPEND_COLS].fillna(0)
        df['TotalSpend']  = df[SPEND_COLS].sum(axis=1)
        df['NoSpending']  = (df['TotalSpend'] == 0).astype(int)
        df['LuxurySpend'] = df['Spa'] + df['VRDeck']

    for df in [df_train, df_test]:
        df.loc[df['CryoSleep'].isna() & (df['TotalSpend'] > 0), 'CryoSleep'] = False
    cryo_mode = df_train['CryoSleep'].mode()[0]
    for df in [df_train, df_test]:
        df['CryoSleep'] = df['CryoSleep'].fillna(cryo_mode)

    for df in [df_train, df_test]:
        df['Age'] = df['Age'].fillna(df_train['Age'].median())
        df['VIP'] = df['VIP'].fillna(df_train['VIP'].mode()[0])
        num = pd.to_numeric(df['Num'], errors='coerce')
        df['Num'] = num.fillna(num.median())

    for df in [df_train, df_test]:
        df['Group_Size'] = df.groupby('Group')['Group'].transform('count')

    for df in [df_train, df_test]:
        df['Is_Child'] = (df['Age'] <= 12).astype(int)
        for c in SPEND_COLS + ['TotalSpend']:
            df[f'{c}_Log'] = np.log1p(df[c])

    drop = ['PassengerId', 'Cabin', 'Name', 'Group']
    df_test_id = df_test['PassengerId'].copy()
    df_train = df_train.drop(columns=drop)
    df_test  = df_test.drop(columns=drop)

    cols_to_dummy = ['HomePlanet', 'Destination', 'Side', 'Deck']
    df_train = pd.get_dummies(df_train, columns=cols_to_dummy)
    df_test  = pd.get_dummies(df_test,  columns=cols_to_dummy)
    df_test  = df_test.reindex(columns=[c for c in df_train.columns if c != TARGET_COL],
                               fill_value=False)

    for col in ['CryoSleep', 'VIP']:
        df_train[col] = df_train[col].astype(int)
        df_test[col]  = df_test[col].astype(int)

    return df_train, df_test, df_test_id


# --------------------------------------------------------------------------- #
# Optuna objective: 3-fold CV mean accuracy, with per-fold pruning.
# --------------------------------------------------------------------------- #
def objective(trial, X_scaled, y):
    params = {
        'iterations':          3000,
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth':               trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':     trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count':        trial.suggest_int('border_count', 64, 254),
        'loss_function':       'Logloss',
        'eval_metric':         'Accuracy',
        'verbose':             0,
        'random_seed':         OPT_SEED,
    }
    skf = StratifiedKFold(n_splits=N_SPLITS_OPT, shuffle=True, random_state=OPT_SEED)
    fold_accs = []
    for fold, (tr_i, va_i) in enumerate(skf.split(X_scaled, y), start=1):
        X_tr, X_va = X_scaled.iloc[tr_i], X_scaled.iloc[va_i]
        y_tr, y_va = y.iloc[tr_i], y.iloc[va_i]
        cb = CatBoostClassifier(**params)
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va),
               early_stopping_rounds=80, verbose=0)
        p_val = cb.predict_proba(X_va)[:, 1]
        fold_accs.append(accuracy_score(y_va, (p_val > 0.5).astype(int)))
        trial.report(float(np.mean(fold_accs)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_accs))


def main():
    print('Loading data + feature engineering (baseline-faithful)...')
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    df_train, df_test, test_id = feature_engineer(train, test)
    y = df_train[TARGET_COL].astype(int)
    X = df_train.drop(columns=[TARGET_COL])
    X_test = df_test.copy()
    print(f'  X: {X.shape}   X_test: {X_test.shape}')

    # Baseline pipeline includes StandardScaler. Keep it for faithfulness even
    # though CatBoost is scale-invariant — it equates v9 to v8 except for params.
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # ----- Phase 1: Optuna search -----
    print(f'\n=== Phase 1: Optuna ({N_TRIALS} trials, {N_SPLITS_OPT}-fold OOF, seed={OPT_SEED}) ===')
    sampler = optuna.samplers.TPESampler(seed=OPT_SEED)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    t0 = time.time()

    def cb_log(study, trial):
        val = f'{trial.value:.4f}' if trial.value is not None else 'PRUNED'
        elapsed = (time.time() - t0) / 60
        print(f'  trial {trial.number:3d}: {val:>7}  best={study.best_value:.4f}  ({elapsed:.1f} min)')

    study.optimize(lambda t: objective(t, X_scaled, y),
                   n_trials=N_TRIALS, callbacks=[cb_log])

    print(f'\n  Optuna time: {(time.time()-t0)/60:.1f} min')
    print(f'  best 3-fold OOF acc: {study.best_value:.4f}')
    print(f'  best params:')
    for k, v in study.best_params.items():
        print(f'    {k}: {v}')

    with open('best_params_v9.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    with open('optuna_v9_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    # ----- Phase 2: Final fit (5 seeds × 5 folds) with tuned params -----
    print(f'\n=== Phase 2: Final fit ({len(SEEDS)} seeds × {N_SPLITS_FINAL} folds) ===')
    cb_params = dict(study.best_params,
                     iterations=3000,
                     loss_function='Logloss',
                     eval_metric='Accuracy',
                     verbose=0)
    n_train, n_test = len(X), len(X_test)
    oof_per_seed  = np.zeros((len(SEEDS), n_train))
    test_per_seed = np.zeros((len(SEEDS), n_test))
    fold_accs, total_time = [], 0.0

    for si, seed in enumerate(SEEDS):
        print(f'\n  --- seed {seed}  ({si+1}/{len(SEEDS)}) ---')
        skf = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=seed)
        for fold, (tr_i, va_i) in enumerate(skf.split(X_scaled, y), start=1):
            tf = time.time()
            X_tr, X_va = X_scaled.iloc[tr_i], X_scaled.iloc[va_i]
            y_tr, y_va = y.iloc[tr_i], y.iloc[va_i]
            params = dict(cb_params, random_seed=seed)
            cb = CatBoostClassifier(**params)
            cb.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                   early_stopping_rounds=100, verbose=0)
            p_val  = cb.predict_proba(X_va)[:, 1]
            p_test = cb.predict_proba(X_test_scaled)[:, 1]
            oof_per_seed[si, va_i] = p_val
            test_per_seed[si]     += p_test / N_SPLITS_FINAL
            acc = accuracy_score(y_va, (p_val > 0.5).astype(int))
            fold_accs.append(acc)
            dt = time.time() - tf; total_time += dt
            print(f'    fold {fold}/{N_SPLITS_FINAL}  acc={acc:.4f}  time={dt:.1f}s')
        seed_oof = accuracy_score(y, (oof_per_seed[si] > 0.5).astype(int))
        print(f'    seed {seed} OOF acc: {seed_oof:.4f}')

    oof  = oof_per_seed.mean(axis=0)
    test_prob = test_per_seed.mean(axis=0)
    oof_acc = accuracy_score(y, (oof > 0.5).astype(int))

    print('\n========== SUMMARY ==========')
    print(f'  3-fold OOF (Optuna)       : {study.best_value:.4f}')
    print(f'  5-fold × 5-seed OOF (final): {oof_acc:.4f}')
    print(f'  mean fold acc (final)      : {np.mean(fold_accs):.4f}  std {np.std(fold_accs):.4f}')
    print(f'  Phase 2 training time      : {total_time/60:.1f} min')

    pred = (test_prob > 0.5).astype(bool)
    pd.DataFrame({'PassengerId': test_id, TARGET_COL: pred}
                 ).to_csv('submission_v9.csv', index=False)
    pd.DataFrame({'PassengerId': test_id, 'prob_v9': test_prob}
                 ).to_csv('probs_v9.csv', index=False)
    np.save('oof_v9.npy', oof)
    np.save('test_v9.npy', test_prob)
    print(f'\n  saved submission_v9.csv (multi-seed avg @ thr=0.50)')
    print(f'  saved best_params_v9.json + optuna_v9_study.pkl')


if __name__ == '__main__':
    main()
