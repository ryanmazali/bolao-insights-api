"""Retreina o Modelo A (resultado), B (gols) e C (BTTS) usando
data/processed/match_features_v2.csv e compara duas configurações:

  ANTES: feature set v2 original (sem features de elenco ricas) +
         sample_weight = tournament_weight * recency_weight
  DEPOIS: feature set v2 novo (com features de elenco ricas) +
          sample_weight = match_weight * recency_weight
          (match_weight = tournament_weight * elo_weight)

Ambas as configurações são treinadas sobre o MESMO conjunto de 19391
partidas (já regenerado com peso composto e features ricas), variando
apenas as colunas de features e o sample_weight, para isolar o efeito
das mudanças. A configuração DEPOIS é a que fica salva em
src/models/saved/ (mesmo formato usado pela API).

Uso:
    python scripts/retrain_compare.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features.build_features import recency_weight, REFERENCE_DATE

# ── FEATURE SETS ────────────────────────────────────────────────────────
OLD_FEATURE_COLS = [
    'home_fifa_points', 'home_goals_for', 'home_goals_against',
    'home_goal_diff', 'home_win_rate', 'home_draw_rate',
    'home_btts_rate', 'home_clean_sheet',
    'home_form_goals_for', 'home_form_goals_against',
    'home_form_win_rate', 'home_form5_pts',
    'home_win_rate_home', 'home_win_rate_neutral',
    'home_avg_opp_points', 'home_sos_goals_for',
    'home_sos_goals_against', 'home_sos_form',
    'home_fm23_attack_strength', 'home_fm23_defense_strength',
    'home_fm23_overall', 'home_fm23_gk_strength', 'home_fm23_std_overall',

    'away_fifa_points', 'away_goals_for', 'away_goals_against',
    'away_goal_diff', 'away_win_rate', 'away_draw_rate',
    'away_btts_rate', 'away_clean_sheet',
    'away_form_goals_for', 'away_form_goals_against',
    'away_form_win_rate', 'away_form5_pts',
    'away_win_rate_away', 'away_win_rate_neutral',
    'away_avg_opp_points', 'away_sos_goals_for',
    'away_sos_goals_against', 'away_sos_form',
    'away_fm23_attack_strength', 'away_fm23_defense_strength',
    'away_fm23_overall', 'away_fm23_gk_strength', 'away_fm23_std_overall',

    'diff_fifa_points', 'diff_goals_for', 'diff_goals_against',
    'diff_win_rate', 'diff_form_win_rate', 'diff_form5',
    'diff_sos_goals', 'diff_sos_form', 'diff_avg_opp',
    'fm23_attack_diff', 'fm23_defense_diff', 'fm23_overall_diff',

    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
    'h2h_goals_avg', 'h2h_n',

    'is_neutral', 'tournament_weight',
]

from src.models.train_v2 import FEATURE_COLS as NEW_FEATURE_COLS


def evaluate_config(df, feature_cols, sample_weight, label, save=False):
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    cols_needed = feature_cols + ['result', 'total_goals', 'btts']
    mask = df[cols_needed].notna().all(axis=1)
    d = df[mask].reset_index(drop=True)
    w = sample_weight[mask.values]

    print(f"Partidas usadas: {len(d)}")

    X = d[feature_cols].values

    # ── MODELO A — Resultado ──────────────────────────────────────────
    le = LabelEncoder()
    y_result = le.fit_transform(d['result'])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_result, w,
        test_size=0.15, random_state=42, stratify=y_result
    )

    xgb_result = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42,
    )

    class_weights = compute_sample_weight('balanced', y_train)
    combined_weights = w_train * class_weights

    model_result = CalibratedClassifierCV(xgb_result, cv=5, method='sigmoid')
    model_result.fit(X_train, y_train, sample_weight=combined_weights)

    y_pred = model_result.predict(X_test)
    y_prob = model_result.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    print(f"[Resultado] Accuracy: {acc:.4f} | Log-loss: {ll:.4f} | AUC-ROC (macro ovr): {auc:.4f}")

    # ── MODELO B — Total de Gols ──────────────────────────────────────
    y_goals = d['total_goals'].values
    X_train_g, X_test_g, y_train_g, y_test_g, w_train_g, _ = train_test_split(
        X, y_goals, w, test_size=0.15, random_state=42
    )

    xgb_goals = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    )
    xgb_goals.fit(X_train_g, y_train_g, sample_weight=w_train_g)

    y_pred_g = xgb_goals.predict(X_test_g)
    mae = mean_absolute_error(y_test_g, y_pred_g)
    over_pred = (y_pred_g > 2.5).astype(int)
    over_true = (y_test_g > 2.5).astype(int)
    over_acc = accuracy_score(over_true, over_pred)
    over_auc = roc_auc_score(over_true, y_pred_g)

    print(f"[Gols] MAE: {mae:.4f} | Over/Under 2.5 acc: {over_acc:.4f} | AUC: {over_auc:.4f}")

    # ── MODELO C — BTTS ───────────────────────────────────────────────
    y_btts = d['btts'].values
    X_train_b, X_test_b, y_train_b, y_test_b, w_train_b, _ = train_test_split(
        X, y_btts, w, test_size=0.15, random_state=42, stratify=y_btts
    )

    xgb_btts = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss', random_state=42,
    )
    model_btts = CalibratedClassifierCV(xgb_btts, cv=5, method='isotonic')
    model_btts.fit(X_train_b, y_train_b, sample_weight=w_train_b)

    btts_pred = model_btts.predict(X_test_b)
    btts_acc = accuracy_score(y_test_b, btts_pred)
    print(f"[BTTS] Accuracy: {btts_acc:.4f}")

    # ── Feature importance (modelo de resultado) ───────────────────────
    importances = np.mean([
        cc.estimator.feature_importances_
        for cc in model_result.calibrated_classifiers_
    ], axis=0)
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features (modelo de resultado):")
    for i in top_idx:
        print(f"  {feature_cols[i]:<35} {importances[i]:.4f}")

    if save:
        os.makedirs('src/models/saved', exist_ok=True)
        joblib.dump(model_result, 'src/models/saved/model_result_v2.pkl')
        joblib.dump(xgb_goals, 'src/models/saved/model_goals_v2.pkl')
        joblib.dump(model_btts, 'src/models/saved/model_btts_v2.pkl')
        joblib.dump(le, 'src/models/saved/label_encoder_v2.pkl')
        with open('src/models/saved/feature_columns_v2.json', 'w') as f:
            json.dump(feature_cols, f)
        print("\n[OK] Modelos salvos em src/models/saved/")

    return {
        'result_acc': acc, 'result_logloss': ll, 'result_auc': auc,
        'goals_mae': mae, 'over_acc': over_acc, 'over_auc': over_auc,
        'btts_acc': btts_acc,
    }


def main():
    df = pd.read_csv('data/processed/match_features_v2.csv', parse_dates=['date'])

    old_recency = df['date'].apply(lambda d: recency_weight(d, REFERENCE_DATE))
    old_sample_weight = (df['tournament_weight'] * old_recency).values
    new_sample_weight = df['sample_weight'].values

    before = evaluate_config(df, OLD_FEATURE_COLS, old_sample_weight, "ANTES (features v2 originais + peso simples)")
    after = evaluate_config(df, NEW_FEATURE_COLS, new_sample_weight, "DEPOIS (features de elenco ricas + peso composto)", save=True)

    print(f"\n{'='*60}\nRESUMO ANTES vs DEPOIS\n{'='*60}")
    print(f"{'Metrica':<22}{'Antes':>10}{'Depois':>10}{'Diff':>10}")
    for key in ['result_acc', 'result_logloss', 'result_auc', 'goals_mae', 'over_acc', 'over_auc', 'btts_acc']:
        delta = after[key] - before[key]
        print(f"{key:<22}{before[key]:>10.4f}{after[key]:>10.4f}{delta:>+10.4f}")


if __name__ == '__main__':
    main()
