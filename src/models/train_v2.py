import sys
import pandas as pd
import numpy as np
import joblib
import json
import os

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, classification_report, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.models.calibration import BinnedCalibrator, get_elo_bin

FEATURE_COLS = [
    'home_fifa_points', 'home_goals_for', 'home_goals_against',
    'home_goal_diff', 'home_win_rate', 'home_draw_rate',
    'home_btts_rate', 'home_clean_sheet',
    'home_form_goals_for', 'home_form_goals_against',
    'home_form_win_rate', 'home_form5_pts',
    'home_win_rate_home', 'home_win_rate_neutral',
    'home_avg_opp_points', 'home_sos_goals_for',
    'home_sos_goals_against', 'home_sos_form',
    'home_fm23_attack_strength', 'home_fm23_best_attacker', 'home_fm23_top3_attack',
    'home_fm23_defense_strength', 'home_fm23_best_defender', 'home_fm23_top5_defense',
    'home_fm23_overall', 'home_fm23_best_overall', 'home_fm23_top11_overall',
    'home_fm23_depth_overall', 'home_fm23_std_overall', 'home_fm23_gk_strength',

    'away_fifa_points', 'away_goals_for', 'away_goals_against',
    'away_goal_diff', 'away_win_rate', 'away_draw_rate',
    'away_btts_rate', 'away_clean_sheet',
    'away_form_goals_for', 'away_form_goals_against',
    'away_form_win_rate', 'away_form5_pts',
    'away_win_rate_away', 'away_win_rate_neutral',
    'away_avg_opp_points', 'away_sos_goals_for',
    'away_sos_goals_against', 'away_sos_form',
    'away_fm23_attack_strength', 'away_fm23_best_attacker', 'away_fm23_top3_attack',
    'away_fm23_defense_strength', 'away_fm23_best_defender', 'away_fm23_top5_defense',
    'away_fm23_overall', 'away_fm23_best_overall', 'away_fm23_top11_overall',
    'away_fm23_depth_overall', 'away_fm23_std_overall', 'away_fm23_gk_strength',

    'diff_fifa_points', 'diff_goals_for', 'diff_goals_against',
    'diff_win_rate', 'diff_form_win_rate', 'diff_form5',
    'diff_sos_goals', 'diff_sos_form', 'diff_avg_opp',
    'fm23_attack_diff', 'fm23_defense_diff', 'fm23_overall_diff',
    'fm23_best_overall_diff', 'fm23_top3_attack_diff', 'fm23_top5_defense_diff',

    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
    'h2h_goals_avg', 'h2h_n',

    'is_neutral', 'tournament_weight',
]

def train_models():
    print("=== CARREGANDO FEATURES ===")
    df = pd.read_csv('data/processed/match_features_v2.csv', parse_dates=['date'])
    df = df.dropna(subset=FEATURE_COLS + ['result', 'total_goals', 'elo_diff_abs'])

    print(f"Partidas disponíveis: {len(df)}")
    print(f"Período: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nDistribuição:")
    print(df['result'].value_counts())

    X = df[FEATURE_COLS].values
    sample_weights = df['sample_weight'].values
    elo_diff_abs = df['elo_diff_abs'].values

    # ── MODELO A — Resultado ──────────────────────────────────────────
    print("\n=== MODELO A — RESULTADO ===")

    le = LabelEncoder()
    y_result = le.fit_transform(df['result'])

    X_train, X_test, y_train, y_test, w_train, w_test, elo_train, elo_test = train_test_split(
        X, y_result, sample_weights, elo_diff_abs,
        test_size=0.15, random_state=42, stratify=y_result
    )

    XGB_RESULT_PARAMS = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
    )

    # Pesos de classe para corrigir o desbalanceamento de empates
    class_weights = compute_sample_weight('balanced', y_train)
    combined_weights = w_train * class_weights

    # Modelo final (sem calibração global — a calibração isotonic é feita
    # por faixa de |elo_diff| abaixo, via BinnedCalibrator).
    model_result = XGBClassifier(**XGB_RESULT_PARAMS)
    model_result.fit(X_train, y_train, sample_weight=combined_weights)

    # ── CALIBRAÇÃO POR FAIXA DE |ELO_DIFF| ─────────────────────────────
    # Probabilidades out-of-fold no treino, para calibrar sem overfitting.
    print("\n=== CALIBRAÇÃO POR FAIXA DE ELO_DIFF ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(X_train), len(le.classes_)))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        fold_model = XGBClassifier(**XGB_RESULT_PARAMS)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx], sample_weight=combined_weights[tr_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])

    binned_calibrator = BinnedCalibrator(n_classes=len(le.classes_))
    binned_calibrator.fit(oof_probs, y_train, elo_train)

    elo_train_bins = np.array([get_elo_bin(d) for d in elo_train])
    for b, label in enumerate(['<100 (equilibrado)', '100-300 (favorito claro)', '>=300 (disparidade extrema)']):
        print(f"  Faixa {b} [{label}]: {(elo_train_bins == b).sum()} partidas de treino")

    # ── AVALIAÇÃO (probabilidades calibradas por faixa) ────────────────
    raw_probs_test = model_result.predict_proba(X_test)
    y_prob = binned_calibrator.transform(raw_probs_test, elo_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    print(f"Accuracy: {acc:.3f}")
    print(f"Log-loss: {ll:.3f}")
    print(f"AUC-ROC (macro, ovr): {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation (accuracy do XGB base, sem calibração)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(XGBClassifier(**XGB_RESULT_PARAMS), X, y_result, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── THRESHOLD TUNING — EMPATE ──────────────────────────────────────
    print("\n=== THRESHOLD TUNING — EMPATE ===")

    draw_idx = list(le.classes_).index('D')
    for threshold in [0.20, 0.25, 0.28, 0.30, 0.32]:
        y_pred_tuned = []
        for probs in y_prob:
            if probs[draw_idx] >= threshold:
                y_pred_tuned.append(draw_idx)
            else:
                y_pred_tuned.append(np.argmax(probs))
        tuned_acc = accuracy_score(y_test, y_pred_tuned)
        draw_recall = recall_score(y_test, y_pred_tuned, labels=[draw_idx], average=None)[0]
        print(f"Threshold {threshold}: acc={tuned_acc:.3f}, draw_recall={draw_recall:.3f}")

    # ── MODELO B — Total de Gols ──────────────────────────────────────
    print("\n=== MODELO B — TOTAL DE GOLS ===")

    y_goals = df['total_goals'].values

    X_train_g, X_test_g, y_train_g, y_test_g, w_train_g, _ = train_test_split(
        X, y_goals, sample_weights,
        test_size=0.15, random_state=42
    )

    xgb_goals = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    xgb_goals.fit(X_train_g, y_train_g, sample_weight=w_train_g)

    y_pred_g = xgb_goals.predict(X_test_g)
    mae = mean_absolute_error(y_test_g, y_pred_g)

    # Over/under 2.5 accuracy
    over_pred = (y_pred_g > 2.5).astype(int)
    over_true = (y_test_g > 2.5).astype(int)
    over_acc = accuracy_score(over_true, over_pred)
    over_auc = roc_auc_score(over_true, y_pred_g)

    # BTTS accuracy
    df_test_btts = df.iloc[int(len(df) * 0.85):]

    print(f"MAE gols: {mae:.3f}")
    print(f"Over/Under 2.5 accuracy: {over_acc:.3f}")
    print(f"Over/Under 2.5 AUC-ROC: {over_auc:.3f}")

    # ── MODELO C — BTTS ───────────────────────────────────────────────
    print("\n=== MODELO C — AMBAS MARCAM ===")

    y_btts = df['btts'].values

    X_train_b, X_test_b, y_train_b, y_test_b, w_train_b, _ = train_test_split(
        X, y_btts, sample_weights,
        test_size=0.15, random_state=42, stratify=y_btts
    )

    xgb_btts = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )

    model_btts = CalibratedClassifierCV(xgb_btts, cv=5, method='isotonic')
    model_btts.fit(X_train_b, y_train_b, sample_weight=w_train_b)

    btts_pred = model_btts.predict(X_test_b)
    btts_acc = accuracy_score(y_test_b, btts_pred)
    print(f"BTTS Accuracy: {btts_acc:.3f}")

    # ── SALVAR ────────────────────────────────────────────────────────
    os.makedirs('src/models/saved', exist_ok=True)

    joblib.dump(model_result, 'src/models/saved/model_result_v2.pkl')
    joblib.dump(binned_calibrator, 'src/models/saved/model_result_calibrator_v2.pkl')
    joblib.dump(xgb_goals, 'src/models/saved/model_goals_v2.pkl')
    joblib.dump(model_btts, 'src/models/saved/model_btts_v2.pkl')
    joblib.dump(le, 'src/models/saved/label_encoder_v2.pkl')

    with open('src/models/saved/feature_columns_v2.json', 'w') as f:
        json.dump(FEATURE_COLS, f)

    print("\n=== MODELOS SALVOS ===")
    print("src/models/saved/model_result_v2.pkl")
    print("src/models/saved/model_result_calibrator_v2.pkl")
    print("src/models/saved/model_goals_v2.pkl")
    print("src/models/saved/model_btts_v2.pkl")

if __name__ == '__main__':
    train_models()
