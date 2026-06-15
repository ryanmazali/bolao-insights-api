import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, log_loss, classification_report,
                              roc_auc_score, precision_recall_curve)
from xgboost import XGBClassifier

FEATURE_COLS = [
    'n_matches',
    'scoring_rate',
    'scoring_rate_recent10',
    'scoring_rate_recent5',
    'scoring_rate_last_12m',
    'scoring_rate_last_24m',
    'sos_scoring_rate',
    'goals_vs_elite',
    'goals_vs_strong',
    'goals_vs_mid',
    'goals_vs_weak',
    'penalty_rate',
    'is_penalty_taker',
    'avg_goals_per_game',
    'position_weight',
    'opp_elo',
    'team_elo',
    'elo_diff',
    'opp_tier_elite',
    'opp_tier_strong',
    'opp_tier_mid',
    'opp_tier_weak',
    'opp_goals_conceded_avg',
    'opp_clean_sheet_rate',
    'team_coverage',
    'is_neutral',
    # Atributos FM23 (data/processed/fm23_player_attributes.csv)
    'Fin', 'OtB', 'Com', 'Dec', 'Pac', 'Acc', 'Hea', 'Pen',
    'Dri', 'Str', 'Vis', 'Ant', 'Fla', 'Lon',
]

def train_scorer_model():
    print("=== CARREGANDO FEATURES ===")
    df = pd.read_csv('data/processed/scorer_features_v1.csv', parse_dates=['date'])
    df = df.dropna(subset=FEATURE_COLS + ['target'])

    # Remove a primeira linha do historico de cada jogador: por construcao,
    # n_matches=0 coincide com a partida do gol de estreia (target=1 em 99.8%
    # dos casos), criando uma correlacao espuria que o modelo explora.
    df = df[df['n_matches'] > 0]

    print(f"Total linhas: {len(df)}")
    print(f"Taxa de gol: {df['target'].mean():.3f}")
    print(f"Jogadores: {df['scorer'].nunique()}")
    print(f"Período: {df['date'].min().date()} -> {df['date'].max().date()}")

    X = df[FEATURE_COLS].values
    y = df['target'].values
    sample_weights = df['recency_weight'].values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

    # ── XGBoost + Calibração ──────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )

    model = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
    model.fit(X_train, y_train, sample_weight=w_train)

    # ── Métricas ──────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.30).astype(int)  # threshold 30%

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll  = log_loss(y_test, model.predict_proba(X_test))

    print(f"\n=== MÉTRICAS (threshold=0.30) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")
    print(f"Log-loss:  {ll:.3f}")
    print(classification_report(y_test, y_pred, target_names=['não marca', 'marca']))

    # Threshold tuning
    print("\n=== THRESHOLD TUNING ===")
    for t in [0.20, 0.25, 0.28, 0.30, 0.33, 0.35]:
        yp = (y_prob >= t).astype(int)
        a = accuracy_score(y_test, yp)
        from sklearn.metrics import recall_score, precision_score
        r = recall_score(y_test, yp)
        p = precision_score(y_test, yp)
        print(f"t={t:.2f} | acc={a:.3f} | precision={p:.3f} | recall={r:.3f}")

    # Feature importance
    print("\n=== TOP FEATURES ===")
    base_model = model.calibrated_classifiers_[0].estimator
    importances = base_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:<30} {imp:.4f}")

    # CV Score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"\nCV AUC-ROC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # ── Salvar ────────────────────────────────────────────────────────
    os.makedirs('src/models/saved', exist_ok=True)
    joblib.dump(model, 'src/models/saved/model_scorer_v1.pkl')

    with open('src/models/saved/scorer_feature_columns_v1.json', 'w') as f:
        json.dump(FEATURE_COLS, f)

    print("\n=== MODELOS SALVOS ===")
    print("src/models/saved/model_scorer_v1.pkl")
    print("src/models/saved/scorer_feature_columns_v1.json")

    # Teste de predição — top marcadores Brasil vs Marrocos
    print("\n=== TESTE: top marcadores Brasil ===")
    brasil_players = df[df['team'] == 'Brazil'].groupby('scorer').last()
    brasil_players = brasil_players[FEATURE_COLS].copy()

    # Simular contexto: Marrocos (Elo ~1755)
    brasil_players['opp_elo'] = 1755
    brasil_players['team_elo'] = 1900
    brasil_players['elo_diff'] = 145
    brasil_players['opp_tier_elite'] = 0
    brasil_players['opp_tier_strong'] = 1
    brasil_players['opp_tier_mid'] = 0
    brasil_players['opp_tier_weak'] = 0
    brasil_players['opp_goals_conceded_avg'] = 0.9
    brasil_players['opp_clean_sheet_rate'] = 0.45
    brasil_players['is_neutral'] = 1

    probs = model.predict_proba(brasil_players.values)[:, 1]
    top_scorers = sorted(zip(brasil_players.index, probs), key=lambda x: x[1], reverse=True)

    print("Top 10 mais prováveis de marcar (Brasil vs Marrocos):")
    for name, prob in top_scorers[:10]:
        print(f"  {name:<25} {prob*100:.1f}%")

if __name__ == '__main__':
    train_scorer_model()
