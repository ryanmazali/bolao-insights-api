import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

SAVE_DIR = "src/models/saved"

# Features usadas no treino — diferenciais + absolutas de ambos os times
FEATURE_COLS = [
    # diferenciais (home - away)
    "diff_xg_for",
    "diff_xg_against",
    "diff_xg_net",
    "diff_win_rate",
    "diff_goals_for",
    "diff_shots",
    "diff_pressures",
    # home team
    "home_xg_for",
    "home_xg_against",
    "home_xg_diff",
    "home_goals_for",
    "home_goals_against",
    "home_win_rate",
    "home_shots",
    "home_sot",
    "home_pressures",
    # away team
    "away_xg_for",
    "away_xg_against",
    "away_xg_diff",
    "away_goals_for",
    "away_goals_against",
    "away_win_rate",
    "away_shots",
    "away_sot",
    "away_pressures",
]


def load_data(path: str = "data/processed/match_features.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # time-based split: 80% treino / 20% teste
    cutoff = int(len(df) * 0.8)
    train = df.iloc[:cutoff].copy()
    test  = df.iloc[cutoff:].copy()

    print(f"[train] Total: {len(df)} | Treino: {len(train)} ({train['match_date'].min().date()} -> {train['match_date'].max().date()})")
    print(f"[train] Teste:  {len(test)}  ({test['match_date'].min().date()} -> {test['match_date'].max().date()})")
    return train, test


def train_result_model(train: pd.DataFrame, test: pd.DataFrame) -> CalibratedClassifierCV:
    """Classifica resultado: 0=away win | 1=draw | 2=home win — com Platt scaling."""
    X_train = train[FEATURE_COLS]
    y_train = train["result"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["result"]

    xgb_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    # treina modelo base isolado só para log e feature importance
    base_for_log = XGBClassifier(**xgb_params)
    base_for_log.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    logloss_raw = log_loss(y_test, base_for_log.predict_proba(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_for_log, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n[resultado] CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"[resultado] Log-loss SEM calibracao: {logloss_raw:.3f}")

    fi = pd.Series(base_for_log.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("[resultado] Top 10 features:")
    print(fi.head(10).round(4).to_string())

    # Platt scaling via CV — treina 5 cópias do XGBoost e calibra sigmoid nos folds held-out
    calibrated = CalibratedClassifierCV(
        XGBClassifier(**xgb_params),
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_train, y_train)

    # avaliação no teste
    y_pred  = calibrated.predict(X_test)
    y_prob  = calibrated.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)

    print(f"\n[resultado] Log-loss COM Platt scaling: {logloss:.3f}  (delta: {logloss - logloss_raw:+.3f})")
    print(f"[resultado] Teste accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"]))

    return calibrated


def train_goals_model(train: pd.DataFrame, test: pd.DataFrame) -> XGBRegressor:
    """Regride total de gols na partida."""
    X_train = train[FEATURE_COLS]
    y_train = train["total_goals"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["total_goals"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    print(f"\n[gols] CV MAE: {-cv_mae.mean():.3f} ± {cv_mae.std():.3f}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"[gols] Teste MAE: {mae:.3f}")
    print(f"[gols] Predições: {y_pred[:8].round(2)} | Real: {y_test.values[:8]}")

    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("[gols] Top 10 features:")
    print(fi.head(10).round(4).to_string())

    return model


def save_artifacts(result_model: CalibratedClassifierCV, goals_model: XGBRegressor) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(result_model, f"{SAVE_DIR}/result_model.pkl")
    joblib.dump(goals_model,  f"{SAVE_DIR}/goals_model.pkl")
    with open(f"{SAVE_DIR}/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"\n[train] Modelos salvos em {SAVE_DIR}/")


def run():
    train, test = load_data()

    print("\n" + "="*50)
    print("MODELO 1 — Resultado (classificação 3 classes)")
    print("="*50)
    result_model = train_result_model(train, test)

    print("\n" + "="*50)
    print("MODELO 2 — Total de Gols (regressão)")
    print("="*50)
    goals_model = train_goals_model(train, test)

    save_artifacts(result_model, goals_model)


if __name__ == "__main__":
    run()
