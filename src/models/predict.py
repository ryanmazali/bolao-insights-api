import json
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

from src.features.build_features import (
    compute_h2h,
    compute_team_features,
    get_tournament_weight,
    load_fifa_rankings,
    load_results,
    normalize_name,
)

_BASE = Path("src/models/saved")

_REQUIRED_FILES = [
    "model_result_v2.pkl",
    "model_goals_v2.pkl",
    "model_btts_v2.pkl",
    "label_encoder_v2.pkl",
    "feature_columns_v2.json",
]


def load_models():
    missing = [name for name in _REQUIRED_FILES if not (_BASE / name).exists()]
    if missing:
        raise RuntimeError(
            f"Modelos v2 não encontrados em {_BASE}: {', '.join(missing)}. "
            "Rode `python -m src.models.train_v2` para gerá-los."
        )

    model_result = joblib.load(_BASE / "model_result_v2.pkl")
    model_goals = joblib.load(_BASE / "model_goals_v2.pkl")
    model_btts = joblib.load(_BASE / "model_btts_v2.pkl")
    le = joblib.load(_BASE / "label_encoder_v2.pkl")

    with open(_BASE / "feature_columns_v2.json") as f:
        feature_cols = json.load(f)

    return model_result, model_goals, model_btts, le, feature_cols


model_result, model_goals, model_btts, le, FEATURE_COLS = load_models()

_df = load_results()
_rankings_df, _current_rankings = load_fifa_rankings()

# Contexto padrão para previsões "ao vivo": Copa do Mundo, jogo em campo neutro
_DEFAULT_TOURNAMENT = 'FIFA World Cup'
_DEFAULT_IS_NEUTRAL = 1


def _build_feature_row(home_team: str, away_team: str, before_date: pd.Timestamp) -> dict:
    home = normalize_name(home_team)
    away = normalize_name(away_team)

    home_f = compute_team_features(_df, home, before_date, _rankings_df, _current_rankings)
    away_f = compute_team_features(_df, away, before_date, _rankings_df, _current_rankings)

    if home_f is None:
        raise ValueError(f"Sem histórico de partidas para o time '{home_team}'.")
    if away_f is None:
        raise ValueError(f"Sem histórico de partidas para o time '{away_team}'.")

    h2h = compute_h2h(_df, home, away, before_date)

    return {
        'home_fifa_points': home_f['fifa_points'],
        'home_goals_for': home_f['goals_for_avg'],
        'home_goals_against': home_f['goals_against_avg'],
        'home_goal_diff': home_f['goal_diff_avg'],
        'home_win_rate': home_f['win_rate'],
        'home_draw_rate': home_f['draw_rate'],
        'home_btts_rate': home_f['btts_rate'],
        'home_clean_sheet': home_f['clean_sheet_rate'],
        'home_form_goals_for': home_f['form_goals_for'],
        'home_form_goals_against': home_f['form_goals_against'],
        'home_form_win_rate': home_f['form_win_rate'],
        'home_form5_pts': home_f['form5_pts'],
        'home_win_rate_home': home_f['win_rate_home'],
        'home_win_rate_neutral': home_f['win_rate_neutral'],
        'home_avg_opp_points': home_f['avg_opp_points'],
        'home_sos_goals_for': home_f['sos_goals_for'],
        'home_sos_goals_against': home_f['sos_goals_against'],
        'home_sos_form': home_f['sos_form_goals'],

        'away_fifa_points': away_f['fifa_points'],
        'away_goals_for': away_f['goals_for_avg'],
        'away_goals_against': away_f['goals_against_avg'],
        'away_goal_diff': away_f['goal_diff_avg'],
        'away_win_rate': away_f['win_rate'],
        'away_draw_rate': away_f['draw_rate'],
        'away_btts_rate': away_f['btts_rate'],
        'away_clean_sheet': away_f['clean_sheet_rate'],
        'away_form_goals_for': away_f['form_goals_for'],
        'away_form_goals_against': away_f['form_goals_against'],
        'away_form_win_rate': away_f['form_win_rate'],
        'away_form5_pts': away_f['form5_pts'],
        'away_win_rate_away': away_f['win_rate_away'],
        'away_win_rate_neutral': away_f['win_rate_neutral'],
        'away_avg_opp_points': away_f['avg_opp_points'],
        'away_sos_goals_for': away_f['sos_goals_for'],
        'away_sos_goals_against': away_f['sos_goals_against'],
        'away_sos_form': away_f['sos_form_goals'],

        'diff_fifa_points': home_f['fifa_points'] - away_f['fifa_points'],
        'diff_goals_for': home_f['goals_for_avg'] - away_f['goals_for_avg'],
        'diff_goals_against': home_f['goals_against_avg'] - away_f['goals_against_avg'],
        'diff_win_rate': home_f['win_rate'] - away_f['win_rate'],
        'diff_form_win_rate': home_f['form_win_rate'] - away_f['form_win_rate'],
        'diff_form5': home_f['form5_pts'] - away_f['form5_pts'],
        'diff_sos_goals': home_f['sos_goals_for'] - away_f['sos_goals_for'],
        'diff_sos_form': home_f['sos_form_goals'] - away_f['sos_form_goals'],
        'diff_avg_opp': home_f['avg_opp_points'] - away_f['avg_opp_points'],

        'h2h_home_wins': h2h['h2h_home_wins'],
        'h2h_draws': h2h['h2h_draws'],
        'h2h_away_wins': h2h['h2h_away_wins'],
        'h2h_goals_avg': h2h['h2h_goals_avg'],
        'h2h_n': h2h['h2h_n'],

        'is_neutral': _DEFAULT_IS_NEUTRAL,
        'tournament_weight': get_tournament_weight(_DEFAULT_TOURNAMENT),
    }


def predict_match(home_team: str, away_team: str) -> dict:
    before_date = pd.Timestamp.now().normalize()
    row = _build_feature_row(home_team, away_team, before_date)
    X = np.array([[row[col] for col in FEATURE_COLS]])

    # Resultado
    probs = model_result.predict_proba(X)[0]
    classes = le.classes_  # ['A', 'D', 'H']

    home_win = float(probs[list(classes).index('H')])
    draw     = float(probs[list(classes).index('D')])
    away_win = float(probs[list(classes).index('A')])

    # Gols esperados
    total_goals = float(max(0.0, model_goals.predict(X)[0]))
    denom = home_win + draw + away_win
    home_goals = total_goals * (home_win + 0.5 * draw) / denom
    away_goals = total_goals - home_goals

    # BTTS
    btts_probs = model_btts.predict_proba(X)[0]
    btts_yes = float(btts_probs[1])

    # Chance dupla (derivado)
    double_chance_1x = home_win + draw
    double_chance_x2 = draw + away_win
    double_chance_12 = home_win + away_win

    # Over/under
    over_25 = 1 / (1 + np.exp(-2 * (total_goals - 2.5)))  # sigmoid suavizado
    under_25 = 1 - over_25

    return {
        "probabilities": {
            "home_win": round(home_win, 3),
            "draw": round(draw, 3),
            "away_win": round(away_win, 3),
        },
        "expected_goals": {
            "total": round(total_goals, 2),
            "home": round(home_goals, 2),
            "away": round(away_goals, 2),
        },
        "markets": {
            "btts": {
                "yes": round(btts_yes, 3),
                "no": round(1 - btts_yes, 3),
            },
            "over_under_25": {
                "over": round(over_25, 3),
                "under": round(under_25, 3),
            },
            "double_chance": {
                "1X": round(double_chance_1x, 3),
                "X2": round(double_chance_x2, 3),
                "12": round(double_chance_12, 3),
            },
        },
        "most_likely_result": "home_win" if home_win > draw and home_win > away_win
                              else "draw" if draw > away_win
                              else "away_win",
    }
