"""Consulta os marcadores mais prováveis das partidas da Copa 2026.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python consultar_marcadores.py                  # roda todas as partidas dos grupos
    python consultar_marcadores.py Brazil Morocco   # roda só uma partida específica
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.predict_scorer import predict_scorers

_ELO_PATH = Path("data/raw/eloratings.csv")
_GROUPS_PATH = Path("data/copa2026_groups.json")

# Nomes da Copa 2026 que aparecem com grafia diferente em data/raw/eloratings.csv
_ELO_NAME_ALIASES = {
    "Czech Republic": "Czechia",
    "Curacao": "Curaçao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "Democratic Republic of Congo",
}


def load_elo() -> pd.DataFrame:
    df = pd.read_csv(_ELO_PATH)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    return df.sort_values("date")


def get_latest_elo(elo_df: pd.DataFrame, team: str) -> float:
    subset = elo_df[elo_df["team"] == team]
    if len(subset) == 0:
        alias = _ELO_NAME_ALIASES.get(team)
        if alias:
            subset = elo_df[elo_df["team"] == alias]
    return float(subset.iloc[-1]["rating"]) if len(subset) > 0 else 1500.0


def show_match(elo_df: pd.DataFrame, home: str, away: str) -> None:
    home_elo = get_latest_elo(elo_df, home)
    away_elo = get_latest_elo(elo_df, away)

    scorers = predict_scorers(
        home_team=home,
        away_team=away,
        home_elo=home_elo,
        away_elo=away_elo,
    )

    print(f"\n=== {home} (Elo {home_elo:.0f}) x {away} (Elo {away_elo:.0f}) ===")
    for team in (home, away):
        print(f"\n{team}:")
        for player in scorers.get(team, []):
            hist = "" if player["has_history"] else "  (sem histórico - estimativa)"
            print(
                f"  {player['player']:<25} {player['probability_pct']:>5.1f}%  "
                f"[{player['position']}]{hist}"
            )


def main() -> None:
    elo_df = load_elo()

    if len(sys.argv) == 3:
        show_match(elo_df, sys.argv[1], sys.argv[2])
        return

    with open(_GROUPS_PATH, encoding="utf-8") as f:
        groups = json.load(f)

    for group_name, teams in groups.items():
        print(f"\n{'#' * 50}\nGrupo {group_name}\n{'#' * 50}")
        for home, away in combinations(teams, 2):
            show_match(elo_df, home, away)


if __name__ == "__main__":
    main()
