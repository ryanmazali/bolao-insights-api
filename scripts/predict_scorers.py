"""Gera os marcadores mais prováveis de cada seleção em uma partida específica,
a partir do terminal.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python scripts/predict_scorers.py "Brazil" "Argentina"

Se os nomes das seleções não forem passados como argumentos, o script
pergunta interativamente. Os nomes devem estar em inglês, no mesmo formato
usado em data/raw/results.csv (ex: "South Korea", "Czech Republic", "USA").
"""

import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict_scorer import predict_scorers

# Nomes da Copa 2026 que aparecem com grafia diferente em data/raw/eloratings.csv
ELO_NAME_ALIASES = {
    "Czech Republic": "Czechia",
    "Curacao": "Curaçao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "Democratic Republic of Congo",
}


def get_team_elo(elo_df: pd.DataFrame, team: str) -> float:
    subset = elo_df[elo_df["team"] == team]
    if len(subset) == 0:
        alias = ELO_NAME_ALIASES.get(team)
        if alias:
            subset = elo_df[elo_df["team"] == alias]
    return float(subset.iloc[-1]["rating"]) if len(subset) > 0 else 1500.0


def print_team_scorers(team: str, scorers: list[dict]) -> None:
    print(f"\n=== {team} ===")
    if not scorers:
        print("  Nenhum convocado encontrado no Supabase.")
        return

    for i, s in enumerate(scorers, start=1):
        marker = "" if s["has_history"] else " (sem histórico)"
        print(f"  {i}. {s['player']} ({s['position']}) — {s['probability_pct']}%{marker}")


def main():
    if len(sys.argv) >= 3:
        home_team = sys.argv[1]
        away_team = sys.argv[2]
    else:
        home_team = input("Seleção da casa: ").strip()
        away_team = input("Seleção visitante: ").strip()

    try:
        elo_df = pd.read_csv("data/raw/eloratings.csv")
        elo_df["date"] = pd.to_datetime(elo_df["date"], format="mixed", dayfirst=False)
        elo_df = elo_df.sort_values("date")
        home_elo = get_team_elo(elo_df, home_team)
        away_elo = get_team_elo(elo_df, away_team)
    except Exception:
        home_elo = away_elo = 1600.0

    results = predict_scorers(
        home_team=home_team,
        away_team=away_team,
        home_elo=home_elo,
        away_elo=away_elo,
    )

    print(f"\n=== {home_team} x {away_team} ===")
    print(f"Elo: {home_team} {home_elo:.0f} x {away_elo:.0f} {away_team}")

    print_team_scorers(home_team, results.get(home_team, []))
    print_team_scorers(away_team, results.get(away_team, []))


if __name__ == "__main__":
    main()
