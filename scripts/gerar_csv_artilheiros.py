"""Gera CSV com as previsoes de artilheiros para a fase de grupos da Copa 2026.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python gerar_csv_artilheiros.py

Busca todas as partidas com stage='group' no Supabase, roda o modelo de
artilheiros (predict_scorers) para cada partida e salva o top 5 de cada
time em data/output/scorer_predictions_group_stage.csv.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.predict_scorer import predict_scorers

load_dotenv()

_ELO_PATH = Path("data/raw/eloratings.csv")
_OUTPUT_PATH = Path("data/output/scorer_predictions_group_stage.csv")

# Nomes em ingles (usados pelo modelo / eloratings.csv) -> nomes em
# portugues cadastrados na tabela `teams` do Supabase.
TEAM_NAME_PT = {
    "Algeria": "Argélia",
    "Argentina": "Argentina",
    "Australia": "Austrália",
    "Austria": "Áustria",
    "Belgium": "Bélgica",
    "Bosnia-Herzegovina": "Bósnia",
    "Brazil": "Brasil",
    "Canada": "Canadá",
    "Cape Verde": "Cabo Verde",
    "Colombia": "Colômbia",
    "Croatia": "Croácia",
    "Curacao": "Curaçao",
    "Czech Republic": "Tchéquia",
    "DR Congo": "Rep. D. Congo",
    "Ecuador": "Equador",
    "Egypt": "Egito",
    "England": "Inglaterra",
    "France": "França",
    "Germany": "Alemanha",
    "Ghana": "Gana",
    "Haiti": "Haiti",
    "Iran": "Irã",
    "Iraq": "Iraque",
    "Ivory Coast": "Costa do Marfim",
    "Japan": "Japão",
    "Jordan": "Jordânia",
    "Mexico": "México",
    "Morocco": "Marrocos",
    "Netherlands": "Holanda",
    "New Zealand": "Nova Zelândia",
    "Norway": "Noruega",
    "Panama": "Panamá",
    "Paraguay": "Paraguai",
    "Portugal": "Portugal",
    "Qatar": "Catar",
    "Saudi Arabia": "Arábia Saudita",
    "Scotland": "Escócia",
    "Senegal": "Senegal",
    "South Africa": "África do Sul",
    "South Korea": "Coreia do Sul",
    "Spain": "Espanha",
    "Sweden": "Suécia",
    "Switzerland": "Suíça",
    "Tunisia": "Tunísia",
    "Turkey": "Turquia",
    "United States": "Estados Unidos",
    "Uruguay": "Uruguai",
    "Uzbekistan": "Uzbequistão",
}

TEAM_NAME_EN = {pt: en for en, pt in TEAM_NAME_PT.items()}

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


def fetch_group_matches() -> list[dict]:
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    resp = (
        supabase.table("matches")
        .select(
            "match_date,"
            "home_team:teams!matches_home_team_id_fkey(name),"
            "away_team:teams!matches_away_team_id_fkey(name)"
        )
        .eq("stage", "group")
        .order("match_date")
        .execute()
    )
    return resp.data


def main() -> None:
    elo_df = load_elo()
    matches = fetch_group_matches()
    print(f"[OK] {len(matches)} partidas da fase de grupos encontradas no Supabase")

    rows = []
    for m in matches:
        match_date = m["match_date"][:10]
        home_pt = m["home_team"]["name"]
        away_pt = m["away_team"]["name"]
        home_en = TEAM_NAME_EN.get(home_pt, home_pt)
        away_en = TEAM_NAME_EN.get(away_pt, away_pt)

        home_elo = get_latest_elo(elo_df, home_en)
        away_elo = get_latest_elo(elo_df, away_en)

        scorers = predict_scorers(
            home_team=home_en,
            away_team=away_en,
            home_elo=home_elo,
            away_elo=away_elo,
            top_n=5,
        )

        for team_en, team_pt in ((home_en, home_pt), (away_en, away_pt)):
            for rank, player in enumerate(scorers.get(team_en, []), start=1):
                if player["player"] == "Gol Contra":
                    continue
                rows.append({
                    "match_date": match_date,
                    "home_team": home_pt,
                    "away_team": away_pt,
                    "team": team_pt,
                    "rank": rank,
                    "player_name": player["player"],
                    "position": player["position"],
                    "probability": f"{player['probability_pct']:.1f}%",
                    "has_history": player["has_history"],
                })

        print(f"  [OK] {match_date}  {home_pt} x {away_pt}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["match_date", "team", "rank"]).reset_index(drop=True)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Salvo em {_OUTPUT_PATH} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
