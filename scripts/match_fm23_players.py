"""Mapeia jogadores do Supabase para o banco de dados do Football Manager 2023.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python scripts/match_fm23_players.py

Carrega o CSV do FM23 (data/raw/merged_players (1).csv), busca todos os
jogadores cadastrados no Supabase (players join teams) e faz fuzzy matching
dos nomes (rapidfuzz, threshold >= 85%), restringindo os candidatos do FM23
pela nacionalidade (coluna "Nat" == teams.country_code).

Salva o mapeamento em data/processed/fm23_player_mapping.csv com as colunas:
supabase_id, supabase_name, supabase_team, fm23_name, fm23_uid, match_score, status.
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from supabase import create_client

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

FM23_PATH = Path("data/raw/merged_players (1).csv")
OUTPUT_PATH = Path("data/processed/fm23_player_mapping.csv")
MATCH_THRESHOLD = 85

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


def normalize_name(name: str) -> str:
    """Remove acentos, pontuação e normaliza para minúsculas."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_fm23_players() -> pd.DataFrame:
    df = pd.read_csv(FM23_PATH, usecols=["UID", "Name", "Nat"])
    df["norm_name"] = df["Name"].apply(normalize_name)
    return df


def fetch_supabase_players() -> list[dict]:
    """Busca todos os jogadores (players join teams), paginando de 1000 em 1000."""
    players = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("players")
            .select("id, name, teams!inner(name, country_code)")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        players.extend(batch)

        if len(batch) < page_size:
            break
        start += page_size

    return players


def build_fm23_index(fm23_df: pd.DataFrame) -> dict:
    """Agrupa jogadores do FM23 por nacionalidade (Nat)."""
    index = {}
    for nat, group in fm23_df.groupby("Nat"):
        index[nat] = {
            "names": group["Name"].tolist(),
            "uids": group["UID"].tolist(),
            "norm_names": group["norm_name"].tolist(),
        }
    return index


def main():
    print("Carregando CSV do FM23...")
    fm23_df = load_fm23_players()
    fm23_index = build_fm23_index(fm23_df)
    print(f"  {len(fm23_df)} jogadores carregados do FM23.")

    print("Buscando jogadores do Supabase...")
    supabase_players = fetch_supabase_players()
    print(f"  {len(supabase_players)} jogadores encontrados no Supabase.")

    rows = []
    for player in supabase_players:
        supabase_id = player["id"]
        supabase_name = player["name"]
        team = player.get("teams") or {}
        supabase_team = team.get("name", "")
        country_code = team.get("country_code")

        candidates = fm23_index.get(country_code)

        fm23_name = ""
        fm23_uid = ""
        match_score = 0.0
        status = "unmatched"

        if candidates:
            result = process.extractOne(
                normalize_name(supabase_name),
                candidates["norm_names"],
                scorer=fuzz.WRatio,
                score_cutoff=MATCH_THRESHOLD,
            )
            if result is not None:
                _, score, idx = result
                fm23_name = candidates["names"][idx]
                fm23_uid = candidates["uids"][idx]
                match_score = round(score, 2)
                status = "matched"

        rows.append({
            "supabase_id": supabase_id,
            "supabase_name": supabase_name,
            "supabase_team": supabase_team,
            "fm23_name": fm23_name,
            "fm23_uid": fm23_uid,
            "match_score": match_score,
            "status": status,
        })

    result_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    total = len(result_df)
    matched = (result_df["status"] == "matched").sum()
    unmatched = total - matched

    print("\n=== Relatorio de mapeamento FM23 x Supabase ===")
    print(f"Total de jogadores no Supabase: {total}")
    print(f"Total com match encontrado:     {matched}")
    print(f"Total sem match:                {unmatched}")
    print(f"\nMapeamento salvo em: {OUTPUT_PATH}")

    if unmatched:
        print("\nJogadores sem match (revisao manual):")
        unmatched_df = result_df[result_df["status"] == "unmatched"]
        for _, row in unmatched_df.iterrows():
            print(f"  - {row['supabase_name']} ({row['supabase_team']})")


if __name__ == "__main__":
    main()
