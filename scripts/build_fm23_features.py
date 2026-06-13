"""Extrai atributos FM23 para cada jogador do Supabase.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python scripts/build_fm23_features.py

Para cada jogador em data/processed/fm23_player_mapping.csv:
  - status == matched/proxy: extrai os atributos FM23 (via fm23_uid) do
    CSV data/raw/merged_players (1).csv.
  - status == unmatched: usa a mediana dos atributos dos jogadores
    matched/proxy da mesma seleção (supabase_team) e mesma posição
    (Position do Supabase). Faz fallback para mediana por posição
    (todas as seleções) e depois mediana global se necessário.

Jogadores com supabase_name == 'Gol Contra' são ignorados.

Salva o resultado em data/processed/fm23_player_attributes.csv com as
colunas: supabase_id, supabase_name, supabase_team + os 14 atributos
FM23 + source (matched/proxy/median).
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

FM23_PATH = Path("data/raw/merged_players (1).csv")
MAPPING_PATH = Path("data/processed/fm23_player_mapping.csv")
OUTPUT_PATH = Path("data/processed/fm23_player_attributes.csv")

FM23_ATTRS = [
    "Fin", "OtB", "Com", "Dec", "Pac", "Acc", "Hea", "Pen",
    "Dri", "Str", "Vis", "Ant", "Fla", "Lon",
]

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


def fetch_supabase_positions() -> pd.DataFrame:
    """Busca id + position de todos os jogadores, paginando de 1000 em 1000."""
    players = []
    page_size = 1000
    start = 0

    while True:
        response = (
            supabase.table("players")
            .select("id, position")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        players.extend(batch)

        if len(batch) < page_size:
            break
        start += page_size

    df = pd.DataFrame(players)
    df = df.rename(columns={"id": "supabase_id", "position": "position"})
    return df


def median_with_fallback(matched_df: pd.DataFrame,
                          team: str,
                          position: str) -> pd.Series:
    """Mediana dos atributos para (team, position); com fallbacks."""
    subset = matched_df[
        (matched_df["supabase_team"] == team) & (matched_df["position"] == position)
    ]
    if len(subset) > 0:
        return subset[FM23_ATTRS].median()

    subset = matched_df[matched_df["position"] == position]
    if len(subset) > 0:
        return subset[FM23_ATTRS].median()

    return matched_df[FM23_ATTRS].median()


def main():
    print("Carregando mapeamento FM23 x Supabase...")
    mapping = pd.read_csv(MAPPING_PATH)
    mapping = mapping[mapping["supabase_name"] != "Gol Contra"].copy()
    print(f"  {len(mapping)} jogadores no mapeamento (sem 'Gol Contra').")

    print("Buscando posições no Supabase...")
    positions = fetch_supabase_positions()
    mapping = mapping.merge(positions, on="supabase_id", how="left")

    print("Carregando atributos FM23...")
    fm23 = pd.read_csv(FM23_PATH, usecols=["UID"] + FM23_ATTRS)
    fm23 = fm23.rename(columns={"UID": "fm23_uid"})
    fm23 = fm23.drop_duplicates(subset="fm23_uid")

    # Jogadores com match real (matched/proxy) -> merge direto pelos atributos
    has_match = mapping["status"].isin(["matched", "proxy"])
    matched_part = mapping[has_match].merge(fm23, on="fm23_uid", how="left")
    matched_part["source"] = matched_part["status"]

    # Mediana de atributos por (seleção, posição) usando apenas jogadores com match real
    median_pool = matched_part.dropna(subset=FM23_ATTRS)

    unmatched_part = mapping[~has_match].copy()
    if len(unmatched_part) > 0:
        medians = unmatched_part.apply(
            lambda row: median_with_fallback(median_pool, row["supabase_team"], row["position"]),
            axis=1,
        )
        unmatched_part = pd.concat([unmatched_part.reset_index(drop=True), medians.reset_index(drop=True)], axis=1)
        unmatched_part["source"] = "median"

    result = pd.concat([matched_part, unmatched_part], ignore_index=True)
    result = result[["supabase_id", "supabase_name", "supabase_team"] + FM23_ATTRS + ["source"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n=== Resumo ===")
    print(result["source"].value_counts().to_string())
    print(f"\nSalvo em: {OUTPUT_PATH}")
    print(f"Total: {len(result)}")


if __name__ == "__main__":
    main()
