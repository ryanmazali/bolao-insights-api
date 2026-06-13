"""Agrega os atributos FM23 por jogador (data/processed/fm23_player_attributes.csv)
em métricas por seleção, para uso como features no modelo de resultados.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python scripts/build_team_fm23_features.py

Métricas calculadas por seleção (supabase_team):
  - attack_strength:  média de (Fin, OtB, Pac) dos jogadores FW e MF
  - defense_strength: média de (Mar, Tck, Pos, Str) dos jogadores DF e GK
  - overall:          média de todos os atributos FM23 de todos os jogadores
  - gk_strength:      média de (Ref, Han, TRO, Cmd, 1v1) dos jogadores GK
  - depth:            desvio padrão do overall por jogador (maior = elenco
                       mais uniforme)

Salva o resultado em data/processed/team_fm23_features.csv com as colunas:
team_name, attack_strength, defense_strength, overall, gk_strength, depth.
"""

import pandas as pd
from pathlib import Path

ATTRS_PATH = Path("data/processed/fm23_player_attributes.csv")
OUTPUT_PATH = Path("data/processed/team_fm23_features.csv")

ALL_ATTRS = [
    "Fin", "OtB", "Com", "Dec", "Pac", "Acc", "Hea", "Pen",
    "Dri", "Str", "Vis", "Ant", "Fla", "Lon",
    "Mar", "Tck", "Pos", "Ref", "Han", "TRO", "Cmd", "1v1",
]

ATTACK_ATTRS = ["Fin", "OtB", "Pac"]
DEFENSE_ATTRS = ["Mar", "Tck", "Pos", "Str"]
GK_ATTRS = ["Ref", "Han", "TRO", "Cmd", "1v1"]


def main():
    print("Carregando atributos FM23 por jogador...")
    df = pd.read_csv(ATTRS_PATH)
    print(f"  {len(df)} jogadores em {df['supabase_team'].nunique()} seleções.")

    df["overall"] = df[ALL_ATTRS].mean(axis=1)

    rows = []
    for team, group in df.groupby("supabase_team"):
        attackers = group[group["position"].isin(["FW", "MF"])]
        defenders = group[group["position"].isin(["DF", "GK"])]
        goalkeepers = group[group["position"] == "GK"]

        rows.append({
            "team_name": team,
            "attack_strength": attackers[ATTACK_ATTRS].mean(axis=1).mean(),
            "defense_strength": defenders[DEFENSE_ATTRS].mean(axis=1).mean(),
            "overall": group["overall"].mean(),
            "gk_strength": goalkeepers[GK_ATTRS].mean(axis=1).mean(),
            "depth": group["overall"].std(),
        })

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nSalvo em: {OUTPUT_PATH}")
    print(f"Total: {len(result)} seleções")
    print("\n=== Amostra ===")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
