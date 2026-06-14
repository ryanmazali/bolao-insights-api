"""Agrega os atributos FM23 por jogador (data/processed/fm23_player_attributes.csv)
em métricas por seleção, para uso como features no modelo de resultados.

Uso (rodar a partir da raiz do projeto, com o venv ativado):

    python scripts/build_team_fm23_features.py

Métricas calculadas por seleção (supabase_team):
  Ataque (jogadores FW + MF):
    - attack_strength:  média de (Fin, OtB, Pac) de todos os atacantes
    - best_attacker:    maior (Fin+OtB+Pac)/3 entre os atacantes
    - top3_attack:      média das 3 maiores pontuações de ataque

  Defesa (jogadores DF + GK):
    - defense_strength: média de (Mar, Tck, Pos, Str) de todos os defensores
    - best_defender:    maior (Mar+Tck+Pos)/3 entre os defensores
    - top5_defense:     média das 5 maiores pontuações de defesa

  Goleiro (jogadores GK):
    - gk_strength:      média de (Ref, Han, TRO, Cmd, 1v1) dos goleiros

  Elenco geral (todos os jogadores, por "overall" = média de ALL_ATTRS):
    - overall:          média do overall de todo o elenco
    - best_overall:     maior overall do elenco
    - top11_overall:    média do overall dos 11 melhores jogadores
    - depth_overall:    média do overall dos jogadores 12º-23º (banco)
    - std_overall:      desvio padrão do overall (uniformidade do elenco)

Salva o resultado em data/processed/team_fm23_features.csv.
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
BEST_DEFENDER_ATTRS = ["Mar", "Tck", "Pos"]
GK_ATTRS = ["Ref", "Han", "TRO", "Cmd", "1v1"]

FM23_METRICS = [
    "attack_strength", "best_attacker", "top3_attack",
    "defense_strength", "best_defender", "top5_defense",
    "gk_strength",
    "overall", "best_overall", "top11_overall", "depth_overall", "std_overall",
]


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

        attack_scores = attackers[ATTACK_ATTRS].mean(axis=1)
        defense_scores = defenders[BEST_DEFENDER_ATTRS].mean(axis=1)
        overall_sorted = group["overall"].sort_values(ascending=False)

        rows.append({
            "team_name": team,

            # Ataque
            "attack_strength": attack_scores.mean(),
            "best_attacker": attack_scores.max(),
            "top3_attack": attack_scores.nlargest(3).mean(),

            # Defesa
            "defense_strength": defenders[DEFENSE_ATTRS].mean(axis=1).mean(),
            "best_defender": defense_scores.max(),
            "top5_defense": defense_scores.nlargest(5).mean(),

            # Goleiro
            "gk_strength": goalkeepers[GK_ATTRS].mean(axis=1).mean(),

            # Elenco geral
            "overall": group["overall"].mean(),
            "best_overall": overall_sorted.max(),
            "top11_overall": overall_sorted.head(11).mean(),
            "depth_overall": overall_sorted.iloc[11:23].mean(),
            "std_overall": group["overall"].std(),
        })

    result = pd.DataFrame(rows)

    # Preenche eventuais NaN (times com elenco menor que o esperado para
    # alguma métrica) com a média global da métrica.
    result[FM23_METRICS] = result[FM23_METRICS].fillna(result[FM23_METRICS].mean())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nSalvo em: {OUTPUT_PATH}")
    print(f"Total: {len(result)} seleções")
    print("\n=== Amostra ===")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
