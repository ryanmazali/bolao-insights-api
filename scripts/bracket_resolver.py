"""bracket_resolver.py — Monta o Round of 32 da Copa 2026 a partir da
classificação final dos 12 grupos.

Fonte da estrutura: data/raw/round_of_32_structure.md (FIFA Tournament
Regulations, Annex C) + data/raw/third_place_combinations.csv (495
combinações oficiais de quais 3ºs colocados caem em qual jogo).

Critérios de desempate para os "8 melhores terceiros colocados"
(regulamento FIFA, em ordem de prioridade):
  1. Pontos
  2. Saldo de gols
  3. Gols marcados
  4. Team conduct score (cartões — menor é melhor)  [PENDÊNCIA: não existe
     ainda no schema do Supabase nem em group_standings; tratado como
     ausente/no-op até existir uma fonte de dados real]
  5. Posição no ranking FIFA (menor é melhor)

Uso:
    from bracket_resolver import resolve_third_place_slots, build_r32_bracket
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THIRD_PLACE_CSV = ROOT / "data/raw/third_place_combinations.csv"

THIRD_PLACE_SLOT_MATCHES = ["match74", "match77", "match79", "match80",
                            "match81", "match82", "match85", "match87"]

# Lado esquerdo fixo dos 8 jogos que dependem de um 3º colocado
# (data/raw/round_of_32_structure.md, tabela "Os 16 jogos do Round of 32")
R32_DEPENDENT_LEFT = {
    "match74": "E",
    "match77": "I",
    "match79": "A",
    "match80": "L",
    "match81": "D",
    "match82": "G",
    "match85": "B",
    "match87": "K",
}

# Os 8 jogos 100% fixos (1ºs e 2ºs colocados, independente dos 3ºs)
# (winner, "W") / (runner-up, "R")
R32_FIXED_MATCHES = {
    "match73": (("A", "R"), ("B", "R")),
    "match75": (("F", "W"), ("C", "R")),
    "match76": (("C", "W"), ("F", "R")),
    "match78": (("E", "R"), ("I", "R")),
    "match83": (("K", "R"), ("L", "R")),
    "match84": (("H", "W"), ("J", "R")),
    "match86": (("J", "W"), ("H", "R")),
    "match88": (("D", "R"), ("G", "R")),
}


def _load_combinations() -> pd.DataFrame:
    return pd.read_csv(THIRD_PLACE_CSV, dtype=str)


def resolve_third_place_slots(qualified_groups: list[str]) -> dict[str, str]:
    """Recebe os 8 grupos cujo 3º colocado se classificou entre os 8
    melhores, e retorna em qual jogo do R32 cai o 3º colocado de cada um.

    Returns: {"match74": "3X", "match77": "3Y", ...} (8 entradas)
    """
    if len(qualified_groups) != 8:
        raise ValueError(
            f"Esperado exatamente 8 grupos qualificados, recebido {len(qualified_groups)}: "
            f"{qualified_groups}"
        )
    if len(set(qualified_groups)) != 8:
        raise ValueError(f"Grupos duplicados em qualified_groups: {qualified_groups}")

    combo_key = ";".join(sorted(qualified_groups))
    df = _load_combinations()
    row = df[df["groups_combo"] == combo_key]
    if len(row) == 0:
        raise ValueError(
            f"Combinação '{combo_key}' não encontrada em "
            f"{THIRD_PLACE_CSV.name} (esperado 1 das 495 combinações oficiais)."
        )
    if len(row) > 1:
        raise ValueError(f"Combinação '{combo_key}' encontrada {len(row)}x — CSV duplicado?")

    row = row.iloc[0]
    return {slot: row[slot] for slot in THIRD_PLACE_SLOT_MATCHES}


def _rank_key(stat: dict) -> tuple:
    """Chave de ordenação (ascendente) para os 8 melhores terceiros.
    Critérios 1-3 (pts, gd, gf) sempre disponíveis; 4 (conduct_score) e
    5 (fifa_ranking) usados como fallback só se presentes nos dados."""
    pts = stat["pts"]
    gd = stat["gd"]
    gf = stat["gf"]
    conduct = stat.get("conduct_score")   # menor é melhor; None = sem dado (no-op)
    fifa_rank = stat.get("fifa_ranking")  # menor é melhor; None = sem dado (no-op)
    return (
        -pts, -gd, -gf,
        conduct if conduct is not None else 0,
        fifa_rank if fifa_rank is not None else 0,
    )


def build_r32_bracket(group_standings: dict[str, list[dict]]) -> dict:
    """Recebe a classificação final dos 12 grupos (1º a 4º já ordenados)
    e monta os 16 jogos completos do Round of 32 com times reais.

    group_standings: {"A": [{"team": ..., "pts": ..., "gd": ..., "gf": ...,
                              "conduct_score"?: ..., "fifa_ranking"?: ...},
                             <2º>, <3º>, <4º>], "B": [...], ...}  (12 grupos)

    Returns: {
        "r32_matches": {"match73": {"home": ..., "away": ...}, ...},  (16 jogos)
        "advancing_thirds": [grupos dos 8 melhores terceiros, ordenados],
        "third_place_slots": {"match74": "3E", ...},
    }
    """
    groups = sorted(group_standings.keys())
    if len(groups) != 12:
        raise ValueError(f"Esperado 12 grupos, recebido {len(groups)}: {groups}")
    for g in groups:
        if len(group_standings[g]) != 4:
            raise ValueError(f"Grupo {g} deve ter 4 times classificados, "
                             f"recebido {len(group_standings[g])}")

    def winner(g):     return group_standings[g][0]["team"]
    def runner_up(g):  return group_standings[g][1]["team"]
    def third(g):      return group_standings[g][2]["team"]

    # 8 melhores terceiros entre os 12 grupos
    thirds_by_group = {g: group_standings[g][2] for g in groups}
    ranked = sorted(thirds_by_group.items(), key=lambda kv: _rank_key(kv[1]))
    best8_groups = sorted(g for g, _ in ranked[:8])

    slot_assignment = resolve_third_place_slots(best8_groups)

    r32 = {}

    # 8 jogos fixos
    for match_id, ((g_home, side_home), (g_away, side_away)) in R32_FIXED_MATCHES.items():
        home = winner(g_home) if side_home == "W" else runner_up(g_home)
        away = winner(g_away) if side_away == "W" else runner_up(g_away)
        r32[match_id] = {"home": home, "away": away}

    # 8 jogos dependentes de 3º colocado
    for match_id, left_group in R32_DEPENDENT_LEFT.items():
        slot_value = slot_assignment[match_id]   # ex.: "3E"
        third_group = slot_value[1:]             # "E"
        r32[match_id] = {"home": winner(left_group), "away": third(third_group)}

    return {
        "r32_matches": r32,
        "advancing_thirds": best8_groups,
        "third_place_slots": slot_assignment,
    }


if __name__ == "__main__":
    print("Use como módulo: from bracket_resolver import build_r32_bracket")
