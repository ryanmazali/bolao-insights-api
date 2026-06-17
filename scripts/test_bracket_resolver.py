"""test_bracket_resolver.py — Testa build_r32_bracket() com 3 cenários
sintéticos de classificação de grupos (a fase de grupos real ainda não
terminou — nenhum dado real é usado aqui).

Verifica:
  1. Os 8 jogos fixos (73,75,76,78,83,84,86,88) saem idênticos nos 3
     cenários, já que só dependem de 1º/2º colocado (mantidos fixos).
  2. Os 8 jogos de 3º colocado (74,77,79,80,81,82,85,87) mudam conforme
     o conjunto de "8 melhores terceiros" muda entre cenários.
  3. Nenhum time aparece duplicado entre os 16 jogos de cada cenário.

Uso:
    python scripts/test_bracket_resolver.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bracket_resolver import build_r32_bracket, R32_FIXED_MATCHES, R32_DEPENDENT_LEFT

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "data/copa2026_groups.json", encoding="utf-8") as f:
    GROUPS = json.load(f)  # {"A": [4 times], "B": [4 times], ...}

SEP = "=" * 78


def make_group_standings(third_place_points: dict[str, int]) -> dict:
    """Monta standings sintéticos para os 12 grupos. 1º/2º/4º colocados
    ficam com pts/gd/gf fixos e idênticos em todos os cenários — só o
    3º colocado de cada grupo varia (via third_place_points), para
    controlar quais 8 grupos entram no "best 8 thirds" de cada cenário."""
    standings = {}
    for g, teams in GROUPS.items():
        t1, t2, t3, t4 = teams  # ordem arbitrária e fixa nos 3 cenários
        standings[g] = [
            {"team": t1, "pts": 9, "gd": 6, "gf": 8, "fifa_ranking": 10},
            {"team": t2, "pts": 6, "gd": 2, "gf": 5, "fifa_ranking": 25},
            {"team": t3, "pts": third_place_points[g], "gd": 0, "gf": 3, "fifa_ranking": 40},
            {"team": t4, "pts": 0, "gd": -8, "gf": 1, "fifa_ranking": 80},
        ]
    return standings


# ── 3 cenários: variando os pontos do 3º colocado por grupo ───────────────────
# Cenário 1: melhores 8 terceiros = A,B,C,D,E,F,G,H (I,J,K,L ficam de fora)
SCENARIO_1_PTS = {g: (4 if g in "ABCDEFGH" else 1) for g in GROUPS}

# Cenário 2: melhores 8 terceiros = E,F,G,H,I,J,K,L (A,B,C,D ficam de fora)
SCENARIO_2_PTS = {g: (4 if g in "EFGHIJKL" else 1) for g in GROUPS}

# Cenário 3: combinação mista = A,C,E,G,I,K,B,L (D,F,H,J de fora)
SCENARIO_3_GROUPS = set("ACEGIKBL")
SCENARIO_3_PTS = {g: (4 if g in SCENARIO_3_GROUPS else 1) for g in GROUPS}

SCENARIOS = [
    ("Cenário 1 — thirds A-H qualificam", SCENARIO_1_PTS),
    ("Cenário 2 — thirds E-L qualificam", SCENARIO_2_PTS),
    ("Cenário 3 — combinação mista", SCENARIO_3_PTS),
]


def print_bracket(label: str, result: dict) -> None:
    print(f"\n{SEP}\n  {label}\n{SEP}")
    print(f"  8 melhores terceiros: {result['advancing_thirds']}")
    print(f"  Slots de terceiro:    {result['third_place_slots']}")
    print(f"\n  {'Jogo':<10}{'Casa':<22}{'Fora':<22}{'Tipo':<10}")
    print("  " + "-" * 62)
    for match_id in sorted(result["r32_matches"], key=lambda m: int(m.replace("match", ""))):
        m = result["r32_matches"][match_id]
        tipo = "fixo" if match_id in R32_FIXED_MATCHES else "3º colocado"
        print(f"  {match_id:<10}{m['home']:<22}{m['away']:<22}{tipo:<10}")


def main():
    results = []
    for label, third_pts in SCENARIOS:
        standings = make_group_standings(third_pts)
        result = build_r32_bracket(standings)
        results.append(result)
        print_bracket(label, result)

    # ── Verificação 1: os 8 jogos fixos são idênticos nos 3 cenários ──────
    print(f"\n{SEP}\n  VERIFICAÇÃO 1 — jogos fixos idênticos entre cenários\n{SEP}")
    fixed_ids = list(R32_FIXED_MATCHES.keys())
    all_fixed_equal = True
    for match_id in fixed_ids:
        values = [r["r32_matches"][match_id] for r in results]
        equal = all(v == values[0] for v in values)
        all_fixed_equal &= equal
        status = "OK" if equal else "FALHOU"
        print(f"  {match_id}: {values[0]['home']} x {values[0]['away']}  [{status}]")
    print(f"\n  Resultado: {'PASSOU' if all_fixed_equal else 'FALHOU'} "
          f"— todos os 8 jogos fixos são idênticos nos 3 cenários.")

    # ── Verificação 2: os 8 jogos de 3º mudam entre cenários ───────────────
    print(f"\n{SEP}\n  VERIFICAÇÃO 2 — jogos de 3º colocado mudam entre cenários\n{SEP}")
    dependent_ids = list(R32_DEPENDENT_LEFT.keys())
    any_changed = False
    for match_id in dependent_ids:
        values = [r["r32_matches"][match_id]["away"] for r in results]
        changed = len(set(values)) > 1
        any_changed |= changed
        print(f"  {match_id}: cenário1={values[0]:<18} cenário2={values[1]:<18} "
              f"cenário3={values[2]:<18}  [{'MUDOU' if changed else 'igual'}]")
    print(f"\n  Resultado: {'PASSOU' if any_changed else 'FALHOU'} "
          f"— pelo menos um jogo de 3º colocado muda conforme o cenário.")

    # ── Verificação 3: nenhum time duplicado em cada cenário ───────────────
    print(f"\n{SEP}\n  VERIFICAÇÃO 3 — nenhum time duplicado entre os 16 jogos\n{SEP}")
    all_no_dup = True
    for (label, _), result in zip(SCENARIOS, results):
        teams_used = []
        for m in result["r32_matches"].values():
            teams_used.append(m["home"])
            teams_used.append(m["away"])
        dups = [t for t in set(teams_used) if teams_used.count(t) > 1]
        ok = len(dups) == 0 and len(teams_used) == 32
        all_no_dup &= ok
        print(f"  {label}: {len(teams_used)} times nos 16 jogos, "
              f"{len(set(teams_used))} únicos, duplicados={dups}  [{'OK' if ok else 'FALHOU'}]")
    print(f"\n  Resultado: {'PASSOU' if all_no_dup else 'FALHOU'} "
          f"— nenhum time duplicado em nenhum cenário.")

    print(f"\n{SEP}")
    geral = all_fixed_equal and any_changed and all_no_dup
    print(f"  RESULTADO GERAL: {'TODOS OS TESTES PASSARAM' if geral else 'FALHAS ENCONTRADAS'}")
    print(SEP)


if __name__ == "__main__":
    main()
