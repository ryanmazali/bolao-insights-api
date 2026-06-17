"""fix_group_standings.py — Recomputa a classificação real dos 12 grupos a
partir de matches+match_results (fonte de verdade) e corrige
group_standings no Supabase onde houver divergência.

Uso:
    python scripts/fix_group_standings.py            # mostra diff, não escreve
    python scripts/fix_group_standings.py --apply     # aplica a correção
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=str(ROOT / ".env"))

from supabase import create_client  # noqa: E402

sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


def recompute_all():
    teams = {t["id"]: t["name"] for t in sb.table("teams").select("id, name").execute().data}
    standings = sb.table("group_standings").select("*").execute().data
    team_group = {s["team_id"]: s["world_cup_group"] for s in standings}
    standings_by_team = {s["team_id"]: s for s in standings}

    matches = sb.table("matches").select("*").eq("stage", "group").execute().data
    match_results = {r["match_id"]: r for r in sb.table("match_results").select("*").execute().data}

    recomputed = {
        tid: {"pts": 0, "gf": 0, "ga": 0, "played": 0, "wins": 0, "draws": 0, "losses": 0}
        for tid in teams
    }
    finished_count = 0
    for m in matches:
        if m["status"] != "finished":
            continue
        r = match_results.get(m["id"])
        if r is None:
            print(f"  [AVISO] match {m['id']} status=finished mas SEM linha em match_results")
            continue
        finished_count += 1
        h_id, a_id = m["home_team_id"], m["away_team_id"]
        sh, sa = r["home_score"], r["away_score"]
        recomputed[h_id]["gf"] += sh; recomputed[h_id]["ga"] += sa; recomputed[h_id]["played"] += 1
        recomputed[a_id]["gf"] += sa; recomputed[a_id]["ga"] += sh; recomputed[a_id]["played"] += 1
        if sh > sa:
            recomputed[h_id]["pts"] += 3; recomputed[h_id]["wins"] += 1
            recomputed[a_id]["losses"] += 1
        elif sa > sh:
            recomputed[a_id]["pts"] += 3; recomputed[a_id]["wins"] += 1
            recomputed[h_id]["losses"] += 1
        else:
            recomputed[h_id]["pts"] += 1; recomputed[h_id]["draws"] += 1
            recomputed[a_id]["pts"] += 1; recomputed[a_id]["draws"] += 1

    print(f"  {finished_count} jogos de grupo finalizados, com resultado em match_results.")
    return teams, team_group, standings_by_team, recomputed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Sobrescreve group_standings no Supabase")
    args = ap.parse_args()

    teams, team_group, standings_by_team, recomputed = recompute_all()

    print(f"\n{'='*100}")
    print(f"  COMPARAÇÃO — todos os 12 grupos, todos os times")
    print(f"{'='*100}")
    header = f"  {'Grupo':<6}{'Time':<18}{'played':>8}{'pts':>6} | {'sup.played':>11}{'sup.pts':>9}{'sup.gf':>8}{'sup.ga':>8} | status"
    print(header)
    print("  " + "-" * (len(header) - 2))

    diffs = []
    for tid, g in sorted(team_group.items(), key=lambda kv: (kv[1], teams[kv[0]])):
        rec = recomputed[tid]
        sup = standings_by_team[tid]
        match = (rec["played"] == sup["played"] and rec["pts"] == sup["points"]
                and rec["gf"] == sup["goals_for"] and rec["ga"] == sup["goals_against"])
        status = "OK" if match else "DIVERGENTE"
        print(f"  {g:<6}{teams[tid]:<18}{rec['played']:>8}{rec['pts']:>6} | "
              f"{sup['played']:>11}{sup['points']:>9}{sup['goals_for']:>8}{sup['goals_against']:>8} | {status}")
        if not match:
            diffs.append((tid, g, teams[tid], rec, sup))

    print(f"\n  Total de times divergentes: {len(diffs)} / {len(team_group)}")
    groups_affected = sorted(set(g for _, g, *_ in diffs))
    print(f"  Grupos afetados: {groups_affected}")

    if not diffs:
        print("\n  Nenhuma divergência encontrada. group_standings já está correto.")
        return

    if not args.apply:
        print("\n  Rodando em modo dry-run (sem --apply). Nenhuma escrita feita no Supabase.")
        print("  Rode com --apply para sobrescrever os valores corretos.")
        return

    print(f"\n  Aplicando correção em {len(diffs)} linhas...")
    for tid, g, name, rec, sup in diffs:
        # goal_difference é coluna gerada (GENERATED ALWAYS) no Postgres — não pode
        # ser escrita diretamente, o banco recalcula sozinho a partir de gf/ga.
        update = {
            "played": rec["played"], "points": rec["pts"],
            "wins": rec["wins"], "draws": rec["draws"], "losses": rec["losses"],
            "goals_for": rec["gf"], "goals_against": rec["ga"],
        }
        sb.table("group_standings").update(update).eq("team_id", tid).execute()
        print(f"    Corrigido: Grupo {g} {name}: played {sup['played']}->{rec['played']}  "
              f"pts {sup['points']}->{rec['pts']}  gf {sup['goals_for']}->{rec['gf']}  "
              f"ga {sup['goals_against']}->{rec['ga']}")

    print(f"\n  {len(diffs)} linhas corrigidas em group_standings.")


if __name__ == "__main__":
    main()
