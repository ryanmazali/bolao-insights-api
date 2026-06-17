"""predict_match_cli.py — CLI de predição completa para uma partida (pipeline v3).

Recebe dois times e devolve todos os mercados (1X2, over/under, BTTS,
placar exato, handicap asiático) usando os modelos XGBoost Poisson v3
+ correção Dixon-Coles + threshold de empate calibrado (models/metrics_v3.json).

A lógica de predição (carga de modelos, features, mercados) vive em
src/models/predict_v3.predict_full_match() — única fonte de verdade,
reaproveitada também por api/routes/full_match.py. Este script só
formata a saída para o terminal.

Uso:
    python scripts/predict_match_cli.py --home "Brazil" --away "Argentina"
    python scripts/predict_match_cli.py --home "Brasil" --away "Argentina" --stage final --neutral
    python scripts/predict_match_cli.py --home "Austria" --away "Jordan" --date 2026-06-17

Nomes de times: aceita inglês (convenção interna do pipeline, ex.: "USA",
"Bosnia-Herzegovina", "Curacao") ou português (convenção Supabase, ex.:
"Brasil", "Estados Unidos", "Curaçao") — ambos são normalizados antes de
buscar nos dados históricos.
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.predict_v3 import predict_full_match  # noqa: E402

SEP = "=" * 72

STAGE_CHOICES = ["group", "r32", "r16", "qf", "sf", "final"]
STAGE_LABELS = {
    "group": "Fase de grupos", "r32": "Oitavas (R32)", "r16": "Oitavas (R16)",
    "qf": "Quartas de final", "sf": "Semifinal", "final": "Final",
}


def section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print("─" * 72)


def fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_result(result: dict, stage: str, tournament: str, is_neutral: bool, match_date) -> None:
    home, away = result["home_team"], result["away_team"]

    print(SEP)
    print(f"  {home} vs {away}  —  {match_date.date()}")
    print(f"  Torneio: {tournament}  |  Neutro: {'Sim' if is_neutral else 'Não'}  |  "
          f"Fase: {STAGE_LABELS[stage]}")
    print(SEP)

    r = result["result_1x2"]
    label = {"home_win": f"Vitória {home}", "draw": "Empate", "away_win": f"Vitória {away}"}[r["predicted"]]
    print(f"\n  λ_{home} = {result['lambda_home']:.3f}   λ_{away} = {result['lambda_away']:.3f}   "
          f"threshold_empate = {r['threshold_used']}")

    section("RESULTADO (1X2)")
    print(f"  P({home} vence)  : {fmt_pct(r['p_home'])}")
    print(f"  P(Empate)        : {fmt_pct(r['p_draw'])}")
    print(f"  P({away} vence)  : {fmt_pct(r['p_away'])}")
    print(f"\n  Previsão (threshold={r['threshold_used']}): {label}")

    eg = result["expected_goals"]
    section("GOLS ESPERADOS")
    print(f"  {home:<28} : {eg['home']:.3f}")
    print(f"  {away:<28} : {eg['away']:.3f}")
    print(f"  {'Total':<28} : {eg['total']:.3f}")

    section("OVER / UNDER (total de gols)")
    print(f"  {'Linha':>6}  {'Over':>10}  {'Under':>10}")
    for ou in result["over_under"]:
        print(f"  {ou['line']:>6.1f}  {fmt_pct(ou['p_over']):>10}  {fmt_pct(ou['p_under']):>10}")

    section("AMBAS MARCAM (BTTS)")
    print(f"  Sim : {fmt_pct(result['btts']['p_yes'])}")
    print(f"  Não : {fmt_pct(result['btts']['p_no'])}")

    section("PLACAR EXATO — top 5 mais prováveis")
    for s in result["exact_score_top5"]:
        print(f"  {s['score']}   {fmt_pct(s['probability'])}")

    section(f"HANDICAP ASIÁTICO (a favor de {home})")
    print(f"  {'Linha':>6}  {'Casa cobre':>12}  {'Push':>10}  {'Fora cobre':>12}")
    for h in result["asian_handicap"]:
        push_str = fmt_pct(h["push"]) if h["push"] is not None else "      —"
        print(f"  {h['line']:>+6.1f}  {fmt_pct(h['home_covers']):>12}  {push_str:>10}  "
              f"{fmt_pct(h['away_covers']):>12}")

    section("CONTEXTO E CONFIANÇA")
    c = result["confidence"]
    for label_, team, sw, matched in [(home, home, c["home_sample_weight"], c["home_matched"]),
                                      (away, away, c["away_sample_weight"], c["away_matched"])]:
        if sw is None:
            print(f"  {label_:<20}: NÃO encontrado em eafc26_team_features.csv — usando média global  [FALLBACK]")
        else:
            flag = "OK" if sw >= 0.7 else "BAIXA CONFIANÇA"
            print(f"  {label_:<20}: {matched} jogadores matchados no EA FC 26  "
                  f"(sample_weight={sw:.2f})  [{flag}]")
    print()
    h2h = result["h2h"]
    if h2h["has_data"]:
        print(f"  H2H: {h2h['n']} confronto(s) anterior(es) — "
              f"{home} venceu {h2h['home_wins']*100:.0f}%, "
              f"empate {h2h['draws']*100:.0f}%, "
              f"{away} venceu {h2h['away_wins']*100:.0f}%")
    else:
        print(f"  H2H: sem confrontos anteriores no histórico — usando prior default "
              f"(33%/33%/33%, 2.5 gols)")

    if stage != "group":
        print(f"\n  [!] --stage={stage} é apenas informativo: o modelo de gols v3 não foi "
              f"treinado com uma feature de fase de torneio (stage_factor só existe hoje no "
              f"endpoint de projeção tática de jogadores, api/routes/player_metrics.py). "
              f"Não afeta λ_home/λ_away aqui.")

    print(f"\n{SEP}")


def parse_args():
    p = argparse.ArgumentParser(description="Predição completa de uma partida (pipeline v3).")
    p.add_argument("--home", required=True, help="Time da casa (inglês ou português)")
    p.add_argument("--away", required=True, help="Time de fora (inglês ou português)")
    p.add_argument("--neutral", action=argparse.BooleanOptionalAction, default=True,
                   help="Jogo em campo neutro (default: True, padrão Copa 2026)")
    p.add_argument("--stage", choices=STAGE_CHOICES, default="group",
                   help="Fase do torneio (apenas informativo — ver nota no output)")
    p.add_argument("--date", default=None,
                   help="Data da partida (YYYY-MM-DD). Default: hoje.")
    p.add_argument("--tournament", default="FIFA World Cup",
                   help='Torneio para tournament_weight (default: "FIFA World Cup")')
    return p.parse_args()


def main():
    args = parse_args()
    match_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    result = predict_full_match(args.home, args.away, is_neutral=args.neutral, match_date=match_date)
    print_result(result, args.stage, args.tournament, args.neutral, match_date)


if __name__ == "__main__":
    main()
