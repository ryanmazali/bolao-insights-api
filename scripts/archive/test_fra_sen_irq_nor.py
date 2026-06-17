"""Teste manual do endpoint POST /predict/player-metrics (servidor local) para
duas partidas:

  PARTIDA 1: França (Camada A - StatsBomb Euro 2024 + FBref completo)
             vs Senegal (Camada B - StatsBomb AFCON + FBref parcial)
  PARTIDA 2: Iraque (Camada C - sem StatsBomb, fifa_rank_estimate, fm23/fm23_median)
             vs Noruega (sem StatsBomb, fifa_rank_estimate)

Para cada partida imprime: metadata de times, top 5 jogadores por
shots_expected de cada lado, soma de shots_expected por time (comparada com
team_shots_expected = shots_p90 * defensive_factor_adversário) e quaisquer
warnings de cap/valores suspeitos.

Não modifica nenhum artefato — apenas lê data/models/ e faz requests HTTP
contra um servidor uvicorn já em execução (api.main:app).
"""

import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8123"

MODELS_DIR = Path("data/models")
with open(MODELS_DIR / "player_metrics_data.json", encoding="utf-8") as f:
    player_metrics_data = json.load(f)

with open(MODELS_DIR / "team_metrics_data.json", encoding="utf-8") as f:
    team_metrics_data = json.load(f)

PLAYERS_BY_TEAM_NAME = {(p["team"], p["name"]): pid for pid, p in player_metrics_data.items()}

STARTING_XI = {
    "França": [
        "Mike Maignan", "Theo Hernandez", "Dayot Upamecano", "William Saliba", "Jules Koundé",
        "Aurélien Tchouaméni", "Adrien Rabiot", "Warren Zaïre-Emery",
        "Ousmane Dembélé", "Kylian Mbappé", "Marcus Thuram",
    ],
    "Senegal": [
        "Édouard Mendy", "Krépin Diatta", "Kalidou Koulibaly", "Abdoulaye Seck", "Ismail Jakobs",
        "Idrissa Gana Gueye", "Pape Matar Sarr", "Pape Gueye",
        "Ismaila Sarr", "Nicolas Jackson", "Sadio Mané",
    ],
    "Iraque": [
        "Jalal Hassan", "Frans Putros", "Ahmed Yahya", "Mirkhas Doski", "Rebin Sulaka",
        "Amir Al-Ammari", "Zidane Iqbal", "Ali Jassim", "Aimar Sher",
        "Aymen Hussein", "Ali Al-Hamadi",
    ],
    "Noruega": [
        "Orjan Nyland", "Julian Ryerson", "Kristoffer Ajer", "Leo Ostigard", "Fredrik Bjorkan",
        "Martin Odegaard", "Sander Berge", "Morten Thorsby",
        "Erling Haaland", "Alexander Sorloth", "Antonio Nusa",
    ],
}


def xi_ids(team_pt: str) -> list[str]:
    return [PLAYERS_BY_TEAM_NAME[(team_pt, name)] for name in STARTING_XI[team_pt]]


def print_team_metadata(team_en: str, opponent_en: str) -> None:
    data = team_metrics_data.get(team_en, {})
    opp_data = team_metrics_data.get(opponent_en, {})
    defensive_factor = opp_data.get("defensive_factor", 1.0)

    source = data.get("shots_p90_source", "?")
    print(f"  {team_en}:")
    print(f"    shots_p90            = {data.get('shots_p90')}")
    print(f"    shots_p90_source     = {source}")
    if source == "statsbomb":
        print(f"    shots_p90_raw        = {data.get('shots_p90_raw')}")
        print(f"    shots_p90_weighted   = {data.get('shots_p90_weighted')}")
        print(f"    avg_opponent_rank    = {data.get('avg_opponent_rank')}")
        print(f"    regression_factor    = {data.get('regression_factor')}")
        print(f"    defensive_factor (próprio, como defesa) = {data.get('defensive_factor')}")
    else:
        print(f"    fifa_ranking         = {data.get('fifa_ranking')}")
        print("    regression_factor    = n/a (sem dados StatsBomb)")
        print("    defensive_factor (próprio, como defesa) = n/a (default 1.0)")
    print(f"    -> defensive_factor do ADVERSÁRIO ({opponent_en}) aplicado a este time = {defensive_factor}"
          + (" (default, sem StatsBomb)" if "defensive_factor" not in opp_data else ""))


def run_match(label, home_pt, home_en, away_pt, away_en, stage="group"):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    print("\n-- METADATA --")
    print_team_metadata(home_en, away_en)
    print_team_metadata(away_en, home_en)

    payload = {
        "home_team": home_en,
        "away_team": away_en,
        "home_players": xi_ids(home_pt),
        "away_players": xi_ids(away_pt),
        "stage": stage,
    }
    resp = requests.post(f"{BASE_URL}/predict/player-metrics", json=payload)
    print(f"\nstatus_code={resp.status_code}")
    body = resp.json()

    print(f"\nresponse.metadata = {json.dumps(body['metadata'], ensure_ascii=False)}")

    warnings = []

    for side_label, team_en, opp_en, players in (
        (f"{home_en} (home)", home_en, away_en, body["home_players"]),
        (f"{away_en} (away)", away_en, home_en, body["away_players"]),
    ):
        team_data = team_metrics_data.get(team_en, {})
        opp_data = team_metrics_data.get(opp_en, {})
        defensive_factor = opp_data.get("defensive_factor", 1.0)
        team_shots_p90 = team_data.get("shots_p90", body["metadata"]["global_avg_shots_p90"])
        team_shots_expected = team_shots_p90 * (1 + (defensive_factor - 1) * 0.3)

        print(f"\n-- {side_label}: Top 5 jogadores por shots_expected --")
        sorted_players = sorted(players, key=lambda p: p["shots_expected"], reverse=True)
        print(f"  {'nome':<24} {'pos':<4} {'Sh_p90':>7} {'share%':>7} "
              f"{'shots_exp':>10} {'xG_exp':>8} {'source':<14} {'confidence':<10}")
        for p in sorted_players[:5]:
            pdata = player_metrics_data.get(p["player_id"], {})
            sh_p90 = pdata.get("Sh_p90")
            share_pct = (p["shots_expected"] / team_shots_expected * 100) if team_shots_expected > 0 else 0
            sh_p90_str = f"{sh_p90:.3f}" if sh_p90 is not None else "n/a"
            print(f"  {p['name']:<24} {p['pos']:<4} {sh_p90_str:>7} {share_pct:>6.1f}% "
                  f"{p['shots_expected']:>10.3f} {p['xg_expected']:>8.3f} {p['source']:<14} {p['confidence']:<10}")

            if share_pct > 35.0 + 1e-6:
                warnings.append(f"  [CAP?] {side_label}: {p['name']} share={share_pct:.1f}% > 35% após resposta")

            if p["pos"] == "GK" and p["shots_expected"] > 0.1:
                warnings.append(f"  [SUSPEITO] {side_label}: GK {p['name']} com shots_expected={p['shots_expected']:.3f}")

        total_shots_exp = sum(p["shots_expected"] for p in players)
        print(f"\n  Soma shots_expected ({team_en}) = {total_shots_exp:.3f}")
        print(f"  team_shots_expected esperado    = shots_p90({team_shots_p90:.4f}) * "
              f"(1 + (defensive_factor_adv({defensive_factor:.4f}) - 1) * 0.3) = {team_shots_expected:.3f}")
        diff = total_shots_exp - team_shots_expected
        print(f"  diferença = {diff:+.3f}")
        if abs(diff) > 0.05:
            warnings.append(f"  [DIFF] {side_label}: soma shots_expected ({total_shots_exp:.3f}) "
                             f"diverge de team_shots_expected ({team_shots_expected:.3f}) por {diff:+.3f}")

    print("\n-- WARNINGS --")
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  (nenhum)")


run_match(
    "PARTIDA 1: França (StatsBomb Euro2024+FBref completo) vs Senegal (StatsBomb AFCON+FBref parcial)",
    "França", "France", "Senegal", "Senegal",
)

run_match(
    "PARTIDA 2: Iraque (sem StatsBomb, fifa_rank_estimate, fm23/fm23_median) vs Noruega (sem StatsBomb, fifa_rank_estimate)",
    "Iraque", "Iraq", "Noruega", "Norway",
)
