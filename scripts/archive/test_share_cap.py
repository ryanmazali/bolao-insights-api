"""Testa o teto de share de chutes (35% / _normalize_shares_with_cap) em
api/routes/player_metrics.py, usando Bélgica vs Egito.

Imprime o share de chutes (antes/depois do teto) de cada titular do
Egito e o shots_expected de Salah e Marmoush antes/depois da correção.

Não modifica nenhum artefato — apenas lê data/models/ e chama as funções
internas do router.
"""

import json
from pathlib import Path

from api.routes import player_metrics as pm

pm.load_player_metrics()

with open(Path("data/models/player_metrics_data.json"), encoding="utf-8") as f:
    player_metrics_data = json.load(f)

PLAYERS_BY_TEAM_NAME = {(v["team"], v["name"]): pid for pid, v in player_metrics_data.items()}

EGITO_XI = [
    "Mohamed El Shenawy", "Mohamed Hany", "Ramy Rabia", "Yasser Ibrahim", "Ahmed Fatouh",
    "Mohamed Salah", "Ahmed Sayed Zizo", "Mahmoud Trezeguet", "Emam Ashour", "Marwan Attia",
    "Omar Marmoush",
]

BELGICA_XI = [
    "Courtois", "Timothy Castagne", "Koni De Winter", "Arthur Theate", "Thomas Meunier",
    "Kevin De Bruyne", "Youri Tielemans", "Amadou Onana",
    "Charles De Ketelaere", "Romelu Lukaku", "Leandro Trossard",
]

egito_ids = [PLAYERS_BY_TEAM_NAME[("Egito", name)] for name in EGITO_XI]
belgica_ids = [PLAYERS_BY_TEAM_NAME[("Bélgica", name)] for name in BELGICA_XI]

HOME_TEAM, AWAY_TEAM = "Belgium", "Egypt"

team_data = pm._team_metrics_data.get(AWAY_TEAM)
opp_data = pm._team_metrics_data.get(HOME_TEAM)
defensive_factor = opp_data.get("defensive_factor", 1.0) if opp_data else 1.0
team_shots_p90 = team_data.get("shots_p90", pm._global_avg_shots_p90) if team_data else pm._global_avg_shots_p90
team_shots_expected = team_shots_p90 * (1 + (defensive_factor - 1) * 0.3)

# ── Shares "antes" (sem teto) — replica a lógica antiga (sh / sum_sh) ──
players_data = [pm._player_metrics_data.get(pid) for pid in egito_ids]

sum_sh = 0.0
for data in players_data:
    if data is None:
        sum_sh += pm._position_sh_p90_medians.get("_all", 0.0)
        continue
    sh = data.get("Sh_p90") or 0
    if sh > 0:
        sum_sh += sh
    else:
        pos = data.get("position")
        sum_sh += pm._position_sh_p90_medians.get(pos, pm._position_sh_p90_medians.get("_all", 0.0))

raw_shares = {
    pid: (data.get("Sh_p90") or 0) / sum_sh if sum_sh > 0 else 0
    for pid, data in zip(egito_ids, players_data)
    if data is not None
}
capped_shares = pm._normalize_shares_with_cap(raw_shares)

print("=" * 90)
print(f"Bélgica vs Egito — team_shots_expected (Egito) = {team_shots_expected:.2f}")
print("=" * 90)
print(f"\n{'jogador':<22} {'share antes':>12} {'share depois':>13} "
      f"{'shots_exp antes':>16} {'shots_exp depois':>17}")
for pid, data in zip(egito_ids, players_data):
    before = raw_shares[pid]
    after = capped_shares[pid]
    print(
        f"{data['name']:<22} {before * 100:>11.1f}% {after * 100:>12.1f}% "
        f"{before * team_shots_expected:>16.2f} {after * team_shots_expected:>17.2f}"
    )

print(f"\nSoma shares antes:  {sum(raw_shares.values()) * 100:.1f}%")
print(f"Soma shares depois: {sum(capped_shares.values()) * 100:.1f}%")

# ── Re-teste via _project_side (valores corrigidos, com teto aplicado) ──
print("\n" + "=" * 90)
print("Re-teste via _project_side (Bélgica vs Egito) — valores corrigidos")
print("=" * 90)
projections = pm._project_side(egito_ids, AWAY_TEAM, HOME_TEAM, stage_factor=1.0)
for proj in projections:
    if proj.name in ("Mohamed Salah", "Omar Marmoush"):
        print(f"\n{proj.name}:")
        print(f"  shots_expected = {proj.shots_expected}")
        print(f"  sot_expected   = {proj.sot_expected}")
        print(f"  xg_expected    = {proj.xg_expected}")
