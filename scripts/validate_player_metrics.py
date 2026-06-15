"""Valida o modelo de métricas táticas por jogador antes de integrá-lo à API.

Simula, para 3 partidas de teste, a projeção de chutes/SoT/xG/desarmes/
faltas por jogador titular, usando os artefatos gerados por
scripts/build_player_metrics_model.py:
    data/models/player_metrics_data.json
    data/models/team_metrics_data.json
    data/models/metrics_metadata.json

Não modifica nenhum modelo existente — apenas lê os artefatos e imprime
um relatório de validação.
"""

import json
from pathlib import Path

import pandas as pd

pd.set_option('display.width', 200)

MODELS_DIR = Path('data/models')

with open(MODELS_DIR / 'player_metrics_data.json', encoding='utf-8') as f:
    player_metrics_data = json.load(f)

with open(MODELS_DIR / 'team_metrics_data.json', encoding='utf-8') as f:
    team_metrics_data = json.load(f)

with open(MODELS_DIR / 'metrics_metadata.json', encoding='utf-8') as f:
    metadata = json.load(f)

global_avg_shots_p90 = metadata['global_avg_shots_p90']
global_avg_xg_conceded_p90 = metadata['global_avg_xg_conceded_p90']
global_avg_sot_p90 = sum(t['sot_p90'] for t in team_metrics_data.values()) / len(team_metrics_data)
global_avg_xg_p90 = sum(t['xg_p90'] for t in team_metrics_data.values()) / len(team_metrics_data)

# Nome da seleção no Supabase (PT) -> nome usado em team_metrics_data (StatsBomb EN)
TEAM_NAME_EN = {
    'Brasil': 'Brazil',
    'Marrocos': 'Morocco',
    'Alemanha': 'Germany',
    'Japão': 'Japan',
    'Espanha': 'Spain',
    'Argentina': 'Argentina',
}

# (team, name) -> player_id
PLAYERS_BY_TEAM_NAME = {(p['team'], p['name']): pid for pid, p in player_metrics_data.items()}

# 11 titulares prováveis por seleção (escolha manual para o teste de validação)
STARTING_XI = {
    'Brasil': [
        'Alisson', 'Danilo', 'Marquinhos', 'Gabriel Magalhães', 'Alex Sandro',
        'Casemiro', 'Bruno Guimarães', 'Lucas Paquetá',
        'Raphinha', 'Vini Jr.', 'Matheus Cunha',
    ],
    'Marrocos': [
        'Bounou', 'Achraf Hakimi', 'Aguerd', 'Diop', 'Mazraoui',
        'Amrabat', 'El Khannouss', 'Azzedine Ounahi',
        'Ez Abde', 'Ayoub El Kaabi', 'Brahim Díaz',
    ],
    'Alemanha': [
        'Manuel Neuer', 'Joshua Kimmich', 'Antonio Rudiger', 'Jonathan Tah', 'David Raum',
        'Florian Wirtz', 'Leon Goretzka', 'Jamal Musiala',
        'Leroy Sané', 'Kai Havertz', 'Maximilian Beier',
    ],
    'Japão': [
        'Zion Suzuki', 'Yukinari Sugawara', 'Ko Itakura', 'Takehiro Tomiyasu', 'Hiroki Ito',
        'Wataru Endo', 'Daichi Kamada', 'Ritsu Doan',
        'Junya Ito', 'Takefusa Kubo', 'Ayase Ueda',
    ],
    'Espanha': [
        'Unai Simón', 'Pedro Porro', 'Cubarsí', 'Eric García', 'Cucurella',
        'Rodri', 'Pedri', 'Gavi',
        'Lamine Yamal', 'Nico Williams', 'Mikel Oyarzabal',
    ],
    'Argentina': [
        'Emiliano Martínez', 'Nahuel Molina', 'Cuti Romero', 'Lisandro Martínez', 'Nicolás Tagliafico',
        'Enzo Fernández', 'De Paul', 'Mac Allister',
        'Lionel Messi', 'Julián Álvarez', 'Lautaro Martínez',
    ],
}

MATCHES = [
    ('Brasil', 'Marrocos'),
    ('Alemanha', 'Japão'),
    ('Espanha', 'Argentina'),
]

FATOR_INTENSIDADE_FASE_GRUPOS = 1.0


def confidence_for(player_data: dict) -> str:
    source = player_data['source']
    if source == 'fbref':
        return {'large': 'high', 'medium': 'medium', 'small': 'medium'}.get(player_data.get('sample_size'), 'medium')
    if source.startswith('fbref_median'):
        return 'medium'
    if source == 'fm23':
        return 'medium'
    return 'low'  # fm23_median e qualquer outro caso


def project_player_metrics(player_id, team, opponent, line_players_data, minutes=90, fator_intensidade=1.0):
    player_data = player_metrics_data[player_id]
    team_data = team_metrics_data.get(team, {})
    opp_data = team_metrics_data.get(opponent, {})

    # Fator defensivo do adversário (1.0 se sem dados StatsBomb)
    defensive_factor = opp_data.get('defensive_factor', 1.0)
    # Fator de xG do adversário, baseado no xG concedido
    xg_factor = opp_data.get('xg_conceded_p90', global_avg_xg_conceded_p90) / global_avg_xg_conceded_p90

    # Volume esperado do time (fallback para média global se sem dados StatsBomb)
    team_shots_p90 = team_data.get('shots_p90', global_avg_shots_p90)
    team_sot_p90 = team_data.get('sot_p90', global_avg_sot_p90)
    team_xg_p90 = team_data.get('xg_p90', global_avg_xg_p90)

    minutes_factor = minutes / 90
    team_shots_expected = team_shots_p90 * defensive_factor * minutes_factor
    team_sot_expected = team_sot_p90 * defensive_factor * minutes_factor
    team_xg_expected = team_xg_p90 * xg_factor * minutes_factor

    # Soma das taxas p90 dos jogadores de linha (DF/MF/FW) do time
    sum_sh = sum(p.get('Sh_p90') or 0 for p in line_players_data)
    sum_sot = sum(p.get('SoT_p90') or 0 for p in line_players_data)
    sum_xg = sum(p.get('xG_p90') or 0 for p in line_players_data)

    sh = player_data.get('Sh_p90') or 0
    sot = player_data.get('SoT_p90') or 0
    xg = player_data.get('xG_p90') or 0

    share_sh = sh / sum_sh if sum_sh > 0 else 0
    share_sot = sot / sum_sot if sum_sot > 0 else 0
    share_xg = xg / sum_xg if sum_xg > 0 else 0

    lambda_shots = share_sh * team_shots_expected
    lambda_sot = share_sot * team_sot_expected
    lambda_xg = share_xg * team_xg_expected

    # Desarmes/faltas: estatística própria do jogador x fator de intensidade
    tkl = player_data.get('Tkl_p90') or 0
    fls = player_data.get('Fls_p90') or 0
    tackles_expected = tkl * fator_intensidade * minutes_factor
    fouls_expected = fls * fator_intensidade * minutes_factor

    return {
        'shots_expected': round(lambda_shots, 2),
        'sot_expected': round(lambda_sot, 2),
        'xg_expected': round(lambda_xg, 3),
        'tackles_expected': round(tackles_expected, 2),
        'fouls_expected': round(fouls_expected, 2),
        'source': player_data['source'],
        'confidence': confidence_for(player_data),
    }


all_rows = []  # para os checks de sanidade ao final
team_sums = []  # soma de shots_expected por time x partida

print('=' * 90)
print('VALIDAÇÃO DO MODELO DE MÉTRICAS TÁTICAS POR JOGADOR — 3 PARTIDAS DE TESTE')
print('=' * 90)

for home, away in MATCHES:
    print('\n' + '#' * 90)
    print(f"# {home} vs {away}")
    print('#' * 90)

    for team, opponent in [(home, away), (away, home)]:
        team_en = TEAM_NAME_EN[team]
        opp_en = TEAM_NAME_EN[opponent]

        team_data = team_metrics_data.get(team_en)
        opp_data = team_metrics_data.get(opp_en)

        print(f"\n--- {team} (vs {opponent}) ---")
        print(f"  StatsBomb {team_en}: {'OK' if team_data else 'sem dados -> usa média global'}")
        print(f"  StatsBomb {opp_en} (adversário): {'OK' if opp_data else 'sem dados -> defensive_factor=1.0'}")

        xi_names = STARTING_XI[team]
        xi_ids = [PLAYERS_BY_TEAM_NAME[(team, name)] for name in xi_names]
        line_players_data = [
            player_metrics_data[pid] for pid in xi_ids
            if player_metrics_data[pid]['position'] != 'GK'
        ]

        rows = []
        for pid, name in zip(xi_ids, xi_names):
            proj = project_player_metrics(
                pid, team_en, opp_en, line_players_data,
                minutes=90, fator_intensidade=FATOR_INTENSIDADE_FASE_GRUPOS,
            )
            row = {
                'player': name,
                'pos': player_metrics_data[pid]['position'],
                **proj,
            }
            rows.append(row)
            all_rows.append(row)

        df = pd.DataFrame(rows).sort_values('shots_expected', ascending=False)
        print(df.to_string(index=False))

        sum_shots = df['shots_expected'].sum()
        team_shots_p90 = (team_data or {}).get('shots_p90', global_avg_shots_p90)
        defensive_factor = (opp_data or {}).get('defensive_factor', 1.0)
        team_shots_expected = team_shots_p90 * defensive_factor
        print(f"\n  Soma shots_expected (XI): {sum_shots:.2f}")
        print(f"  team_shots_p90 ({team_en}): {team_shots_p90:.2f}")
        print(f"  team_shots_expected (= team_shots_p90 x defensive_factor): {team_shots_expected:.2f}")

        team_sums.append({
            'match': f'{home} vs {away}',
            'team': team,
            'sum_shots_expected': sum_shots,
            'team_shots_p90': team_shots_p90,
            'team_shots_expected': team_shots_expected,
        })


# ════════════════════════════════════════════════════════════════════
# CHECAGENS DE SANIDADE
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('CHECAGENS DE SANIDADE')
print('=' * 90)

warnings = []

# 1. Nenhum jogador com shots_expected > 6
for row in all_rows:
    if row['shots_expected'] > 6:
        warnings.append(f"shots_expected > 6 para {row['player']} ({row['pos']}): {row['shots_expected']}")

# 2. Soma dos shots_expected do time deve ser próxima de team_shots_expected
for ts in team_sums:
    diff = abs(ts['sum_shots_expected'] - ts['team_shots_expected'])
    ratio = diff / ts['team_shots_expected'] if ts['team_shots_expected'] else 0
    status = 'OK' if ratio < 0.05 else 'DIVERGENTE'
    print(
        f"  [{status}] {ts['match']:<25} {ts['team']:<10} "
        f"soma_XI={ts['sum_shots_expected']:.2f}  team_shots_expected={ts['team_shots_expected']:.2f}  "
        f"diff%={ratio * 100:.1f}%"
    )
    if ratio >= 0.05:
        warnings.append(
            f"Soma shots_expected do XI de {ts['team']} ({ts['match']}) diverge de "
            f"team_shots_expected em {ratio * 100:.1f}% "
            f"({ts['sum_shots_expected']:.2f} vs {ts['team_shots_expected']:.2f})"
        )

# 3. Nenhum GK com shots_expected > 0.1
for row in all_rows:
    if row['pos'] == 'GK' and row['shots_expected'] > 0.1:
        warnings.append(f"GK {row['player']} com shots_expected={row['shots_expected']} (> 0.1)")

# 4. source=fm23_median => confidence=low
for row in all_rows:
    if row['source'] == 'fm23_median' and row['confidence'] != 'low':
        warnings.append(f"{row['player']}: source=fm23_median mas confidence={row['confidence']}")

print(f"\nTotal de jogadores avaliados: {len(all_rows)}")
print(f"Distribuição de confidence: {pd.Series([r['confidence'] for r in all_rows]).value_counts().to_dict()}")
print(f"Distribuição de source: {pd.Series([r['source'] for r in all_rows]).value_counts().to_dict()}")

if warnings:
    print(f"\n⚠ {len(warnings)} warning(s):")
    for w in warnings:
        print(f"  - {w}")
else:
    print("\nNenhum warning — todas as checagens de sanidade passaram.")
