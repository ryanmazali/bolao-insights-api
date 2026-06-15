"""Combina StatsBomb, FBref, FM23 e Supabase em métricas táticas por
jogador e por seleção, para uso futuro pelo Insights IA.

Não modifica nenhum modelo existente (model_result_v2, model_goals_v2,
model_btts_v2, predict_scorer.py) — apenas gera novos artefatos em
data/processed/ e data/models/.

Uso:
    python scripts/build_player_metrics_model.py
"""

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from supabase import create_client

pd.set_option('display.width', 200)
load_dotenv()

# ── Caminhos ──────────────────────────────────────────────────────────
TEAM_STATS_PATH = 'data/processed/team_stats_statsbomb.csv'
FBREF_PATH = 'data/processed/fbref_players_per90.csv'
FBREF_MEDIANS_PATH = 'data/processed/fbref_medians.json'
FM23_RAW_PATH = 'data/raw/merged_players (1).csv'
FM23_MAPPING_PATH = 'data/processed/fm23_player_mapping.csv'

OUT_TEAM_AGG = 'data/processed/team_aggregated_stats.csv'
OUT_PLAYER_MAPPING = 'data/processed/player_fbref_mapping.json'

MODELS_DIR = Path('data/models')
OUT_PLAYER_METRICS = MODELS_DIR / 'player_metrics_data.json'
OUT_TEAM_METRICS = MODELS_DIR / 'team_metrics_data.json'
OUT_METADATA = MODELS_DIR / 'metrics_metadata.json'

TOP5_LEAGUES = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
FM23_ATTRS = ['Fin', 'OtB', 'Tck', 'Mar', 'Agg']

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))

# country_code (Supabase teams, == código FBref de 3 letras) -> nome do
# time em inglês usado pelo StatsBomb. Mapeamento das 48 seleções da
# Copa 2026 (mesma base usada em scripts/coverage_check.py).
COUNTRY_TO_STATSBOMB = {
    'MEX': 'Mexico', 'RSA': 'South Africa', 'KOR': 'South Korea', 'CZE': 'Czech Republic',
    'CAN': 'Canada', 'BIH': 'Bosnia and Herzegovina', 'QAT': 'Qatar', 'SUI': 'Switzerland',
    'BRA': 'Brazil', 'MAR': 'Morocco', 'HAI': 'Haiti', 'SCO': 'Scotland',
    'USA': 'United States', 'PAR': 'Paraguay', 'AUS': 'Australia', 'TUR': 'Turkey',
    'GER': 'Germany', 'CUW': 'Curacao', 'CIV': 'Ivory Coast', 'ECU': 'Ecuador',
    'NED': 'Netherlands', 'JPN': 'Japan', 'SWE': 'Sweden', 'TUN': 'Tunisia',
    'BEL': 'Belgium', 'EGY': 'Egypt', 'IRN': 'Iran', 'NZL': 'New Zealand',
    'ESP': 'Spain', 'CPV': 'Cape Verde', 'KSA': 'Saudi Arabia', 'URU': 'Uruguay',
    'FRA': 'France', 'SEN': 'Senegal', 'IRQ': 'Iraq', 'NOR': 'Norway',
    'ARG': 'Argentina', 'ALG': 'Algeria', 'AUT': 'Austria', 'JOR': 'Jordan',
    'POR': 'Portugal', 'COD': 'DR Congo', 'UZB': 'Uzbekistan', 'COL': 'Colombia',
    'ENG': 'England', 'CRO': 'Croatia', 'GHA': 'Ghana', 'PAN': 'Panama',
}


def normalize_name(name) -> str:
    """Remove acentos, pontuação e normaliza para minúsculas."""
    if not isinstance(name, str):
        return ''
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def to_native(obj):
    """Converte recursivamente tipos numpy/pandas para tipos nativos do Python."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ════════════════════════════════════════════════════════════════════
# PARTE A — Agrega stats de seleção (Nível 1)
# ════════════════════════════════════════════════════════════════════
print('=' * 70)
print('PARTE A — Agregando stats de seleção (StatsBomb)')
print('=' * 70)

team_games = pd.read_csv(TEAM_STATS_PATH)

AGG_COLS = {
    'shots_p90': 'mean', 'sot_p90': 'mean', 'xg_p90': 'mean',
    'shots_conceded_p90': 'mean', 'xg_conceded_p90': 'mean',
    'tackles_p90': 'mean', 'fouls_p90': 'mean', 'interceptions_p90': 'mean',
}

team_agg = team_games.groupby('team').agg(AGG_COLS).reset_index()
n_games = team_games.groupby('team').size().rename('n_games').reset_index()
tournaments = (
    team_games.groupby('team')['tournament']
    .unique().apply(list).rename('tournaments').reset_index()
)

team_agg = team_agg.merge(n_games, on='team').merge(tournaments, on='team')

global_avg_shots_conceded = team_agg['shots_conceded_p90'].mean()
team_agg['defensive_factor'] = team_agg['shots_conceded_p90'] / global_avg_shots_conceded

team_agg.to_csv(OUT_TEAM_AGG, index=False)

print(f"Times agregados: {len(team_agg)}")
print(f"Média global shots_conceded_p90: {global_avg_shots_conceded:.2f}")

print('\n-- Top 5 defesas mais fortes (defensive_factor menor) --')
print(team_agg.sort_values('defensive_factor').head(5)[
    ['team', 'shots_conceded_p90', 'xg_conceded_p90', 'defensive_factor']
].to_string(index=False))

print('\n-- Top 5 defesas mais fracas (defensive_factor maior) --')
print(team_agg.sort_values('defensive_factor', ascending=False).head(5)[
    ['team', 'shots_conceded_p90', 'xg_conceded_p90', 'defensive_factor']
].to_string(index=False))

print(f"\nSalvo em {OUT_TEAM_AGG}")


# ════════════════════════════════════════════════════════════════════
# PARTE B — Mapeamento de jogadores Copa 2026 → FBref / FM23
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PARTE B — Mapeamento Copa 2026 → FBref / FM23')
print('=' * 70)

# -- FBref --
fbref = pd.read_csv(FBREF_PATH)
fbref['norm_name'] = fbref['Player'].apply(normalize_name)
fbref['nation_code'] = fbref['Nation'].str.split().str[-1]

with open(FBREF_MEDIANS_PATH, encoding='utf-8') as f:
    fbref_medians = json.load(f)

METRIC_COLS = [
    'Gls_p90', 'Ast_p90', 'xG_p90', 'xAG_p90', 'npxG_p90', 'Sh_p90', 'SoT_p90',
    'KP_p90', 'PrgP_p90', 'PrgC_p90', 'Tkl_p90', 'TklW_p90', 'Int_p90',
    'Blocks_p90', 'Clr_p90', 'CrdY_p90', 'Fls_p90', 'Recov_p90',
    'Touches_p90', 'Carries_p90', 'GA_p90', 'Save%', 'CS%', 'CS',
]

# Squad (liga top-5) -> nome da liga, para o Nível 2 (via clube do FM23)
squad_to_league = {}
for _, r in fbref.iterrows():
    if r['Comp'] in TOP5_LEAGUES:
        squad_to_league[normalize_name(r['Squad'])] = r['Comp']

# -- FM23 --
fm23 = pd.read_csv(FM23_RAW_PATH, usecols=['UID', 'Club'] + FM23_ATTRS)
fm23 = fm23.rename(columns={'UID': 'fm23_uid'}).drop_duplicates(subset='fm23_uid')
fm23 = fm23.set_index('fm23_uid')

fm23_mapping = pd.read_csv(FM23_MAPPING_PATH)
fm23_by_supabase = fm23_mapping.set_index('supabase_id')[['fm23_uid', 'status']].to_dict('index')

# -- Supabase: jogadores convocados (players join teams) --
players_raw = []
start = 0
while True:
    resp = (
        supabase.table('players')
        .select('id, name, position, teams!inner(name, country_code)')
        .range(start, start + 999)
        .execute()
    )
    batch = resp.data or []
    players_raw.extend(batch)
    if len(batch) < 1000:
        break
    start += 1000

players_df = pd.DataFrame([{
    'id': p['id'],
    'name': p['name'],
    'position': p['position'],
    'team_name': p['teams']['name'],
    'country_code': p['teams']['country_code'],
} for p in players_raw])

players_df = players_df[players_df['name'] != 'Gol Contra'].copy()
players_df['norm_name'] = players_df['name'].apply(normalize_name)

print(f"Jogadores convocados (Supabase, excluindo 'Gol Contra'): {len(players_df)}")

# Medianas FM23 por posição (Nível 4), a partir dos jogadores com match real
matched_fm23 = fm23_mapping[fm23_mapping['status'].isin(['matched', 'proxy'])].copy()
matched_fm23 = matched_fm23.merge(players_df[['id', 'position']], left_on='supabase_id', right_on='id', how='inner')
matched_fm23 = matched_fm23.merge(fm23[FM23_ATTRS], left_on='fm23_uid', right_index=True, how='left')
fm23_pos_medians = matched_fm23.groupby('position')[FM23_ATTRS].median()
fm23_global_median = matched_fm23[FM23_ATTRS].median()


def fm23_estimate(attrs, position) -> dict:
    """Converte atributos FM23 (escala 1-20) em estimativas de métricas p90."""
    fin, otb, tck, mar, agg = attrs['Fin'], attrs['OtB'], attrs['Tck'], attrs['Mar'], attrs['Agg']

    if position == 'FW':
        sh_p90 = (fin + otb) / 200 * 3.5
    elif position == 'MF':
        sh_p90 = (fin + otb) / 200 * 2.0
    elif position == 'DF':
        sh_p90 = (fin + otb) / 200 * 0.8
    else:
        sh_p90 = 0.0

    xg_p90 = sh_p90 * (fin / 200)

    if position == 'DF':
        tkl_p90 = (tck + mar) / 200 * 4.0
    elif position == 'MF':
        tkl_p90 = (tck + mar) / 200 * 2.5
    else:
        tkl_p90 = 0.0

    fls_p90 = (agg / 100) * 2.0

    return {
        'Sh_p90': float(sh_p90),
        'xG_p90': float(xg_p90),
        'Tkl_p90': float(tkl_p90),
        'Fls_p90': float(fls_p90),
    }


mapping = {}
counts = {'fbref': 0, 'fbref_median': 0, 'fm23': 0, 'fm23_median': 0}
unmatched_log = []

for _, p in players_df.iterrows():
    pid, name, pos, country, norm = p['id'], p['name'], p['position'], p['country_code'], p['norm_name']

    candidates = fbref[fbref['nation_code'] == country]

    row = None
    if not candidates.empty:
        exact = candidates[candidates['norm_name'] == norm]
        if not exact.empty:
            row = exact.sort_values('Min', ascending=False).iloc[0]
        else:
            result = process.extractOne(
                norm, candidates['norm_name'].tolist(), scorer=fuzz.WRatio, score_cutoff=85,
            )
            if result is not None:
                row = candidates.iloc[result[2]]

    if row is not None:
        # Nível 1 — FBref individual
        entry = {
            'fbref_name': row['Player'],
            'source': 'fbref',
            'sample_size': row['sample_size'],
        }
        for col in METRIC_COLS:
            val = row[col] if col in row.index else np.nan
            entry[col] = None if pd.isna(val) else float(val)
        counts['fbref'] += 1
    else:
        # Tenta Nível 2 — mediana FBref por liga/posição, via clube do FM23
        league = None
        fm23_info = fm23_by_supabase.get(pid)
        if fm23_info is not None and not pd.isna(fm23_info['fm23_uid']):
            uid = fm23_info['fm23_uid']
            if uid in fm23.index:
                club = fm23.loc[uid, 'Club']
                league = squad_to_league.get(normalize_name(club))

        if league is not None:
            med = fbref_medians.get(league, {}).get(pos, {})
            entry = {
                'fbref_name': None,
                'source': f'fbref_median_{league}',
                'sample_size': None,
            }
            for col, val in med.items():
                entry[col] = val
            counts['fbref_median'] += 1
        elif fm23_info is not None and fm23_info['status'] in ('matched', 'proxy') and fm23_info['fm23_uid'] in fm23.index:
            # Nível 3 — FM23 individual
            attrs = fm23.loc[fm23_info['fm23_uid']]
            entry = {'fbref_name': None, 'source': 'fm23', 'sample_size': None}
            entry.update(fm23_estimate(attrs, pos))
            counts['fm23'] += 1
        else:
            # Nível 4 — mediana FM23 por posição (último recurso)
            attrs = fm23_pos_medians.loc[pos] if pos in fm23_pos_medians.index else fm23_global_median
            entry = {'fbref_name': None, 'source': 'fm23_median', 'sample_size': None, 'confidence': 'low'}
            entry.update(fm23_estimate(attrs, pos))
            counts['fm23_median'] += 1
            unmatched_log.append(f"{name} ({p['team_name']}, {pos})")

    entry['name'] = name
    entry['position'] = pos
    entry['team'] = p['team_name']
    entry['country_code'] = country

    mapping[pid] = entry

print('\n-- Resultado do mapeamento por nível de fallback --')
for k, v in counts.items():
    print(f"  {k:<14}: {v}")

if unmatched_log:
    print(f"\nJogadores em Nível 4 (fm23_median, confidence=low): {len(unmatched_log)}")
    for line in unmatched_log[:20]:
        print(f"  - {line}")
    if len(unmatched_log) > 20:
        print(f"  ... e mais {len(unmatched_log) - 20}")

mapping_native = to_native(mapping)

with open(OUT_PLAYER_MAPPING, 'w', encoding='utf-8') as f:
    json.dump(mapping_native, f, indent=2, ensure_ascii=False)

print(f"\nSalvo em {OUT_PLAYER_MAPPING}")

# Exemplo de entrada por nível
print('\n-- Exemplo de entrada por nível --')
for level in ['fbref', 'fm23', 'fm23_median']:
    example = next((v for v in mapping_native.values() if v['source'] == level), None)
    if example is None:
        example = next((v for v in mapping_native.values() if v['source'].startswith(level)), None)
    if example:
        print(f"\n[{level}] {example['name']} ({example['team']}, {example['position']}):")
        print(json.dumps(example, indent=2, ensure_ascii=False))


# ════════════════════════════════════════════════════════════════════
# PARTE C — Salva artefatos finais
# ════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PARTE C — Salvando artefatos finais em data/models/')
print('=' * 70)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# player_metrics_data.json (== player_fbref_mapping.json renomeado)
with open(OUT_PLAYER_METRICS, 'w', encoding='utf-8') as f:
    json.dump(mapping_native, f, indent=2, ensure_ascii=False)

# team_metrics_data.json (== team_aggregated_stats.csv como JSON)
team_metrics = {}
for _, r in team_agg.iterrows():
    team_metrics[r['team']] = to_native({k: v for k, v in r.items() if k != 'team'})

with open(OUT_TEAM_METRICS, 'w', encoding='utf-8') as f:
    json.dump(team_metrics, f, indent=2, ensure_ascii=False)

metadata = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'tournaments_used': sorted(team_games['tournament'].unique().tolist()),
    'n_teams_with_statsbomb': len(team_agg),
    'n_players_fbref': counts['fbref'],
    'n_players_fbref_median': counts['fbref_median'],
    'n_players_fm23_fallback': counts['fm23'],
    'n_players_fm23_median_fallback': counts['fm23_median'],
    'global_avg_shots_p90': float(team_agg['shots_p90'].mean()),
    'global_avg_xg_conceded_p90': float(team_agg['xg_conceded_p90'].mean()),
}

with open(OUT_METADATA, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Salvo: {OUT_PLAYER_METRICS}")
print(f"Salvo: {OUT_TEAM_METRICS}")
print(f"Salvo: {OUT_METADATA}")
print('\nmetrics_metadata.json:')
print(json.dumps(metadata, indent=2, ensure_ascii=False))


# ── Resumo final de cobertura por seleção da Copa 2026 ──────────────
print('\n' + '=' * 70)
print('RESUMO FINAL — Cobertura por seleção (Copa 2026)')
print('=' * 70)

team_agg_names = set(team_agg['team'])
summary_rows = []
for country_code, group in players_df.groupby('country_code'):
    team_name = group['team_name'].iloc[0]
    sources = [mapping[i]['source'] for i in group['id']]
    statsbomb_en = COUNTRY_TO_STATSBOMB.get(country_code)
    summary_rows.append({
        'team': team_name,
        'n_players': len(sources),
        'fbref': sum(1 for s in sources if s == 'fbref'),
        'fbref_median': sum(1 for s in sources if s.startswith('fbref_median')),
        'fm23': sum(1 for s in sources if s == 'fm23'),
        'fm23_median': sum(1 for s in sources if s == 'fm23_median'),
        'statsbomb_team_data': 'yes' if statsbomb_en in team_agg_names else 'no',
    })

summary_df = pd.DataFrame(summary_rows).sort_values('team')
print(summary_df.to_string(index=False))

print(f"\nTotal de seleções com dados StatsBomb: {(summary_df['statsbomb_team_data'] == 'yes').sum()} / {len(summary_df)}")
