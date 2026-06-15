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
FIFA_RANKINGS_PATH = 'data/fifa_rankings.json'
FIFA_RANKING_FULL_PATH = 'data/raw/fifa_ranking-2024-04-04.csv'

OUT_TEAM_AGG = 'data/processed/team_aggregated_stats.csv'
OUT_PLAYER_MAPPING = 'data/processed/player_fbref_mapping.json'

MODELS_DIR = Path('data/models')
OUT_PLAYER_METRICS = MODELS_DIR / 'player_metrics_data.json'
OUT_TEAM_METRICS = MODELS_DIR / 'team_metrics_data.json'
OUT_METADATA = MODELS_DIR / 'metrics_metadata.json'

TOP5_LEAGUES = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
FM23_ATTRS = ['Fin', 'OtB', 'Tck', 'Mar', 'Agg']

# Snapshot do player_metrics_data.json anterior (antes da recalibração de
# Sh_p90 para source=fm23/fm23_median) — usado para imprimir antes/depois.
if OUT_PLAYER_METRICS.exists():
    with open(OUT_PLAYER_METRICS, encoding='utf-8') as f:
        old_player_metrics_data = json.load(f)
else:
    old_player_metrics_data = {}

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


# Compatibilidade de posição entre Supabase (convocação) e FBref (match
# por nome). GK só pode casar com GK; demais posições aceitam a vizinhança
# tática mais próxima (ex.: FW casa com FW/MF, mas não com DF).
POSITION_COMPAT = {
    'GK': {'GK'},
    'FW': {'FW', 'MF'},
    'MF': {'MF', 'FW', 'DF'},
    'DF': {'DF', 'MF'},
}


def positions_compatible(supabase_pos, fbref_pos) -> bool:
    return fbref_pos in POSITION_COMPAT.get(supabase_pos, {supabase_pos})


# Nomes de seleções da Copa 2026 (COUNTRY_TO_STATSBOMB) que aparecem com
# grafia diferente em data/fifa_rankings.json.
FIFA_NAME_ALIASES = {
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
}


def fifa_shots_p90_factor(rank: int) -> float:
    """Fator de ajuste de shots_p90 (sobre a média global) com base no
    ranking FIFA, para seleções sem dados StatsBomb (Nível 1 do endpoint
    /predict/player-metrics)."""
    if rank <= 20:
        return 1.10
    if rank <= 40:
        return 1.0
    if rank <= 60:
        return 0.85
    return 0.72


# Nomes de adversários (StatsBomb, team_stats_statsbomb.csv) que aparecem
# com grafia diferente em data/raw/fifa_ranking-2024-04-04.csv.
OPPONENT_FIFA_NAME_ALIASES = {
    'Cape Verde Islands': 'Cabo Verde',
    'Czech Republic': 'Czechia',
    'Gambia': 'The Gambia',
    'United States': 'USA',
}

# Peso da força do adversário na agregação de shots_p90 por time
# (PARTE A). Adversário fraco pesa menos — evita que seleções de
# torneios continentais (ex.: AFCON) com calendário pesado contra
# seleções fracas tenham shots_p90 inflado.
DEFAULT_OPPONENT_WEIGHT = 0.6


def fifa_opponent_weight(rank: float | None) -> float:
    """Peso do jogo na média de shots_p90, com base no ranking FIFA do
    adversário: adversário top-20 pesa 1.0, top-40 pesa 0.8, top-60 pesa
    0.6, e acima de 60 (ou ranking desconhecido) pesa 0.4/0.6."""
    if rank is None or pd.isna(rank):
        return DEFAULT_OPPONENT_WEIGHT
    if rank <= 20:
        return 1.0
    if rank <= 40:
        return 0.8
    if rank <= 60:
        return 0.6
    return 0.4


def shots_p90_regression_factor(avg_opponent_rank: float) -> float:
    """Fator de regressão sobre shots_p90_weighted, com base no ranking
    FIFA médio dos adversários enfrentados pelo time: times cuja amostra
    é dominada por adversários fracos (avg_opponent_rank alto) têm
    shots_p90 puxado para baixo, por ser uma amostra menos confiável para
    projetar jogos de Copa do Mundo."""
    if avg_opponent_rank < 40:
        return 1.0
    if avg_opponent_rank <= 55:
        return 0.85
    return 0.60


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

# -- shots_p90_weighted: pondera cada jogo pela força do adversário (ranking
# FIFA), para que jogos contra seleções fracas (comuns em torneios
# continentais como a AFCON) pesem menos na média de shots_p90 do time. --
fifa_ranking_full = pd.read_csv(FIFA_RANKING_FULL_PATH)
fifa_ranking_latest = fifa_ranking_full[fifa_ranking_full['rank_date'] == fifa_ranking_full['rank_date'].max()]
fifa_rank_by_name = fifa_ranking_latest.set_index('country_full')['rank'].to_dict()

team_games['opp_fifa_name'] = team_games['opponent'].map(lambda o: OPPONENT_FIFA_NAME_ALIASES.get(o, o))
team_games['opp_fifa_rank'] = team_games['opp_fifa_name'].map(fifa_rank_by_name)
team_games['opp_weight'] = team_games['opp_fifa_rank'].apply(fifa_opponent_weight)

unranked_opponents = sorted(set(team_games.loc[team_games['opp_fifa_rank'].isna(), 'opponent']))
if unranked_opponents:
    print(f"\nAdversários sem ranking FIFA em {FIFA_RANKING_FULL_PATH} "
          f"(peso default={DEFAULT_OPPONENT_WEIGHT}): {unranked_opponents}")

team_games['_w_shots'] = team_games['shots_p90'] * team_games['opp_weight']
shots_weighted = (
    team_games.groupby('team')
    .apply(lambda g: g['_w_shots'].sum() / g['opp_weight'].sum(), include_groups=False)
    .rename('shots_p90_weighted').reset_index()
)
team_agg = team_agg.merge(shots_weighted, on='team')

# -- shots_p90_final: regressão adicional com base no ranking FIFA médio
# dos adversários enfrentados pelo time (amostras dominadas por
# adversários fracos são puxadas para baixo). --
avg_opponent_rank = (
    team_games.groupby('team')['opp_fifa_rank'].mean()
    .rename('avg_opponent_rank').reset_index()
)
team_agg = team_agg.merge(avg_opponent_rank, on='team')
team_agg['regression_factor'] = team_agg['avg_opponent_rank'].apply(shots_p90_regression_factor)
team_agg['shots_p90_final'] = team_agg['shots_p90_weighted'] * team_agg['regression_factor']

team_agg.to_csv(OUT_TEAM_AGG, index=False)

print(f"Times agregados: {len(team_agg)}")
print(f"Média global shots_conceded_p90: {global_avg_shots_conceded:.2f}")

# -- Antes/depois da ponderação por força do adversário --
print('\n-- shots_p90 antes/depois da ponderação por ranking FIFA do adversário --')
weighting_diff = (team_agg['shots_p90'] - team_agg['shots_p90_weighted']).abs()
most_affected = team_agg.assign(diff=weighting_diff).sort_values('diff', ascending=False).head(8)
print(most_affected[['team', 'shots_p90', 'shots_p90_weighted', 'diff']].to_string(index=False))

print('\n-- Destaques AFCON (Egito, Costa do Marfim, Nigéria) --')
for team_name in ['Egypt', "Côte d'Ivoire", 'Nigeria']:
    row = team_agg[team_agg['team'] == team_name]
    if row.empty:
        print(f"  {team_name}: não encontrado em team_agg")
        continue
    r = row.iloc[0]
    games = team_games[team_games['team'] == team_name][['opponent', 'opp_fifa_rank', 'opp_weight', 'shots_p90']]
    print(
        f"\n  {team_name}: shots_p90 (antes)={r['shots_p90']:.2f}  ->  "
        f"shots_p90_weighted={r['shots_p90_weighted']:.2f}  ->  "
        f"shots_p90_final={r['shots_p90_final']:.2f}  "
        f"(avg_opponent_rank={r['avg_opponent_rank']:.1f}, regression_factor={r['regression_factor']:.2f})"
    )
    print(games.to_string(index=False))

# -- Times afetados pela regressão por ranking médio dos adversários --
print('\n-- Times com regression_factor < 1.0 (avg_opponent_rank >= 40) --')
regressed = team_agg[team_agg['regression_factor'] < 1.0].sort_values('avg_opponent_rank')
print(regressed[['team', 'avg_opponent_rank', 'regression_factor', 'shots_p90_weighted', 'shots_p90_final']].to_string(index=False))

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


FBREF_GLOBAL_SH_P90_MEDIAN = {pos: vals['Sh_p90'] for pos, vals in fbref_medians['global'].items()}


def fm23_estimate(attrs, position) -> dict:
    """Converte atributos FM23 (escala 1-20) em estimativas de métricas p90.

    Sh_p90 é calibrado contra a mediana real do FBref por posição: um
    jogador com fm23_score (Fin/OtB normalizados) = 0.5 tem Sh_p90 igual
    à mediana FBref da posição; acima/abaixo de 0.5 escala
    proporcionalmente. GK sempre tem Sh_p90 = 0.
    """
    fin, otb, tck, mar, agg = attrs['Fin'], attrs['OtB'], attrs['Tck'], attrs['Mar'], attrs['Agg']

    if position == 'GK':
        sh_p90 = 0.0
    else:
        fm23_score = (fin * 0.6 + otb * 0.4) / 20
        sh_p90 = fm23_score * FBREF_GLOBAL_SH_P90_MEDIAN[position] * 2

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
position_mismatch_log = []
gk_sh_fix_log = []

for _, p in players_df.iterrows():
    pid, name, pos, country, norm = p['id'], p['name'], p['position'], p['country_code'], p['norm_name']

    candidates = fbref[fbref['nation_code'] == country]

    row = None
    if not candidates.empty:
        exact = candidates[candidates['norm_name'] == norm]
        if not exact.empty:
            # Entre matches exatos por nome, prioriza candidatos com
            # posição compatível antes de desempatar por minutos (evita
            # casar com um homônimo de posição diferente que tenha mais Min).
            compat_exact = exact[exact['Pos'].apply(lambda fp: positions_compatible(pos, fp))]
            if not compat_exact.empty:
                row = compat_exact.sort_values('Min', ascending=False).iloc[0]
            else:
                row = exact.sort_values('Min', ascending=False).iloc[0]
        else:
            result = process.extractOne(
                norm, candidates['norm_name'].tolist(), scorer=fuzz.WRatio, score_cutoff=85,
            )
            if result is not None:
                row = candidates.iloc[result[2]]

    # BUG 4 — valida compatibilidade de posição entre Supabase e FBref;
    # descarta o match (cai no fallback) se incompatível.
    mismatch_entry = None
    if row is not None and not positions_compatible(pos, row['Pos']):
        mismatch_entry = {
            'name': name, 'position': pos, 'team': p['team_name'],
            'fbref_name': row['Player'], 'fbref_squad': row['Squad'],
            'fbref_comp': row['Comp'], 'fbref_pos': row['Pos'], 'fbref_min': int(row['Min']),
        }
        position_mismatch_log.append(mismatch_entry)
        row = None

    if name == 'Mohamed Salah' and row is not None:
        print(
            f"\n[BUG4 check] Mohamed Salah casou com FBref: "
            f"fbref_name={row['Player']!r}, Squad={row['Squad']!r}, "
            f"Comp={row['Comp']!r}, Pos={row['Pos']!r}, Min={int(row['Min'])}"
        )

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

        # BUG 1 — GK não pode ter Sh_p90/xG_p90 herdados de um match errado.
        if pos == 'GK' and (entry.get('Sh_p90') or 0) > 0.1:
            print(
                f"WARNING: GK {name} ({p['team_name']}) tem Sh_p90="
                f"{entry['Sh_p90']:.2f} (fbref_name={entry['fbref_name']!r}) "
                f"— forçando Sh_p90=0, xG_p90=0"
            )
            entry['Sh_p90'] = 0.0
            entry['xG_p90'] = 0.0
            gk_sh_fix_log.append(name)

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

    if mismatch_entry is not None:
        mismatch_entry['fallback_used'] = entry['source']

print('\n-- Resultado do mapeamento por nível de fallback --')
for k, v in counts.items():
    print(f"  {k:<14}: {v}")

if gk_sh_fix_log:
    print("\n-- BUG 1: GKs com Sh_p90/xG_p90 zerados (match com Sh_p90 > 0.1) --")
    for n in gk_sh_fix_log:
        print(f"  - {n}")

if position_mismatch_log:
    print(f"\n-- BUG 4: Matches descartados por incompatibilidade de posição ({len(position_mismatch_log)}) --")
    for m in position_mismatch_log:
        print(
            f"  - {m['name']} ({m['team']}, pos={m['position']}) "
            f"<x> FBref {m['fbref_name']!r} (Squad={m['fbref_squad']!r}, Comp={m['fbref_comp']!r}, "
            f"Pos={m['fbref_pos']!r}, Min={m['fbref_min']}) "
            f"-> fallback usado: {m['fallback_used']}"
        )

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

# -- Recalibração de Sh_p90 (fm23 / fm23_median) -- antes/depois (Egito) --
print('\n-- Sh_p90 antes/depois da recalibração FM23 vs mediana FBref (Egito, fm23/fm23_median) --')
for pid, entry in mapping_native.items():
    if entry['country_code'] != 'EGY' or entry['source'] not in ('fm23', 'fm23_median'):
        continue
    old_sh = old_player_metrics_data.get(pid, {}).get('Sh_p90')
    new_sh = entry['Sh_p90']
    old_str = f"{old_sh:.3f}" if old_sh is not None else "n/a"
    print(f"  {entry['name']:<24} pos={entry['position']:<3} source={entry['source']:<12} "
          f"Sh_p90 antes={old_str:>6}  ->  depois={new_sh:.3f}")


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
# shots_p90 usa shots_p90_final (ponderado por força do adversário e
# regredido pelo ranking FIFA médio dos adversários); shots_p90_raw
# guarda a média simples original e shots_p90_weighted o valor
# intermediário (sem a regressão por avg_opponent_rank).
global_avg_shots_p90 = float(team_agg['shots_p90_final'].mean())

team_metrics = {}
for _, r in team_agg.iterrows():
    entry = to_native({k: v for k, v in r.items() if k != 'team'})
    entry['shots_p90_raw'] = entry.pop('shots_p90')
    entry['shots_p90'] = entry.pop('shots_p90_final')
    entry['shots_p90_source'] = 'statsbomb'
    team_metrics[r['team']] = entry

# Nível 1 — para seleções da Copa 2026 sem dados StatsBomb, estima
# shots_p90 a partir do ranking FIFA (em vez de usar sempre a média
# global, que superestima times medianos/fracos).
with open(FIFA_RANKINGS_PATH, encoding='utf-8') as f:
    fifa_rankings = json.load(f)

statsbomb_teams = set(team_agg['team'])
fifa_estimate_log = []
for team_name in sorted(set(COUNTRY_TO_STATSBOMB.values())):
    if team_name in statsbomb_teams:
        continue
    fifa_key = FIFA_NAME_ALIASES.get(team_name, team_name)
    rank = fifa_rankings.get(fifa_key)
    if rank is None:
        continue
    shots_p90_est = global_avg_shots_p90 * fifa_shots_p90_factor(rank)
    team_metrics[team_name] = {
        'shots_p90': shots_p90_est,
        'shots_p90_source': 'fifa_rank_estimate',
        'fifa_ranking': rank,
    }
    fifa_estimate_log.append((team_name, rank, shots_p90_est))

with open(OUT_TEAM_METRICS, 'w', encoding='utf-8') as f:
    json.dump(to_native(team_metrics), f, indent=2, ensure_ascii=False)

metadata = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'tournaments_used': sorted(team_games['tournament'].unique().tolist()),
    'n_teams_with_statsbomb': len(team_agg),
    'n_players_fbref': counts['fbref'],
    'n_players_fbref_median': counts['fbref_median'],
    'n_players_fm23_fallback': counts['fm23'],
    'n_players_fm23_median_fallback': counts['fm23_median'],
    'global_avg_shots_p90': global_avg_shots_p90,
    'global_avg_xg_conceded_p90': float(team_agg['xg_conceded_p90'].mean()),
}

with open(OUT_METADATA, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Salvo: {OUT_PLAYER_METRICS}")
print(f"Salvo: {OUT_TEAM_METRICS}")
print(f"Salvo: {OUT_METADATA}")
print('\nmetrics_metadata.json:')
print(json.dumps(metadata, indent=2, ensure_ascii=False))

print(f"\n-- Nível 1: shots_p90 estimado via ranking FIFA ({len(fifa_estimate_log)} seleções sem StatsBomb) --")
for team_name, rank, shots_p90_est in sorted(fifa_estimate_log, key=lambda x: x[1]):
    print(f"  {team_name:<25} FIFA #{rank:<4} -> shots_p90={shots_p90_est:.2f} (global_avg={global_avg_shots_p90:.2f})")

teams_without_estimate = sorted(set(COUNTRY_TO_STATSBOMB.values()) - statsbomb_teams - {t for t, _, _ in fifa_estimate_log})
if teams_without_estimate:
    print(f"\n  Sem StatsBomb e sem ranking FIFA (usarão global_avg_shots_p90): {teams_without_estimate}")


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
