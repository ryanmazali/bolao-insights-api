import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────
CUTOFF_DATE = '2006-01-01'
REFERENCE_DATE = pd.Timestamp('2026-06-10')

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup':                     1.00,
    'UEFA Euro':                          0.90,
    'Copa América':                       0.90,
    'Africa Cup of Nations':              0.85,
    'UEFA Nations League':                0.80,
    'CONCACAF Gold Cup':                  0.75,
    'AFC Asian Cup':                      0.75,
    'FIFA Confederations Cup':            0.75,
    'FIFA World Cup qualification':       0.70,
    'UEFA Euro qualification':            0.65,
    'Copa América qualification':         0.65,
    'African Cup of Nations qualification': 0.60,
    'CONCACAF Nations League':            0.60,
    'Friendly':                           0.25,
}

# Normalização de nomes
NAME_MAPPING = {
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cote d\'Ivoire': 'Ivory Coast',
    'Türkiye': 'Turkey',
    'China PR': 'China',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'United States': 'USA',
    'Czechia': 'Czech Republic',
    'North Macedonia': 'North Macedonia',
    'Congo DR': 'DR Congo',
    'Cabo Verde': 'Cape Verde',
    'Curaçao': 'Curacao',
    'Saint Kitts and Nevis': 'St Kitts and Nevis',
    'São Tomé and Príncipe': 'Sao Tome and Principe',
    'Democratic Republic of Congo': 'DR Congo',
}

# Elo médio global (~1750), usado para normalizar elo_weight e como fallback
# para seleções sem histórico em data/raw/eloratings.csv
GLOBAL_AVG_ELO = 1750.0

def normalize_name(name: str) -> str:
    return NAME_MAPPING.get(name, name)

# ── FM23 — TRADUÇÃO DE NOMES (Supabase, pt-BR) → INGLÊS (results.csv) ──
FM23_TEAM_NAME_PT_TO_EN = {
    'Alemanha': 'Germany',
    'Argentina': 'Argentina',
    'Argélia': 'Algeria',
    'Arábia Saudita': 'Saudi Arabia',
    'Austrália': 'Australia',
    'Brasil': 'Brazil',
    'Bélgica': 'Belgium',
    'Bósnia': 'Bosnia-Herzegovina',
    'Cabo Verde': 'Cape Verde',
    'Canadá': 'Canada',
    'Catar': 'Qatar',
    'Colômbia': 'Colombia',
    'Coreia do Sul': 'South Korea',
    'Costa do Marfim': 'Ivory Coast',
    'Croácia': 'Croatia',
    'Curaçao': 'Curacao',
    'Egito': 'Egypt',
    'Equador': 'Ecuador',
    'Escócia': 'Scotland',
    'Espanha': 'Spain',
    'Estados Unidos': 'USA',
    'França': 'France',
    'Gana': 'Ghana',
    'Haiti': 'Haiti',
    'Holanda': 'Netherlands',
    'Inglaterra': 'England',
    'Iraque': 'Iraq',
    'Irã': 'Iran',
    'Japão': 'Japan',
    'Jordânia': 'Jordan',
    'Marrocos': 'Morocco',
    'México': 'Mexico',
    'Noruega': 'Norway',
    'Nova Zelândia': 'New Zealand',
    'Panamá': 'Panama',
    'Paraguai': 'Paraguay',
    'Portugal': 'Portugal',
    'Rep. D. Congo': 'DR Congo',
    'Senegal': 'Senegal',
    'Suécia': 'Sweden',
    'Suíça': 'Switzerland',
    'Tchéquia': 'Czech Republic',
    'Tunísia': 'Tunisia',
    'Turquia': 'Turkey',
    'Uruguai': 'Uruguay',
    'Uzbequistão': 'Uzbekistan',
    'África do Sul': 'South Africa',
    'Áustria': 'Austria',
}

FM23_METRICS = [
    'attack_strength', 'best_attacker', 'top3_attack',
    'defense_strength', 'best_defender', 'top5_defense',
    'gk_strength',
    'overall', 'best_overall', 'top11_overall', 'depth_overall', 'std_overall',
]

def load_team_fm23_features():
    """Carrega métricas agregadas FM23 por seleção (data/processed/team_fm23_features.csv),
    traduzidas para os nomes em inglês usados em match_features. Retorna:
      - fm23_lookup: dict {team_name -> {métrica: valor}}
      - fm23_defaults: médias globais, usadas para times sem dados FM23
      - former_to_current: dict {nome antigo -> nome atual normalizado},
        construído a partir de data/raw/former_names.csv
    """
    df = pd.read_csv('data/processed/team_fm23_features.csv')
    df['team_name'] = df['team_name'].map(FM23_TEAM_NAME_PT_TO_EN).fillna(df['team_name'])

    former_names = pd.read_csv('data/raw/former_names.csv')
    former_to_current = {
        row['former']: normalize_name(row['current'])
        for _, row in former_names.iterrows()
    }

    fm23_lookup = df.set_index('team_name')[FM23_METRICS].to_dict('index')
    fm23_defaults = df[FM23_METRICS].mean().to_dict()

    return fm23_lookup, fm23_defaults, former_to_current

def get_team_fm23(team: str, fm23_lookup: dict, fm23_defaults: dict, former_to_current: dict) -> dict:
    """Retorna as métricas FM23 de `team`, com fallback via former_names.csv
    e, por fim, médias globais para times fora das 48 seleções mapeadas."""
    if team in fm23_lookup:
        return fm23_lookup[team]
    resolved = former_to_current.get(team)
    if resolved in fm23_lookup:
        return fm23_lookup[resolved]
    return fm23_defaults

def get_tournament_weight(tournament: str) -> float:
    t_lower = tournament.lower()
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t_lower:
            return w
    return 0.50

def recency_weight(match_date: pd.Timestamp, reference: pd.Timestamp) -> float:
    """Peso exponencial por recência. Decai de 1.0 até mínimo 0.15."""
    days_ago = (reference - match_date).days
    return max(0.15, np.exp(-0.0003 * days_ago))

# ── CARREGAR DADOS ────────────────────────────────────────────────────
def load_results(elo_df: pd.DataFrame, current_elo: dict) -> pd.DataFrame:
    df = pd.read_csv('data/raw/results.csv', parse_dates=['date'])

    # Remover partidas sem score (futuras)
    df = df.dropna(subset=['home_score', 'away_score'])

    # Filtrar a partir de 2006
    df = df[df['date'] >= CUTOFF_DATE].copy()

    # Normalizar nomes
    df['home_team'] = df['home_team'].apply(normalize_name)
    df['away_team'] = df['away_team'].apply(normalize_name)

    # Adicionar resultado
    df['result'] = np.where(
        df['home_score'] > df['away_score'], 'H',
        np.where(df['home_score'] < df['away_score'], 'A', 'D')
    )
    df['total_goals'] = df['home_score'] + df['away_score']
    df['btts'] = ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)

    # Pesos
    df['tournament_weight'] = df['tournament'].apply(get_tournament_weight)
    df['recency_weight'] = df['date'].apply(
        lambda d: recency_weight(d, REFERENCE_DATE)
    )

    # Peso composto: tournament_weight * elo_weight, onde elo_weight é o
    # Elo médio das duas seleções normalizado pela média global (~1750).
    # Jogos entre seleções fortes pesam mais; jogos assimétricos (ex.:
    # Gibraltar, Curaçao) continuam presentes mas com peso reduzido.
    df['home_elo'] = df.apply(
        lambda r: get_elo_at_date(elo_df, current_elo, r['home_team'], r['date']),
        axis=1
    )
    df['away_elo'] = df.apply(
        lambda r: get_elo_at_date(elo_df, current_elo, r['away_team'], r['date']),
        axis=1
    )
    df['avg_elo'] = (df['home_elo'] + df['away_elo']) / 2
    df['elo_weight'] = df['avg_elo'] / GLOBAL_AVG_ELO
    df['match_weight'] = df['tournament_weight'] * df['elo_weight']

    df['sample_weight'] = df['match_weight'] * df['recency_weight']

    print(f"[Data] Partidas carregadas: {len(df)}")
    print(f"[Data] Período: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"[Data] Torneios únicos: {df['tournament'].nunique()}")

    return df.sort_values('date').reset_index(drop=True)

# ── RANKING FIFA HISTÓRICO ────────────────────────────────────────────
def load_fifa_rankings() -> pd.DataFrame:
    """Carrega e consolida os 3 arquivos de ranking FIFA."""

    files = [
        'data/raw/fifa_ranking-2023-07-20.csv',
        'data/raw/fifa_ranking-2024-04-04.csv',
        'data/raw/fifa_ranking-2024-06-20.csv',
    ]

    dfs = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f, parse_dates=['rank_date'])
            dfs.append(df)

    rankings = pd.concat(dfs, ignore_index=True)
    rankings['country_full'] = rankings['country_full'].apply(normalize_name)
    rankings = rankings.sort_values('rank_date')

    # Ranking atual (junho 2026) — construído a partir dos dados colados
    current_rankings = {
        'Argentina': 1877.27, 'France': 1870.70, 'Spain': 1874.71,
        'England': 1827.05, 'Brazil': 1765.86, 'Belgium': 1742.24,
        'Netherlands': 1753.57, 'Portugal': 1766.18, 'Germany': 1735.77,
        'Croatia': 1714.87, 'Italy': 1704.73, 'Colombia': 1698.35,
        'Mexico': 1687.48, 'Senegal': 1684.07, 'Uruguay': 1673.07,
        'USA': 1671.23, 'Japan': 1661.58, 'Switzerland': 1650.06,
        'Iran': 1619.58, 'Denmark': 1619.47, 'Turkey': 1605.73,
        'Ecuador': 1598.52, 'Austria': 1597.40, 'South Korea': 1591.63,
        'Nigeria': 1586.69, 'Australia': 1579.34, 'Algeria': 1571.03,
        'Egypt': 1562.37, 'Canada': 1559.48, 'Norway': 1557.44,
        'Ukraine': 1549.29, 'Ivory Coast': 1540.87, 'Panama': 1539.16,
        'Russia': 1529.60, 'Poland': 1526.18, 'Wales': 1516.95,
        'Sweden': 1509.79, 'Hungary': 1506.39, 'Czech Republic': 1505.74,
        'Paraguay': 1505.35, 'Scotland': 1503.34, 'Serbia': 1502.13,
        'Cameroon': 1481.24, 'Tunisia': 1476.41, 'DR Congo': 1474.43,
        'Slovakia': 1473.66, 'Greece': 1473.19, 'Venezuela': 1469.18,
        'Uzbekistan': 1458.73, 'Chile': 1458.20, 'Peru': 1457.69,
        'Costa Rica': 1457.00, 'Romania': 1455.89, 'Mali': 1455.59,
        'Qatar': 1450.31, 'Iraq': 1446.28, 'Morocco': 1755.10,
        'South Africa': 1428.38, 'Saudi Arabia': 1423.88,
        'Ghana': 1346.88, 'Haiti': 1293.10, 'Curacao': 1294.77,
        'Cape Verde': 1371.11, 'New Zealand': 1275.58,
        'Bosnia-Herzegovina': 1387.22, 'Jordan': 1387.74,
    }

    print(f"[Rankings] Registros históricos: {len(rankings)}")
    return rankings, current_rankings

# ── ELO HISTÓRICO ──────────────────────────────────────────────────────
def load_elo_ratings():
    """Carrega o histórico de Elo ratings (data/raw/eloratings.csv),
    normalizado para os nomes usados em results.csv. Retorna:
      - elo_df: DataFrame com colunas country_full, rank_date, total_points
        (reutiliza o mesmo formato/índice do ranking FIFA)
      - current_elo: dict {team -> último rating conhecido}, usado como
        fallback para partidas após o fim do histórico
    """
    df = pd.read_csv('data/raw/eloratings.csv')
    df['rank_date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
    df['country_full'] = df['team'].apply(normalize_name)
    df = df.rename(columns={'rating': 'total_points'})
    df = df.dropna(subset=['total_points'])
    df = df.sort_values('rank_date')

    current_elo = df.groupby('country_full')['total_points'].last().to_dict()

    print(f"[Elo] Registros históricos: {len(df)}")
    return df[['country_full', 'rank_date', 'total_points']], current_elo

_RANKING_INDEX_CACHE = {}

def _get_ranking_index(rankings_df: pd.DataFrame) -> dict:
    """Pré-computa, por time, arrays ordenados de (rank_date, total_points)
    para permitir busca binária em get_ranking_at_date."""

    key = id(rankings_df)
    index = _RANKING_INDEX_CACHE.get(key)
    if index is None:
        index = {}
        for team, group in rankings_df.groupby('country_full'):
            g = group.sort_values('rank_date')
            index[team] = (
                g['rank_date'].values.astype('datetime64[ns]'),
                g['total_points'].values,
            )
        _RANKING_INDEX_CACHE[key] = index
    return index

def get_ranking_at_date(rankings_df: pd.DataFrame,
                         current_rankings: dict,
                         team: str,
                         date: pd.Timestamp) -> float:
    """Retorna pontos FIFA do time na data mais próxima anterior à partida."""

    index = _get_ranking_index(rankings_df)
    dates_points = index.get(team)

    if dates_points is not None:
        dates, points = dates_points
        pos = np.searchsorted(dates, np.datetime64(date), side='right')
        if pos > 0:
            return points[pos - 1]

    # Fallback para ranking atual
    return current_rankings.get(team, 1200.0)

def get_elo_at_date(elo_df: pd.DataFrame,
                     current_elo: dict,
                     team: str,
                     date: pd.Timestamp) -> float:
    """Retorna o Elo do time na data mais próxima anterior à partida,
    com fallback para o último Elo conhecido e, por fim, para a média
    global (GLOBAL_AVG_ELO), usada por seleções sem histórico."""

    index = _get_ranking_index(elo_df)
    dates_points = index.get(team)

    if dates_points is not None:
        dates, points = dates_points
        pos = np.searchsorted(dates, np.datetime64(date), side='right')
        if pos > 0:
            return points[pos - 1]

    return current_elo.get(team, GLOBAL_AVG_ELO)

# ── FEATURES POR TIME ─────────────────────────────────────────────────
def compute_team_features(df: pd.DataFrame,
                           team: str,
                           before_date: pd.Timestamp,
                           rankings_df: pd.DataFrame,
                           current_rankings: dict,
                           window: int = 20) -> dict:
    """
    Calcula features ponderadas para um time antes de uma data específica.
    Usa todas as partidas disponíveis com recency weighting,
    mas foca nas últimas `window` partidas para forma recente.
    """

    # Partidas do time antes da data
    home_matches = df[
        (df['home_team'] == team) & (df['date'] < before_date)
    ].copy()
    home_matches['is_home'] = True
    home_matches['goals_for'] = home_matches['home_score']
    home_matches['goals_against'] = home_matches['away_score']
    home_matches['won'] = (home_matches['result'] == 'H').astype(int)
    home_matches['drew'] = (home_matches['result'] == 'D').astype(int)
    home_matches['opp_team'] = home_matches['away_team']

    away_matches = df[
        (df['away_team'] == team) & (df['date'] < before_date)
    ].copy()
    away_matches['is_home'] = False
    away_matches['goals_for'] = away_matches['away_score']
    away_matches['goals_against'] = away_matches['home_score']
    away_matches['won'] = (away_matches['result'] == 'A').astype(int)
    away_matches['drew'] = (away_matches['result'] == 'D').astype(int)
    away_matches['opp_team'] = away_matches['home_team']

    all_matches = pd.concat([home_matches, away_matches]).sort_values('date')

    if len(all_matches) == 0:
        return None

    # Calcular ranking do adversário em cada partida
    all_matches['opp_points'] = all_matches.apply(
        lambda r: get_ranking_at_date(rankings_df, current_rankings,
                                       r['opp_team'], r['date']),
        axis=1
    )

    # Recency weight por partida
    all_matches['rec_w'] = all_matches['date'].apply(
        lambda d: recency_weight(d, before_date)
    )
    all_matches['total_w'] = all_matches['rec_w'] * all_matches['tournament_weight']

    # SOS adjustment para gols
    opp_avg_points = np.average(all_matches['opp_points'],
                                 weights=all_matches['total_w'])

    def sos_adjust(goals_avg, opp_pts_avg):
        normalized = (opp_pts_avg - 1200) / 700
        adjustment = 0.6 + (normalized * 0.8)
        return goals_avg * adjustment

    # ── Features globais (todas as partidas ponderadas) ──
    w = all_matches['total_w'].values

    goals_for_avg = np.average(all_matches['goals_for'], weights=w)
    goals_against_avg = np.average(all_matches['goals_against'], weights=w)
    win_rate = np.average(all_matches['won'], weights=w)
    draw_rate = np.average(all_matches['drew'], weights=w)
    btts_rate = np.average(
        ((all_matches['goals_for'] > 0) & (all_matches['goals_against'] > 0)).astype(int),
        weights=w
    )
    total_goals_avg = np.average(
        all_matches['goals_for'] + all_matches['goals_against'], weights=w
    )
    clean_sheet_rate = np.average(
        (all_matches['goals_against'] == 0).astype(int), weights=w
    )

    # ── Forma recente (últimas window partidas) ──
    recent = all_matches.tail(window)
    w_r = recent['total_w'].values

    form_goals_for = np.average(recent['goals_for'], weights=w_r) if len(recent) > 0 else goals_for_avg
    form_goals_against = np.average(recent['goals_against'], weights=w_r) if len(recent) > 0 else goals_against_avg
    form_win_rate = np.average(recent['won'], weights=w_r) if len(recent) > 0 else win_rate

    # ── Forma últimas 5 partidas ──
    last5 = all_matches.tail(5)
    w5 = last5['total_w'].values
    form5_pts = np.average(
        last5['won'] * 3 + last5['drew'], weights=w5
    ) / 3.0 if len(last5) > 0 else 0.5

    # ── Win rate por contexto ──
    home_m = all_matches[all_matches['is_home'] == True]
    away_m = all_matches[all_matches['is_home'] == False]
    neutral_m = all_matches[all_matches['neutral'] == True]

    win_rate_home = np.average(home_m['won'], weights=home_m['total_w']) if len(home_m) > 0 else win_rate
    win_rate_away = np.average(away_m['won'], weights=away_m['total_w']) if len(away_m) > 0 else win_rate
    win_rate_neutral = np.average(neutral_m['won'], weights=neutral_m['total_w']) if len(neutral_m) > 0 else win_rate

    # ── SOS adjusted ──
    sos_goals_for = sos_adjust(goals_for_avg, opp_avg_points)
    sos_goals_against = sos_adjust(goals_against_avg, opp_avg_points)
    sos_form_goals = sos_adjust(form_goals_for, opp_avg_points)

    # ── Ranking FIFA atual do time ──
    fifa_points = get_ranking_at_date(rankings_df, current_rankings, team, before_date)

    return {
        'n_matches': len(all_matches),
        'fifa_points': fifa_points,
        'goals_for_avg': goals_for_avg,
        'goals_against_avg': goals_against_avg,
        'goal_diff_avg': goals_for_avg - goals_against_avg,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'btts_rate': btts_rate,
        'total_goals_avg': total_goals_avg,
        'clean_sheet_rate': clean_sheet_rate,
        'form_goals_for': form_goals_for,
        'form_goals_against': form_goals_against,
        'form_win_rate': form_win_rate,
        'form5_pts': form5_pts,
        'win_rate_home': win_rate_home,
        'win_rate_away': win_rate_away,
        'win_rate_neutral': win_rate_neutral,
        'avg_opp_points': opp_avg_points,
        'sos_goals_for': sos_goals_for,
        'sos_goals_against': sos_goals_against,
        'sos_form_goals': sos_form_goals,
    }

def compute_h2h(df: pd.DataFrame,
                home_team: str,
                away_team: str,
                before_date: pd.Timestamp,
                n: int = 10) -> dict:
    """Histórico de confrontos diretos."""

    h2h = df[
        (
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ) & (df['date'] < before_date)
    ].tail(n)

    if len(h2h) == 0:
        return {
            'h2h_home_wins': 0.33,
            'h2h_draws': 0.33,
            'h2h_away_wins': 0.33,
            'h2h_goals_avg': 2.5,
            'h2h_n': 0,
        }

    home_wins = 0
    draws = 0
    away_wins = 0

    for _, row in h2h.iterrows():
        if row['home_team'] == home_team:
            if row['result'] == 'H': home_wins += 1
            elif row['result'] == 'D': draws += 1
            else: away_wins += 1
        else:
            if row['result'] == 'A': home_wins += 1
            elif row['result'] == 'D': draws += 1
            else: away_wins += 1

    n_games = len(h2h)
    return {
        'h2h_home_wins': home_wins / n_games,
        'h2h_draws': draws / n_games,
        'h2h_away_wins': away_wins / n_games,
        'h2h_goals_avg': h2h['total_goals'].mean(),
        'h2h_n': n_games,
    }

# ── BUILD MATCH FEATURES ──────────────────────────────────────────────
def build_match_features(df: pd.DataFrame,
                          rankings_df: pd.DataFrame,
                          current_rankings: dict) -> pd.DataFrame:

    print("[Features] Construindo features por partida...")
    print(f"[Features] Total de partidas para processar: {len(df)}")

    fm23_lookup, fm23_defaults, former_to_current = load_team_fm23_features()

    rows = []

    for i, match in df.iterrows():
        if i % 500 == 0:
            print(f"[Features] Processando partida {i}/{len(df)}...")

        home = match['home_team']
        away = match['away_team']
        date = match['date']

        home_f = compute_team_features(df, home, date, rankings_df, current_rankings)
        away_f = compute_team_features(df, away, date, rankings_df, current_rankings)

        if home_f is None or away_f is None:
            continue

        h2h = compute_h2h(df, home, away, date)

        home_fm23 = get_team_fm23(home, fm23_lookup, fm23_defaults, former_to_current)
        away_fm23 = get_team_fm23(away, fm23_lookup, fm23_defaults, former_to_current)

        # Contexto da partida
        is_neutral = int(match['neutral'])
        t_weight = match['tournament_weight']
        r_weight = match['recency_weight']

        row = {
            # Identificação
            'date': date,
            'home_team': home,
            'away_team': away,
            'tournament': match['tournament'],
            'is_neutral': is_neutral,
            'tournament_weight': t_weight,
            'sample_weight': match['sample_weight'],
            'home_elo': match['home_elo'],
            'away_elo': match['away_elo'],
            'elo_diff_abs': abs(match['home_elo'] - match['away_elo']),

            # Features home
            'home_fifa_points': home_f['fifa_points'],
            'home_goals_for': home_f['goals_for_avg'],
            'home_goals_against': home_f['goals_against_avg'],
            'home_goal_diff': home_f['goal_diff_avg'],
            'home_win_rate': home_f['win_rate'],
            'home_draw_rate': home_f['draw_rate'],
            'home_btts_rate': home_f['btts_rate'],
            'home_clean_sheet': home_f['clean_sheet_rate'],
            'home_form_goals_for': home_f['form_goals_for'],
            'home_form_goals_against': home_f['form_goals_against'],
            'home_form_win_rate': home_f['form_win_rate'],
            'home_form5_pts': home_f['form5_pts'],
            'home_win_rate_home': home_f['win_rate_home'],
            'home_win_rate_neutral': home_f['win_rate_neutral'],
            'home_avg_opp_points': home_f['avg_opp_points'],
            'home_sos_goals_for': home_f['sos_goals_for'],
            'home_sos_goals_against': home_f['sos_goals_against'],
            'home_sos_form': home_f['sos_form_goals'],
            'home_n_matches': home_f['n_matches'],
            'home_fm23_attack_strength': home_fm23['attack_strength'],
            'home_fm23_best_attacker': home_fm23['best_attacker'],
            'home_fm23_top3_attack': home_fm23['top3_attack'],
            'home_fm23_defense_strength': home_fm23['defense_strength'],
            'home_fm23_best_defender': home_fm23['best_defender'],
            'home_fm23_top5_defense': home_fm23['top5_defense'],
            'home_fm23_overall': home_fm23['overall'],
            'home_fm23_best_overall': home_fm23['best_overall'],
            'home_fm23_top11_overall': home_fm23['top11_overall'],
            'home_fm23_depth_overall': home_fm23['depth_overall'],
            'home_fm23_std_overall': home_fm23['std_overall'],
            'home_fm23_gk_strength': home_fm23['gk_strength'],

            # Features away
            'away_fifa_points': away_f['fifa_points'],
            'away_goals_for': away_f['goals_for_avg'],
            'away_goals_against': away_f['goals_against_avg'],
            'away_goal_diff': away_f['goal_diff_avg'],
            'away_win_rate': away_f['win_rate'],
            'away_draw_rate': away_f['draw_rate'],
            'away_btts_rate': away_f['btts_rate'],
            'away_clean_sheet': away_f['clean_sheet_rate'],
            'away_form_goals_for': away_f['form_goals_for'],
            'away_form_goals_against': away_f['form_goals_against'],
            'away_form_win_rate': away_f['form_win_rate'],
            'away_form5_pts': away_f['form5_pts'],
            'away_win_rate_away': away_f['win_rate_away'],
            'away_win_rate_neutral': away_f['win_rate_neutral'],
            'away_avg_opp_points': away_f['avg_opp_points'],
            'away_sos_goals_for': away_f['sos_goals_for'],
            'away_sos_goals_against': away_f['sos_goals_against'],
            'away_sos_form': away_f['sos_form_goals'],
            'away_n_matches': away_f['n_matches'],
            'away_fm23_attack_strength': away_fm23['attack_strength'],
            'away_fm23_best_attacker': away_fm23['best_attacker'],
            'away_fm23_top3_attack': away_fm23['top3_attack'],
            'away_fm23_defense_strength': away_fm23['defense_strength'],
            'away_fm23_best_defender': away_fm23['best_defender'],
            'away_fm23_top5_defense': away_fm23['top5_defense'],
            'away_fm23_overall': away_fm23['overall'],
            'away_fm23_best_overall': away_fm23['best_overall'],
            'away_fm23_top11_overall': away_fm23['top11_overall'],
            'away_fm23_depth_overall': away_fm23['depth_overall'],
            'away_fm23_std_overall': away_fm23['std_overall'],
            'away_fm23_gk_strength': away_fm23['gk_strength'],

            # Diferenciais
            'diff_fifa_points': home_f['fifa_points'] - away_f['fifa_points'],
            'diff_goals_for': home_f['goals_for_avg'] - away_f['goals_for_avg'],
            'diff_goals_against': home_f['goals_against_avg'] - away_f['goals_against_avg'],
            'diff_win_rate': home_f['win_rate'] - away_f['win_rate'],
            'diff_form_win_rate': home_f['form_win_rate'] - away_f['form_win_rate'],
            'diff_form5': home_f['form5_pts'] - away_f['form5_pts'],
            'diff_sos_goals': home_f['sos_goals_for'] - away_f['sos_goals_for'],
            'diff_sos_form': home_f['sos_form_goals'] - away_f['sos_form_goals'],
            'diff_avg_opp': home_f['avg_opp_points'] - away_f['avg_opp_points'],
            'fm23_attack_diff': home_fm23['attack_strength'] - away_fm23['attack_strength'],
            'fm23_defense_diff': home_fm23['defense_strength'] - away_fm23['defense_strength'],
            'fm23_overall_diff': home_fm23['overall'] - away_fm23['overall'],
            'fm23_best_overall_diff': home_fm23['best_overall'] - away_fm23['best_overall'],
            'fm23_top3_attack_diff': home_fm23['top3_attack'] - away_fm23['top3_attack'],
            'fm23_top5_defense_diff': home_fm23['top5_defense'] - away_fm23['top5_defense'],

            # H2H
            'h2h_home_wins': h2h['h2h_home_wins'],
            'h2h_draws': h2h['h2h_draws'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'h2h_goals_avg': h2h['h2h_goals_avg'],
            'h2h_n': h2h['h2h_n'],

            # Targets
            'result': match['result'],
            'total_goals': match['total_goals'],
            'btts': match['btts'],
            'home_score': match['home_score'],
            'away_score': match['away_score'],
        }

        rows.append(row)

    features_df = pd.DataFrame(rows)
    print(f"[Features] Features construídas: {len(features_df)} partidas")
    print(f"[Features] Colunas: {len(features_df.columns)}")

    return features_df

# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)

    print("=== CARREGANDO DADOS ===")
    elo_df, current_elo = load_elo_ratings()
    df = load_results(elo_df, current_elo)
    rankings_df, current_rankings = load_fifa_rankings()

    print("\n=== CONSTRUINDO FEATURES ===")
    features = build_match_features(df, rankings_df, current_rankings)

    # Salvar
    features.to_csv('data/processed/match_features_v2.csv', index=False)
    print(f"\n[OK] Salvo em data/processed/match_features_v2.csv")
    print(f"[OK] Shape: {features.shape}")
    print(f"\nDistribuição de resultados:")
    print(features['result'].value_counts())
    print(f"\nEstatísticas de gols:")
    print(features['total_goals'].describe())
