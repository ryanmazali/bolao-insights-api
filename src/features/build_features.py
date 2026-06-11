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
}

def normalize_name(name: str) -> str:
    return NAME_MAPPING.get(name, name)

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
def load_results() -> pd.DataFrame:
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
    df['sample_weight'] = df['tournament_weight'] * df['recency_weight']

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
    df = load_results()
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
