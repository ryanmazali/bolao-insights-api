import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

CUTOFF_DATE = '2015-01-01'
REFERENCE_DATE = pd.Timestamp('2026-06-10')

# Mapeamento manual de nomes
MANUAL_MAPPING = {
    'Cuti Romero': 'Cristian Romero',
    'Otamendi': 'Nicolás Otamendi',
    'Nico González': 'Nicolás González',
    'Vinícius Jr.': 'Vinícius Júnior',
    'Emiliano Martínez': 'Emiliano Martínez',
    'Dibu Martínez': 'Emiliano Martínez',
    'Alexis Mac Allister': 'Alexis Mac Allister',
    'China PR': 'China',
}

POSITION_BASELINE = {
    'FW': 0.35,   # atacante marca em ~35% das partidas
    'MF': 0.12,   # meia marca em ~12%
    'DF': 0.04,   # defensor marca em ~4%
    'GK': 0.001,  # goleiro quase nunca
}

def player_recency_weight(match_date: pd.Timestamp,
                           reference: pd.Timestamp) -> float:
    days_ago = (reference - match_date).days
    return max(0.15, np.exp(-0.0004 * days_ago))

def normalize_name(name: str) -> str:
    return MANUAL_MAPPING.get(name, name)

def get_elo_at_date(elo_df: pd.DataFrame,
                    team: str,
                    date: pd.Timestamp) -> float:
    """Retorna Elo do time na data mais próxima anterior."""
    mask = (elo_df['team'] == team) & (elo_df['date'] <= date)
    subset = elo_df[mask]
    if len(subset) == 0:
        return 1500.0  # default neutro
    return subset.iloc[-1]['rating']

def elo_tier(elo_rating: float) -> str:
    """Classifica adversário por tier de Elo."""
    if elo_rating >= 1900: return 'elite'      # top ~5
    if elo_rating >= 1800: return 'strong'     # top ~15
    if elo_rating >= 1700: return 'mid'        # top ~30
    if elo_rating >= 1600: return 'below_mid'  # top ~60
    return 'weak'                               # resto

def load_data():
    """Carrega e prepara todos os datasets."""

    # Results
    results = pd.read_csv('data/raw/results.csv', parse_dates=['date'])
    results = results[results['date'] >= CUTOFF_DATE]
    results = results.dropna(subset=['home_score', 'away_score'])
    results['home_team'] = results['home_team'].apply(normalize_name)
    results['away_team'] = results['away_team'].apply(normalize_name)

    # Scorers
    scorers = pd.read_csv('data/raw/goalscorers.csv', parse_dates=['date'])
    scorers = scorers[scorers['date'] >= CUTOFF_DATE]
    scorers = scorers[scorers['own_goal'] == False]
    scorers['scorer'] = scorers['scorer'].apply(normalize_name)
    scorers['home_team'] = scorers['home_team'].apply(normalize_name)
    scorers['away_team'] = scorers['away_team'].apply(normalize_name)
    scorers['team'] = scorers['team'].apply(normalize_name)

    # Elo
    elo = pd.read_csv('data/raw/eloratings.csv')
    elo['date'] = pd.to_datetime(elo['date'], format='mixed', dayfirst=False)
    elo = elo.sort_values('date').reset_index(drop=True)

    print(f"[Data] Results: {len(results)} partidas")
    print(f"[Data] Scorers: {len(scorers)} gols")
    print(f"[Data] Elo: {len(elo)} registros")

    return results, scorers, elo

def get_player_history(scorers: pd.DataFrame,
                        results: pd.DataFrame,
                        elo: pd.DataFrame,
                        scorer_name: str,
                        team_name: str) -> pd.DataFrame:
    """
    Constrói histórico completo do jogador:
    - Partidas onde marcou (target=1) do goalscorers
    - Partidas onde não marcou (target=0) do results
    A partir da primeira aparição no goalscorers.
    """

    # Primeira aparição do jogador
    player_goals = scorers[scorers['scorer'] == scorer_name]
    if len(player_goals) == 0:
        return pd.DataFrame()

    debut_date = player_goals['date'].min()

    # Todas as partidas da seleção após a estreia
    team_matches = results[
        (
            (results['home_team'] == team_name) |
            (results['away_team'] == team_name)
        ) &
        (results['date'] >= debut_date)
    ].copy()

    if len(team_matches) == 0:
        return pd.DataFrame()

    rows = []
    for _, match in team_matches.iterrows():
        date = match['date']
        home = match['home_team']
        away = match['away_team']
        is_home = (home == team_name)
        opp_team = away if is_home else home

        # Verificar se marcou nessa partida
        scored = scorers[
            (scorers['scorer'] == scorer_name) &
            (scorers['date'] == date) &
            (scorers['home_team'] == home) &
            (scorers['away_team'] == away)
        ]
        target = 1 if len(scored) > 0 else 0
        n_goals = len(scored)
        is_penalty = int(scored['penalty'].any()) if len(scored) > 0 else 0

        # Elo do adversário
        opp_elo = get_elo_at_date(elo, opp_team, date)
        team_elo = get_elo_at_date(elo, team_name, date)
        tier = elo_tier(opp_elo)

        # Pesos
        rec_w = player_recency_weight(date, REFERENCE_DATE)

        rows.append({
            'date': date,
            'scorer': scorer_name,
            'team': team_name,
            'opp_team': opp_team,
            'target': target,
            'n_goals': n_goals,
            'is_penalty': is_penalty,
            'opp_elo': opp_elo,
            'team_elo': team_elo,
            'elo_diff': team_elo - opp_elo,
            'opp_tier': tier,
            'is_neutral': int(match['neutral']),
            'tournament': match['tournament'],
            'recency_weight': rec_w,
        })

    return pd.DataFrame(rows)

def compute_player_features(history: pd.DataFrame,
                              before_date: pd.Timestamp,
                              position: str) -> dict:
    """
    Calcula features do jogador baseadas no histórico até before_date.
    """
    past = history[history['date'] < before_date].copy()

    if len(past) == 0:
        # Sem histórico — usar baseline da posição
        return {
            'n_matches': 0,
            'scoring_rate': POSITION_BASELINE.get(position, 0.05),
            'scoring_rate_recent10': POSITION_BASELINE.get(position, 0.05),
            'scoring_rate_recent5': POSITION_BASELINE.get(position, 0.05),
            'sos_scoring_rate': POSITION_BASELINE.get(position, 0.05),
            'goals_vs_elite': 0.0,
            'goals_vs_strong': 0.0,
            'goals_vs_mid': POSITION_BASELINE.get(position, 0.05),
            'goals_vs_weak': POSITION_BASELINE.get(position, 0.05) * 1.5,
            'penalty_rate': 0.0,
            'is_penalty_taker': 0,
            'avg_goals_per_game': 0.0,
            'position_weight': {'FW': 1.0, 'MF': 0.5, 'DF': 0.15, 'GK': 0.01}.get(position, 0.1),
        }

    w = past['recency_weight'].values

    # Taxa de gol global ponderada
    scoring_rate = np.average(past['target'], weights=w)

    # Taxa de gol recente
    recent10 = past.tail(10)
    w10 = recent10['recency_weight'].values
    scoring_rate_recent10 = np.average(recent10['target'], weights=w10) if len(recent10) > 0 else scoring_rate

    recent5 = past.tail(5)
    w5 = recent5['recency_weight'].values
    scoring_rate_recent5 = np.average(recent5['target'], weights=w5) if len(recent5) > 0 else scoring_rate

    # Taxa ajustada por nível do adversário (SOS)
    # Normalizar Elo do adversário para peso
    opp_elo_norm = (past['opp_elo'] - 1400) / 600  # ~0 a 1
    sos_weights = w * (0.5 + opp_elo_norm * 0.5)
    sos_scoring_rate = np.average(past['target'], weights=sos_weights) if sos_weights.sum() > 0 else scoring_rate

    # Taxa por tier de adversário
    def tier_rate(tier):
        subset = past[past['opp_tier'] == tier]
        if len(subset) == 0:
            return np.nan
        return subset['target'].mean()

    goals_vs_elite  = tier_rate('elite')
    goals_vs_strong = tier_rate('strong')
    goals_vs_mid    = tier_rate('mid')
    goals_vs_weak   = tier_rate('weak')

    # Preencher NaN com scoring_rate global
    goals_vs_elite  = goals_vs_elite  if not np.isnan(goals_vs_elite)  else scoring_rate * 0.5
    goals_vs_strong = goals_vs_strong if not np.isnan(goals_vs_strong) else scoring_rate * 0.7
    goals_vs_mid    = goals_vs_mid    if not np.isnan(goals_vs_mid)    else scoring_rate
    goals_vs_weak   = goals_vs_weak   if not np.isnan(goals_vs_weak)   else scoring_rate * 1.4

    # Pênaltis
    penalty_goals = past[past['is_penalty'] == 1]
    penalty_rate = len(penalty_goals) / len(past) if len(past) > 0 else 0.0
    is_penalty_taker = int(penalty_rate > 0.05)

    # Média de gols por jogo (não binário)
    avg_goals = np.average(past['n_goals'], weights=w)

    position_weight = {'FW': 1.0, 'MF': 0.5, 'DF': 0.15, 'GK': 0.01}.get(position, 0.1)

    return {
        'n_matches': len(past),
        'scoring_rate': scoring_rate,
        'scoring_rate_recent10': scoring_rate_recent10,
        'scoring_rate_recent5': scoring_rate_recent5,
        'sos_scoring_rate': sos_scoring_rate,
        'goals_vs_elite': goals_vs_elite,
        'goals_vs_strong': goals_vs_strong,
        'goals_vs_mid': goals_vs_mid,
        'goals_vs_weak': goals_vs_weak,
        'penalty_rate': penalty_rate,
        'is_penalty_taker': is_penalty_taker,
        'avg_goals_per_game': avg_goals,
        'position_weight': position_weight,
    }

def build_scorer_dataset(results: pd.DataFrame,
                          scorers: pd.DataFrame,
                          elo: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói dataset completo:
    Para cada partida, para cada jogador ativo da seleção:
    - target = 1 se marcou, 0 se não marcou
    """

    # Jogadores com 5+ partidas com gol
    player_counts = scorers.groupby(['scorer', 'team'])['date'].nunique()
    active_players = player_counts[player_counts >= 5].reset_index()
    active_players.columns = ['scorer', 'team', 'n_scoring_matches']

    print(f"[Scorers] Jogadores ativos (5+ partidas): {len(active_players)}")

    # Cobertura por seleção (para normalização do viés UEFA)
    team_results_count = pd.concat([
        results[['date', 'home_team']].rename(columns={'home_team': 'team'}),
        results[['date', 'away_team']].rename(columns={'away_team': 'team'}),
    ]).drop_duplicates().groupby('team').size()

    team_scorers_count = scorers.groupby('team')['date'].nunique()
    team_coverage = (team_scorers_count / team_results_count).fillna(0.5).clip(0.1, 1.0)

    print(f"[Coverage] Cobertura média: {team_coverage.mean():.2f}")

    all_rows = []

    for i, (_, player_row) in enumerate(active_players.iterrows()):
        scorer_name = player_row['scorer']
        team_name = player_row['team']

        if i % 50 == 0:
            print(f"[Scorers] Processando jogador {i}/{len(active_players)}: {scorer_name}")

        # Histórico completo do jogador
        history = get_player_history(scorers, results, elo, scorer_name, team_name)
        if len(history) == 0:
            continue

        # Coverage da seleção para normalização
        coverage = team_coverage.get(team_name, 0.5)

        # Para cada partida no histórico, construir features
        for _, match_row in history.iterrows():
            date = match_row['date']

            # Features do jogador até essa data
            player_f = compute_player_features(
                history, date, position='FW'  # posição será corrigida no merge com Supabase
            )

            # Contexto da partida
            opp_elo = match_row['opp_elo']
            team_elo = match_row['team_elo']
            opp_tier = match_row['opp_tier']

            # Taxa de gols sofridos pelo adversário (proxy)
            opp_matches = results[
                (
                    (results['home_team'] == match_row['opp_team']) |
                    (results['away_team'] == match_row['opp_team'])
                ) &
                (results['date'] < date)
            ].tail(10)

            opp_goals_conceded = 0.0
            if len(opp_matches) > 0:
                opp_home = opp_matches[opp_matches['away_team'] == match_row['opp_team']]['home_score']
                opp_away = opp_matches[opp_matches['home_team'] == match_row['opp_team']]['away_score']
                all_conceded = pd.concat([opp_home, opp_away])
                opp_goals_conceded = all_conceded.mean() if len(all_conceded) > 0 else 1.2

            opp_clean_sheets = results[
                (
                    (results['home_team'] == match_row['opp_team']) |
                    (results['away_team'] == match_row['opp_team'])
                ) &
                (results['date'] < date)
            ].tail(10)

            clean_sheet_rate = 0.25  # default
            if len(opp_clean_sheets) > 0:
                home_cs = (opp_clean_sheets['away_team'] == match_row['opp_team']) & (opp_clean_sheets['home_score'] == 0)
                away_cs = (opp_clean_sheets['home_team'] == match_row['opp_team']) & (opp_clean_sheets['away_score'] == 0)
                clean_sheet_rate = (home_cs | away_cs).mean()

            row = {
                # Identificação
                'date': date,
                'scorer': scorer_name,
                'team': team_name,
                'opp_team': match_row['opp_team'],

                # Features do jogador
                'n_matches': player_f['n_matches'],
                'scoring_rate': player_f['scoring_rate'],
                'scoring_rate_recent10': player_f['scoring_rate_recent10'],
                'scoring_rate_recent5': player_f['scoring_rate_recent5'],
                'sos_scoring_rate': player_f['sos_scoring_rate'],
                'goals_vs_elite': player_f['goals_vs_elite'],
                'goals_vs_strong': player_f['goals_vs_strong'],
                'goals_vs_mid': player_f['goals_vs_mid'],
                'goals_vs_weak': player_f['goals_vs_weak'],
                'penalty_rate': player_f['penalty_rate'],
                'is_penalty_taker': player_f['is_penalty_taker'],
                'avg_goals_per_game': player_f['avg_goals_per_game'],
                'position_weight': player_f['position_weight'],

                # Contexto da partida
                'opp_elo': opp_elo,
                'team_elo': team_elo,
                'elo_diff': team_elo - opp_elo,
                'opp_tier_elite': int(opp_tier == 'elite'),
                'opp_tier_strong': int(opp_tier == 'strong'),
                'opp_tier_mid': int(opp_tier == 'mid'),
                'opp_tier_weak': int(opp_tier == 'weak'),
                'opp_goals_conceded_avg': opp_goals_conceded,
                'opp_clean_sheet_rate': clean_sheet_rate,
                'team_coverage': coverage,
                'is_neutral': match_row['is_neutral'],
                'recency_weight': match_row['recency_weight'],

                # Target
                'target': match_row['target'],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"\n[Scorers] Dataset construído: {len(df)} linhas")
    print(f"[Scorers] Taxa de gol média: {df['target'].mean():.3f}")
    print(f"[Scorers] Jogadores únicos: {df['scorer'].nunique()}")
    return df

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    results, scorers, elo = load_data()
    df = build_scorer_dataset(results, scorers, elo)
    df.to_csv('data/processed/scorer_features_v1.csv', index=False)
    print(f"\n[OK] Salvo em data/processed/scorer_features_v1.csv")
    print(f"[OK] Shape: {df.shape}")
