import sys
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

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
    'FW': 0.12,   # atacante marca em ~12% das partidas
    'MF': 0.05,   # meia marca em ~5%
    'DF': 0.02,   # defensor marca em ~2%
    'GK': 0.01,   # goleiro quase nunca
}

# Nomes que aparecem em results.csv/goalscorers.csv com grafia diferente em eloratings.csv
ELO_NAME_ALIASES = {
    'Czech Republic': 'Czechia',
    'DR Congo': 'Democratic Republic of Congo',
}

# Atributos FM23 (data/processed/fm23_player_attributes.csv) usados como features.
FM23_ATTRS = [
    'Fin', 'OtB', 'Com', 'Dec', 'Pac', 'Acc', 'Hea', 'Pen',
    'Dri', 'Str', 'Vis', 'Ant', 'Fla', 'Lon',
]

FM23_ATTRS_PATH = 'data/processed/fm23_player_attributes.csv'

# Nomes em ingles (usados pelo modelo / eloratings.csv) -> nomes em
# portugues cadastrados na tabela `teams` do Supabase.
TEAM_NAME_PT = {
    "Algeria": "Argélia",
    "Argentina": "Argentina",
    "Australia": "Austrália",
    "Austria": "Áustria",
    "Belgium": "Bélgica",
    "Bosnia-Herzegovina": "Bósnia",
    "Brazil": "Brasil",
    "Canada": "Canadá",
    "Cape Verde": "Cabo Verde",
    "Colombia": "Colômbia",
    "Croatia": "Croácia",
    "Curacao": "Curaçao",
    "Czech Republic": "Tchéquia",
    "DR Congo": "Rep. D. Congo",
    "Ecuador": "Equador",
    "Egypt": "Egito",
    "England": "Inglaterra",
    "France": "França",
    "Germany": "Alemanha",
    "Ghana": "Gana",
    "Haiti": "Haiti",
    "Iran": "Irã",
    "Iraq": "Iraque",
    "Ivory Coast": "Costa do Marfim",
    "Japan": "Japão",
    "Jordan": "Jordânia",
    "Mexico": "México",
    "Morocco": "Marrocos",
    "Netherlands": "Holanda",
    "New Zealand": "Nova Zelândia",
    "Norway": "Noruega",
    "Panama": "Panamá",
    "Paraguay": "Paraguai",
    "Portugal": "Portugal",
    "Qatar": "Catar",
    "Saudi Arabia": "Arábia Saudita",
    "Scotland": "Escócia",
    "Senegal": "Senegal",
    "South Africa": "África do Sul",
    "South Korea": "Coreia do Sul",
    "Spain": "Espanha",
    "Sweden": "Suécia",
    "Switzerland": "Suíça",
    "Tunisia": "Tunísia",
    "Turkey": "Turquia",
    "United States": "Estados Unidos",
    "Uruguay": "Uruguai",
    "Uzbekistan": "Uzbequistão",
}

# Nomes cadastrados no Supabase -> nomes canonicos usados em scorer_features
# (mesma grafia que aparece na coluna `scorer` deste dataset, apos normalize_name).
SUPABASE_NAME_TO_SCORER_NAME = {
    'Cuti Romero': 'Cristian Romero',
    'Otamendi': 'Nicolás Otamendi',
    'Nico González': 'Nicolás González',
    'Vinícius Jr.': 'Vinícius Júnior',
    'Vini Jr.': 'Vinícius Júnior',
    'Dibu Martínez': 'Emiliano Martínez',
    'Hwang Heechan': 'Hwang Hee-chan',
    'Lee Jaesung': 'Lee Jae-sung',
    'Lee Kangin': 'Lee Kang-in',
    'Son Heungmin': 'Son Heung-min',
    'Edin Dzeko': 'Edin Džeko',
    'AlMoez Ali': 'Almoez Ali',
    'Ricardo Rodriguez': 'Ricardo Rodríguez',
    'Hakan Calhanoglu': 'Hakan Çalhanoğlu',
    'Arda Guler': 'Arda Güler',
    'Kerem Akturkoglu': 'Kerem Aktürkoğlu',
    'Ritsu Doan': 'Ritsu Dōan',
    'Jeremy Doku': 'Jérémy Doku',
    'Salem Al Dawsari': 'Salem Al-Dawsari',
    'Saleh Al Shehri': 'Saleh Al-Shehri',
    'Giorgian De Arrascaeta': 'Giorgian de Arrascaeta',
    'Alexander Sorloth': 'Alexander Sørloth',
    'Ismaila Sarr': 'Ismaïla Sarr',
    'Luka Modric': 'Luka Modrić',
    'Mario Pasalic': 'Mario Pašalić',
    'Nikola Vlasic': 'Nikola Vlašić',
    'Ivan Perisic': 'Ivan Perišić',
    'Andrej Kramaric': 'Andrej Kramarić',
}

def player_recency_weight(match_date: pd.Timestamp,
                           reference: pd.Timestamp) -> float:
    days_ago = (reference - match_date).days
    return max(0.15, np.exp(-0.0004 * days_ago))

def normalize_name(name: str) -> str:
    return MANUAL_MAPPING.get(name, name)

def build_elo_index(elo_df: pd.DataFrame) -> dict:
    """Pré-indexa o Elo por seleção: team -> (datas ordenadas, ratings).

    Ignora linhas com rating ausente (ex.: Moldova em eloratings.csv tem
    `rating` vazio em quase todo o histórico) para nunca retornar NaN.
    """
    elo_valid = elo_df[elo_df['rating'].notna()]
    index = {}
    for team, group in elo_valid.groupby('team'):
        group = group.sort_values('date', kind='stable')
        index[team] = (group['date'].values, group['rating'].values)
    return index

def _lookup_elo(arr, date: pd.Timestamp):
    if arr is None:
        return None
    dates, ratings = arr
    pos = np.searchsorted(dates, np.datetime64(date), side='right')
    if pos == 0:
        return None
    return ratings[pos - 1]

def get_elo_at_date(elo_index: dict,
                    team: str,
                    date: pd.Timestamp) -> float:
    """Retorna Elo do time na data mais próxima anterior, via índice pré-construído."""
    rating = _lookup_elo(elo_index.get(team), date)
    if rating is None:
        alias = ELO_NAME_ALIASES.get(team)
        if alias:
            rating = _lookup_elo(elo_index.get(alias), date)
    return rating if rating is not None else 1500.0  # default neutro

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

def build_scorer_match_index(scorers: pd.DataFrame) -> dict:
    """Indexa gols por (scorer, date, home_team, away_team) -> (n_goals, is_penalty)."""
    index = {}
    for key, group in scorers.groupby(['scorer', 'date', 'home_team', 'away_team']):
        index[key] = (len(group), bool(group['penalty'].any()))
    return index

def build_team_index(results: pd.DataFrame) -> tuple[dict, dict]:
    """Pré-indexa as partidas na perspectiva de cada seleção (uma linha por
    time/partida), ordenadas por data:
    - team_matches_index: team -> DataFrame de partidas (date, opp_team,
      is_home, neutral, tournament)
    - team_conceded_index: team -> (datas ordenadas, gols sofridos)
    """
    cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral', 'tournament']

    home = results[cols].rename(columns={
        'home_team': 'team', 'away_team': 'opp_team',
        'home_score': 'team_score', 'away_score': 'opp_score',
    })
    home['is_home'] = True

    away = results[cols].rename(columns={
        'away_team': 'team', 'home_team': 'opp_team',
        'away_score': 'team_score', 'home_score': 'opp_score',
    })
    away['is_home'] = False

    long_df = pd.concat([home, away], ignore_index=True)

    team_matches_index = {}
    team_conceded_index = {}
    for team, group in long_df.groupby('team'):
        group = group.sort_values('date', kind='stable').reset_index(drop=True)
        team_matches_index[team] = group
        team_conceded_index[team] = (group['date'].values, group['opp_score'].values.astype(float))

    return team_matches_index, team_conceded_index

def opp_conceded_stats(team_conceded_index: dict,
                        opp_team: str,
                        date: pd.Timestamp) -> tuple[float, float]:
    """Gols sofridos médios e taxa de clean sheet do adversário nas últimas
    10 partidas anteriores a `date`."""
    arr = team_conceded_index.get(opp_team)
    if arr is None:
        return 0.0, 0.25

    dates, conceded = arr
    pos = np.searchsorted(dates, np.datetime64(date), side='left')
    window = conceded[max(0, pos - 10):pos]

    if len(window) == 0:
        return 0.0, 0.25

    return float(window.mean()), float((window == 0).mean())

def get_player_history(team_matches_index: dict,
                        elo_index: dict,
                        debut_dates: dict,
                        scorer_match_index: dict,
                        scorer_name: str,
                        team_name: str) -> pd.DataFrame:
    """
    Constrói histórico completo do jogador:
    - Partidas onde marcou (target=1) do goalscorers
    - Partidas onde não marcou (target=0) do results
    A partir da primeira aparição no goalscorers.
    """

    debut_date = debut_dates.get(scorer_name)
    if debut_date is None:
        return pd.DataFrame()

    team_matches_all = team_matches_index.get(team_name)
    if team_matches_all is None:
        return pd.DataFrame()

    # Todas as partidas da seleção após a estreia
    team_matches = team_matches_all[team_matches_all['date'] >= debut_date]
    if len(team_matches) == 0:
        return pd.DataFrame()

    rows = []
    for _, match in team_matches.iterrows():
        date = match['date']
        opp_team = match['opp_team']
        is_home = bool(match['is_home'])
        home = team_name if is_home else opp_team
        away = opp_team if is_home else team_name

        # Verificar se marcou nessa partida
        n_goals, has_penalty = scorer_match_index.get(
            (scorer_name, date, home, away), (0, False)
        )
        target = 1 if n_goals > 0 else 0
        is_penalty = int(has_penalty) if n_goals > 0 else 0

        # Elo do adversário
        opp_elo = get_elo_at_date(elo_index, opp_team, date)
        team_elo = get_elo_at_date(elo_index, team_name, date)
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
            'scoring_rate_last_12m': POSITION_BASELINE.get(position, 0.05),
            'scoring_rate_last_24m': POSITION_BASELINE.get(position, 0.05),
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

    # Taxa de gol recente (com shrinkage bayesiano em direção à taxa de
    # carreira, pseudo-contagem alpha=3). Com p~0.085-0.15 por partida,
    # P(0 gols em 5/10 jogos seguidos) é alta mesmo para artilheiros de
    # elite, e sem shrinkage essas janelas zeram completamente o sinal
    # para jogadores como Havertz/Musiala (recent5=recent10=0.0 apesar de
    # scoring_rate de carreira ~0.13-0.15).
    RECENT_SHRINKAGE_ALPHA = 3

    recent10 = past.tail(10)
    w10 = recent10['recency_weight'].values
    raw_recent10 = np.average(recent10['target'], weights=w10) if len(recent10) > 0 else scoring_rate
    scoring_rate_recent10 = (raw_recent10 * 10 + RECENT_SHRINKAGE_ALPHA * scoring_rate) / (10 + RECENT_SHRINKAGE_ALPHA)

    recent5 = past.tail(5)
    w5 = recent5['recency_weight'].values
    raw_recent5 = np.average(recent5['target'], weights=w5) if len(recent5) > 0 else scoring_rate
    scoring_rate_recent5 = (raw_recent5 * 5 + RECENT_SHRINKAGE_ALPHA * scoring_rate) / (5 + RECENT_SHRINKAGE_ALPHA)

    # Taxa de gol em janelas temporais (12/24 meses), em vez de janelas por
    # quantidade de partidas: goalscorers.csv só registra partidas onde o
    # jogador marcou, então "últimas N partidas" do histórico (que inclui
    # todas as partidas da seleção) ainda é válido, mas com N pequeno (5/10)
    # a taxa zera com frequência só por acaso (p ~ 0.85^5 ~ 44% de chance de
    # 5 partidas sem gol mesmo para um artilheiro prolífico). Uma janela
    # temporal mais larga reduz esse ruído.
    last_12m_cutoff = before_date - pd.Timedelta(days=365)
    last_24m_cutoff = before_date - pd.Timedelta(days=730)

    last_12m = past[past['date'] >= last_12m_cutoff]
    w12 = last_12m['recency_weight'].values
    scoring_rate_last_12m = np.average(last_12m['target'], weights=w12) if len(last_12m) > 0 else scoring_rate

    last_24m = past[past['date'] >= last_24m_cutoff]
    w24 = last_24m['recency_weight'].values
    scoring_rate_last_24m = np.average(last_24m['target'], weights=w24) if len(last_24m) > 0 else scoring_rate

    # Taxa ajustada por nível do adversário (SOS)
    # Normalizar Elo do adversário para peso
    opp_elo_norm = (past['opp_elo'] - 1400) / 600  # ~0 a 1
    sos_weights = (w * (0.5 + opp_elo_norm * 0.5)).values
    valid = ~np.isnan(sos_weights)
    if valid.any() and sos_weights[valid].sum() > 0:
        sos_scoring_rate = np.average(past['target'].values[valid], weights=sos_weights[valid])
    else:
        sos_scoring_rate = scoring_rate

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
        'scoring_rate_last_12m': scoring_rate_last_12m,
        'scoring_rate_last_24m': scoring_rate_last_24m,
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

def load_fm23_lookup() -> tuple[dict, dict]:
    """Carrega data/processed/fm23_player_attributes.csv e monta:
    - lookup: (scorer_name, team_en) -> dict de atributos FM23
    - global_median: dict com a mediana global dos atributos (jogadores com match real)

    `scorer_name` e `team_en` usam a mesma grafia da coluna `scorer`/`team`
    de scorer_features_v1.csv, para permitir o merge direto no
    build_scorer_dataset.
    """
    attrs = pd.read_csv(FM23_ATTRS_PATH)

    attrs['scorer_name'] = attrs['supabase_name'].map(
        lambda n: SUPABASE_NAME_TO_SCORER_NAME.get(n, n)
    )
    team_name_en = {pt: en for en, pt in TEAM_NAME_PT.items()}
    attrs['team_en'] = attrs['supabase_team'].map(
        lambda t: team_name_en.get(t, t)
    )

    real_attrs = attrs[attrs['source'].isin(['matched', 'proxy'])]
    global_median = real_attrs[FM23_ATTRS].median().to_dict()

    attrs = attrs.drop_duplicates(subset=['scorer_name', 'team_en'])
    lookup = {
        (row['scorer_name'], row['team_en']): {col: row[col] for col in FM23_ATTRS}
        for _, row in attrs.iterrows()
    }
    return lookup, global_median


def get_fm23_attributes(lookup: dict,
                         global_median: dict,
                         scorer_name: str,
                         team_name: str) -> dict:
    """Atributos FM23 do jogador; usa mediana global se não encontrado."""
    return lookup.get((scorer_name, team_name), global_median)


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

    fm23_lookup, fm23_global_median = load_fm23_lookup()
    print(f"[FM23] {len(fm23_lookup)} jogadores com atributos mapeados")

    # Índices pré-construídos para evitar refiltrar os dataframes inteiros
    # a cada partida do histórico de cada jogador.
    debut_dates = scorers.groupby('scorer')['date'].min().to_dict()
    scorer_match_index = build_scorer_match_index(scorers)
    elo_index = build_elo_index(elo)
    team_matches_index, team_conceded_index = build_team_index(results)

    all_rows = []

    for i, (_, player_row) in enumerate(active_players.iterrows()):
        scorer_name = player_row['scorer']
        team_name = player_row['team']

        if i % 50 == 0:
            print(f"[Scorers] Processando jogador {i}/{len(active_players)}: {scorer_name}")

        # Histórico completo do jogador
        history = get_player_history(
            team_matches_index, elo_index, debut_dates, scorer_match_index,
            scorer_name, team_name,
        )
        if len(history) == 0:
            continue

        # Coverage da seleção para normalização
        coverage = team_coverage.get(team_name, 0.5)

        # Atributos FM23 do jogador (constantes ao longo do histórico)
        fm23_attrs = get_fm23_attributes(fm23_lookup, fm23_global_median, scorer_name, team_name)

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

            # Taxa de gols sofridos / clean sheet do adversário (últimas 10 partidas)
            opp_goals_conceded, clean_sheet_rate = opp_conceded_stats(
                team_conceded_index, match_row['opp_team'], date
            )

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
                'scoring_rate_last_12m': player_f['scoring_rate_last_12m'],
                'scoring_rate_last_24m': player_f['scoring_rate_last_24m'],
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
            row.update(fm23_attrs)
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
