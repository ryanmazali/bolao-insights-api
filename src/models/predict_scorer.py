import joblib
import json
import numpy as np
import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Carregar modelos
MODEL_DIR = 'src/models/saved'
model_scorer = joblib.load(f'{MODEL_DIR}/model_scorer_v1.pkl')

with open(f'{MODEL_DIR}/scorer_feature_columns_v1.json') as f:
    SCORER_FEATURES = json.load(f)

with open(f'{MODEL_DIR}/scorer_team_map.json', encoding='utf-8') as f:
    SCORER_TEAM_MAP = json.load(f)

# Cliente Supabase
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Mapeamento nomes Supabase -> CSV
MANUAL_MAPPING = {
    'Cuti Romero': 'Cristian Romero',
    'Otamendi': 'Nicolás Otamendi',
    'Nico González': 'Nicolás González',
    'Vinícius Jr.': 'Vinícius Júnior',
    'Vini Jr.': 'Vinícius Júnior',
    'Emiliano Martínez': 'Emiliano Martínez',
    'Dibu Martínez': 'Emiliano Martínez',
}

# Carregar features históricas dos jogadores
_scorer_features_cache = None

def get_scorer_features_df():
    global _scorer_features_cache
    if _scorer_features_cache is None:
        _scorer_features_cache = pd.read_csv(
            'data/processed/scorer_features_v1.csv',
            parse_dates=['date']
        )
    return _scorer_features_cache

def get_convocados(team_name_en: str) -> list:
    """Busca jogadores convocados do Supabase pelo nome da seleção em inglês."""

    # Mapeamento inglês -> português (nome no Supabase)
    TEAM_NAME_PT = {
        'Brazil': 'Brasil',
        'Argentina': 'Argentina',
        'France': 'França',
        'England': 'Inglaterra',
        'Germany': 'Alemanha',
        'Spain': 'Espanha',
        'Portugal': 'Portugal',
        'Netherlands': 'Holanda',
        'Morocco': 'Marrocos',
        'Japan': 'Japão',
        'Colombia': 'Colômbia',
        'Uruguay': 'Uruguai',
        'Belgium': 'Bélgica',
        'Croatia': 'Croácia',
        'Switzerland': 'Suíça',
        'Mexico': 'México',
        'USA': 'Estados Unidos',
        'Ecuador': 'Equador',
        'Senegal': 'Senegal',
        'South Korea': 'Coreia do Sul',
        'Australia': 'Austrália',
        'Canada': 'Canadá',
        'Turkey': 'Turquia',
        'Denmark': 'Dinamarca',
        'Norway': 'Noruega',
        'Austria': 'Áustria',
        'Serbia': 'Sérvia',
        'Poland': 'Polônia',
        'Tunisia': 'Tunísia',
        'Ghana': 'Gana',
        'Cameroon': 'Camarões',
        'Egypt': 'Egito',
        'Saudi Arabia': 'Arábia Saudita',
        'Iran': 'Irã',
        'Scotland': 'Escócia',
        'Sweden': 'Suécia',
        'Czech Republic': 'Tchéquia',
        'Algeria': 'Argélia',
        'Paraguay': 'Paraguai',
        'South Africa': 'África do Sul',
        'Ivory Coast': 'Costa do Marfim',
        'DR Congo': 'Rep. D. Congo',
        'New Zealand': 'Nova Zelândia',
        'Bosnia-Herzegovina': 'Bósnia',
        'Curacao': 'Curaçao',
        'Haiti': 'Haiti',
        'Cape Verde': 'Cabo Verde',
        'Qatar': 'Catar',
        'Iraq': 'Iraque',
        'Jordan': 'Jordânia',
        'Uzbekistan': 'Uzbequistão',
    }

    team_pt = TEAM_NAME_PT.get(team_name_en, team_name_en)

    response = supabase.table('players')\
        .select('name, position, teams!inner(name)')\
        .eq('teams.name', team_pt)\
        .neq('name', 'Gol Contra')\
        .execute()

    return response.data or []

def predict_scorers(home_team: str,
                     away_team: str,
                     home_elo: float,
                     away_elo: float,
                     opp_goals_conceded: float = 1.2,
                     opp_clean_sheet_rate: float = 0.30,
                     is_neutral: bool = True,
                     top_n: int = 5) -> dict:
    """
    Retorna top N jogadores mais prováveis de marcar para cada time.
    Filtra pelos convocados reais do Supabase.
    """

    df = get_scorer_features_df()
    results = {}

    for team_en, team_elo, opp_elo in [
        (home_team, home_elo, away_elo),
        (away_team, away_elo, home_elo),
    ]:
        # Buscar convocados do Supabase
        convocados = get_convocados(team_en)

        if not convocados:
            results[team_en] = []
            continue

        team_results = []

        for player in convocados:
            player_name_supabase = player['name']
            position = player['position']

            # Mapear nome para o CSV
            player_name_csv = MANUAL_MAPPING.get(
                player_name_supabase, player_name_supabase
            )

            # Buscar histórico do jogador
            player_history = df[df['scorer'] == player_name_csv]

            # Position weight
            position_weight = {
                'FW': 1.0, 'MF': 0.5, 'DF': 0.15, 'GK': 0.01
            }.get(position, 0.1)

            # Baseline por posição se sem histórico
            POSITION_BASELINE = {
                'FW': 0.35, 'MF': 0.12, 'DF': 0.04, 'GK': 0.001
            }

            if len(player_history) == 0:
                # Sem histórico — usar baseline
                features = {
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
                    'position_weight': position_weight,
                    'opp_elo': opp_elo,
                    'team_elo': team_elo,
                    'elo_diff': team_elo - opp_elo,
                    'opp_tier_elite': int(opp_elo >= 1900),
                    'opp_tier_strong': int(1800 <= opp_elo < 1900),
                    'opp_tier_mid': int(1700 <= opp_elo < 1800),
                    'opp_tier_weak': int(opp_elo < 1700),
                    'opp_goals_conceded_avg': opp_goals_conceded,
                    'opp_clean_sheet_rate': opp_clean_sheet_rate,
                    'team_coverage': 0.5,
                    'is_neutral': int(is_neutral),
                }
            else:
                # Usar última linha do histórico como base
                last = player_history.sort_values('date').iloc[-1]
                features = {col: last[col] for col in SCORER_FEATURES
                           if col in last.index}

                # Atualizar contexto da partida atual
                features.update({
                    'opp_elo': opp_elo,
                    'team_elo': team_elo,
                    'elo_diff': team_elo - opp_elo,
                    'opp_tier_elite': int(opp_elo >= 1900),
                    'opp_tier_strong': int(1800 <= opp_elo < 1900),
                    'opp_tier_mid': int(1700 <= opp_elo < 1800),
                    'opp_tier_weak': int(opp_elo < 1700),
                    'opp_goals_conceded_avg': opp_goals_conceded,
                    'opp_clean_sheet_rate': opp_clean_sheet_rate,
                    'position_weight': position_weight,
                    'is_neutral': int(is_neutral),
                })

            # Garantir ordem correta das features
            feature_vector = [features.get(col, 0.0) for col in SCORER_FEATURES]
            prob = float(model_scorer.predict_proba([feature_vector])[0][1])

            # GK com prob mínima
            if position == 'GK':
                prob = min(prob, 0.005)

            team_results.append({
                'player': player_name_supabase,
                'position': position,
                'probability': round(prob, 4),
                'probability_pct': round(prob * 100, 1),
                'has_history': len(player_history) > 0,
            })

        # Ordenar por probabilidade e retornar top N
        team_results.sort(key=lambda x: x['probability'], reverse=True)
        results[team_en] = team_results[:top_n]

    return results
