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

    # Diacríticos/hífens: nome no Supabase está sem acento/hífen, mas o CSV
    # (data/processed/scorer_features_v1.csv) usa a grafia original.
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

# Atributos FM23 (data/processed/fm23_player_attributes.csv)
FM23_ATTRS = [
    'Fin', 'OtB', 'Com', 'Dec', 'Pac', 'Acc', 'Hea', 'Pen',
    'Dri', 'Str', 'Vis', 'Ant', 'Fla', 'Lon',
]

_fm23_attributes_cache = None

def get_fm23_attributes_df():
    global _fm23_attributes_cache
    if _fm23_attributes_cache is None:
        _fm23_attributes_cache = pd.read_csv(
            'data/processed/fm23_player_attributes.csv'
        ).set_index('supabase_id')
    return _fm23_attributes_cache

def get_fm23_attributes(supabase_id: str) -> dict:
    """Atributos FM23 do jogador pelo supabase_id; dict vazio se não encontrado."""
    fm23_df = get_fm23_attributes_df()
    if supabase_id not in fm23_df.index:
        return {}
    row = fm23_df.loc[supabase_id]
    return {col: float(row[col]) for col in FM23_ATTRS}

def get_team_coverage(df: pd.DataFrame, team_name_csv: str) -> float:
    """team_coverage da seleção (constante entre as linhas); 0.5 (neutro)
    se a seleção não tiver nenhuma linha em scorer_features_v1.csv."""
    rows = df[df['team'] == team_name_csv]
    if len(rows) == 0:
        return 0.5
    return float(rows['team_coverage'].iloc[0])

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
        'United States': 'Estados Unidos',
        'Panama': 'Panamá',
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
        .select('id, name, position, teams!inner(name)')\
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

        team_coverage = get_team_coverage(df, team_en)

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

            # Atributos FM23 do jogador (pelo supabase_id)
            fm23_attrs = get_fm23_attributes(player['id'])

            # Score de qualidade intrínseca a partir dos atributos FM23
            # chave para finalizadores (escala 0-1).
            fm23_score = sum(
                fm23_attrs.get(attr, 0.0) for attr in ('Fin', 'OtB', 'Pac', 'Ant')
            ) / (20 * 4)

            # Position weight
            position_weight = {
                'FW': 1.0, 'MF': 0.5, 'DF': 0.15, 'GK': 0.01
            }.get(position, 0.1)

            # Baseline por posição se sem histórico
            POSITION_BASELINE = {
                'FW': 0.12, 'MF': 0.05, 'DF': 0.02, 'GK': 0.01
            }

            if len(player_history) == 0:
                opp_tier_multiplier = (
                    0.7 if opp_elo >= 1900 else
                    0.85 if opp_elo >= 1800 else
                    1.0 if opp_elo >= 1700 else
                    1.25
                )

                if position == 'FW':
                    # Sem histórico, mas atacante — monta um vetor sintético
                    # (n_matches=0, forma recente = baseline da posição,
                    # contexto da partida atual) e usa o modelo. Com a
                    # regularização atual (ver train_scorer.py), o modelo dá
                    # mais peso a Fin/OtB/Pac/Ant, então a probabilidade
                    # diferencia atacantes sem histórico pela qualidade FM23
                    # (ex: Gyökeres Fin=17 > atacante mediano Fin=12). Para
                    # outras posições o modelo extrapola de forma errática
                    # fora da posição FW (GK/DF "sem histórico" previstos a
                    # 9-11%, acima do baseline real), então mantém o
                    # baseline puro por posição.
                    fw_baseline = POSITION_BASELINE['FW']
                    no_history_features = {
                        'n_matches': 0,
                        'scoring_rate': fw_baseline,
                        'scoring_rate_recent10': fw_baseline,
                        'scoring_rate_recent5': fw_baseline,
                        'scoring_rate_last_12m': fw_baseline,
                        'scoring_rate_last_24m': fw_baseline,
                        'sos_scoring_rate': fw_baseline,
                        'goals_vs_elite': 0.0,
                        'goals_vs_strong': 0.0,
                        'goals_vs_mid': fw_baseline,
                        'goals_vs_weak': fw_baseline * 1.5,
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
                        'team_coverage': team_coverage,
                        'is_neutral': int(is_neutral),
                    }
                    no_history_features.update(fm23_attrs)
                    feature_vector = [no_history_features.get(col, 0.0) for col in SCORER_FEATURES]
                    prob = float(model_scorer.predict_proba([feature_vector])[0][1])
                else:
                    prob = POSITION_BASELINE.get(position, 0.05) * opp_tier_multiplier
                    if position == 'GK':
                        prob = min(prob, 0.005)

                # Sem histórico: componentes de forma ficam zerados, o score
                # composto passa a depender de qualidade FM23 + prob.
                composite_score = (
                    0.35 * fm23_score +
                    0.30 * 0.0 +
                    0.20 * 0.0 +
                    0.15 * prob
                )

                team_results.append({
                    'player': player_name_supabase,
                    'position': position,
                    'probability': round(prob, 4),
                    'probability_pct': round(prob * 100, 1),
                    'composite_score': round(composite_score, 4),
                    'has_history': False,
                })
                continue
            else:
                # Usar última linha do histórico como base
                last = player_history.sort_values('date').iloc[-1]
                features = {col: last[col] for col in SCORER_FEATURES
                           if col in last.index}

                # Histórico geral e forma recente para o score composto
                hist_score = float(last['scoring_rate'])
                recent_score = (
                    float(last['scoring_rate_last_24m']) * 0.6 +
                    float(last['scoring_rate_last_12m']) * 0.4
                )

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
                features.update(fm23_attrs)

            # Garantir ordem correta das features
            feature_vector = [features.get(col, 0.0) for col in SCORER_FEATURES]
            prob = float(model_scorer.predict_proba([feature_vector])[0][1])

            # GK com prob mínima
            if position == 'GK':
                prob = min(prob, 0.005)

            # Score composto: pondera qualidade FM23, histórico geral, forma
            # recente e a probabilidade do XGBoost (componente menor).
            composite_score = (
                0.35 * fm23_score +
                0.30 * hist_score +
                0.20 * recent_score +
                0.15 * prob
            )

            team_results.append({
                'player': player_name_supabase,
                'position': position,
                'probability': round(prob, 4),
                'probability_pct': round(prob * 100, 1),
                'composite_score': round(composite_score, 4),
                'has_history': len(player_history) > 0,
            })

        # Ordenar pelo score composto (qualidade + forma) e retornar top N.
        # 'probability' segue como raw_prob do XGBoost, mantendo a calibração
        # exibida ao usuário.
        team_results.sort(key=lambda x: x['composite_score'], reverse=True)
        results[team_en] = team_results[:top_n]

    return results
