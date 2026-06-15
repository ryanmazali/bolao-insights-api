import numpy as np
import pandas as pd
import os
from supabase import create_client
from dotenv import load_dotenv

from src.models.predict import predict_match

load_dotenv()

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

# Peso de cada posição na distribuição dos gols esperados do time. FW e MF
# competem em pé de igualdade (a diferença de perfil já está nas fórmulas de
# fm23_offensive_score); DF/GK são fortemente penalizados pois raramente
# marcam, apesar de seus atributos FM23 (Hea/Str) terem magnitude parecida
# com os de um atacante.
POSITION_WEIGHT = {'FW': 1.0, 'MF': 1.0, 'DF': 0.05, 'GK': 0.001}

def fm23_offensive_score(attrs: dict, position: str) -> float:
    """Score de ameaça ofensiva (escala ~0-20) a partir dos atributos FM23,
    ponderado conforme o papel típico da posição no ataque."""
    g = lambda attr: attrs.get(attr, 10.0)
    if position == 'FW':
        return (
            g('Fin') * 0.30 + g('OtB') * 0.25 + g('Pac') * 0.15 +
            g('Dri') * 0.15 + g('Hea') * 0.10 + g('Pen') * 0.05
        )
    if position == 'MF':
        # 'Pas' (passe) não existe nos atributos FM23 disponíveis;
        # usa 'Vis' (visão de jogo) como proxy.
        return (
            g('Fin') * 0.20 + g('OtB') * 0.20 + g('Lon') * 0.20 +
            g('Vis') * 0.15 + g('Dri') * 0.15 + g('Ant') * 0.10
        )
    if position == 'DF':
        return g('Hea') * 0.40 + g('Str') * 0.30 + g('OtB') * 0.20 + g('Fin') * 0.10
    return 0.01  # GK

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
                     home_elo: float = None,
                     away_elo: float = None,
                     opp_goals_conceded: float = 1.2,
                     opp_clean_sheet_rate: float = 0.30,
                     is_neutral: bool = True,
                     top_n: int = 5) -> dict:
    """
    Retorna top N jogadores mais prováveis de marcar para cada time.

    Modelo bottom-up: os gols esperados do time (modelo de resultado/gols,
    via predict_match) são distribuídos entre os convocados proporcionalmente
    a um score de ameaça ofensiva derivado dos atributos FM23, e a
    probabilidade de marcar é obtida via Poisson (P(gols do jogador >= 1)).
    Jogadores com histórico real de gols recebem um boost leve.

    `home_elo`/`away_elo`/`opp_goals_conceded`/`opp_clean_sheet_rate`/
    `is_neutral` são aceitos por compatibilidade com chamadores existentes,
    mas não são usados por este modelo.
    """

    df = get_scorer_features_df()

    match = predict_match(home_team, away_team)
    expected_goals = {
        home_team: match['expected_goals']['home'],
        away_team: match['expected_goals']['away'],
    }

    results = {}

    for team_en in (home_team, away_team):
        # Buscar convocados do Supabase
        convocados = get_convocados(team_en)

        if not convocados:
            results[team_en] = []
            continue

        team_players = []

        for player in convocados:
            position = player['position']

            # Mapear nome para o CSV
            player_name_csv = MANUAL_MAPPING.get(player['name'], player['name'])

            # Buscar histórico do jogador
            player_history = df[df['scorer'] == player_name_csv]
            has_history = len(player_history) > 0
            scoring_rate = (
                float(player_history.sort_values('date').iloc[-1]['scoring_rate'])
                if has_history else 0.0
            )

            # Atributos FM23 do jogador (pelo supabase_id)
            fm23_attrs = get_fm23_attributes(player['id'])
            offensive_score = (
                fm23_offensive_score(fm23_attrs, position) * POSITION_WEIGHT.get(position, 0.1)
            )

            team_players.append({
                'player': player['name'],
                'position': position,
                'offensive_score': offensive_score,
                'has_history': has_history,
                'scoring_rate': scoring_rate,
            })

        total_score = sum(p['offensive_score'] for p in team_players)

        team_results = []
        for p in team_players:
            share = p['offensive_score'] / total_score if total_score > 0 else 0.0
            prob_marcar = 1 - np.exp(-share * expected_goals[team_en])

            # Boost leve para jogadores com histórico real de gols.
            history_multiplier = 1.0 + min(0.30, p['scoring_rate'] * 2) if p['has_history'] else 1.0
            prob = min(0.99, prob_marcar * history_multiplier)

            team_results.append({
                'player': p['player'],
                'position': p['position'],
                'probability': round(prob, 4),
                'probability_pct': round(prob * 100, 1),
                'has_history': p['has_history'],
            })

        team_results.sort(key=lambda x: x['probability'], reverse=True)
        results[team_en] = team_results[:top_n]

    return results
