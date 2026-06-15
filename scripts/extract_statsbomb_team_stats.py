import pandas as pd
from statsbombpy import sb

pd.set_option('display.width', 200)

# ── 1. Competições e season_ids ─────────────────────────────────────────
comps = sb.competitions()

TOURNAMENTS = []
for label, comp_id, season_name in [
    ('UEFA Euro 2024', 55, '2024'),
    ('Copa America 2024', 223, '2024'),
    ('African Cup of Nations 2023', 1267, '2023'),
]:
    row = comps[(comps['competition_id'] == comp_id) & (comps['season_name'] == season_name)]
    season_id = int(row.iloc[0]['season_id'])
    TOURNAMENTS.append((label, comp_id, season_id))
    print(f"{label}: competition_id={comp_id}, season_id={season_id}")

# ── 2/3/4. Carregamento e extração ──────────────────────────────────────
SHOT_ON_TARGET_OUTCOMES = ['Goal', 'Saved']
TACKLE_WON_OUTCOMES = ['Won', 'Success In Play', 'Success Out']

EVENT_COUNT_TYPES = {
    'fouls': 'Foul Committed',
    'pressures': 'Pressure',
    'interceptions': 'Interception',
    'recoveries': 'Ball Recovery',
}


def match_duration_minutes(events: pd.DataFrame) -> float:
    last = events.loc[events['minute'].idxmax()]
    return float(last['minute']) + float(last.get('second', 0) or 0) / 60.0


def team_match_stats(events: pd.DataFrame, team_name: str) -> dict:
    team_events = events[events['team'] == team_name]

    shots = team_events[team_events['type'] == 'Shot']
    shots_total = len(shots)
    shots_on_target = shots['shot_outcome'].isin(SHOT_ON_TARGET_OUTCOMES).sum() if 'shot_outcome' in shots else 0
    xg_total = float(shots['shot_statsbomb_xg'].fillna(0).sum()) if 'shot_statsbomb_xg' in shots else 0.0

    if 'duel_type' in team_events.columns:
        tackles = team_events[team_events['duel_type'] == 'Tackle']
    else:
        tackles = team_events.iloc[0:0]
    tackles_attempted = len(tackles)
    tackles_won = tackles['duel_outcome'].isin(TACKLE_WON_OUTCOMES).sum() if 'duel_outcome' in tackles else 0

    stats = {
        'shots_total': shots_total,
        'shots_on_target': int(shots_on_target),
        'xg_total': xg_total,
        'tackles_attempted': tackles_attempted,
        'tackles_won': int(tackles_won),
    }
    for key, ev_type in EVENT_COUNT_TYPES.items():
        stats[key] = int((team_events['type'] == ev_type).sum())

    return stats


rows = []
duel_outcome_values = set()
games_per_tournament = {}
games_under_3 = {}

for label, comp_id, season_id in TOURNAMENTS:
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"\n=== {label}: {len(matches)} partidas ===")
    games_per_tournament[label] = 0

    for i, (_, m) in enumerate(matches.iterrows(), start=1):
        match_id = int(m['match_id'])
        home, away = m['home_team'], m['away_team']
        print(f"Jogo {i}/{len(matches)}: {home} vs {away} (match_id={match_id})")

        try:
            events = sb.events(match_id=match_id)
        except Exception as e:
            print(f"  ERRO ao carregar eventos: {e}")
            continue

        if 'duel_outcome' in events.columns:
            duel_outcome_values.update(events['duel_outcome'].dropna().unique().tolist())

        duration = match_duration_minutes(events)
        factor = 90.0 / duration if duration > 0 else 1.0

        home_stats = team_match_stats(events, home)
        away_stats = team_match_stats(events, away)

        for team, opponent, own, opp in [
            (home, away, home_stats, away_stats),
            (away, home, away_stats, home_stats),
        ]:
            rows.append({
                'team': team,
                'tournament': label,
                'match_id': match_id,
                'opponent': opponent,
                'match_date': m['match_date'],
                'shots_p90': own['shots_total'] * factor,
                'sot_p90': own['shots_on_target'] * factor,
                'xg_p90': own['xg_total'] * factor,
                'shots_conceded_p90': opp['shots_total'] * factor,
                'sot_conceded_p90': opp['shots_on_target'] * factor,
                'xg_conceded_p90': opp['xg_total'] * factor,
                'tackles_p90': own['tackles_attempted'] * factor,
                'tackles_won_p90': own['tackles_won'] * factor,
                'fouls_p90': own['fouls'] * factor,
                'pressures_p90': own['pressures'] * factor,
                'interceptions_p90': own['interceptions'] * factor,
                'recoveries_p90': own['recoveries'] * factor,
            })
            games_per_tournament[label] += 1

print(f"\nValores únicos de duel_outcome encontrados: {sorted(duel_outcome_values)}")

df = pd.DataFrame(rows)
df.to_csv('data/processed/team_stats_statsbomb.csv', index=False)

print("\n=== RESUMO ===")
print(f"Total de linhas geradas: {len(df)}")
print(f"Times únicos cobertos: {df['team'].nunique()}")

print("\nJogos (linhas time-jogo) por torneio:")
print(df.groupby('tournament').size().to_string())

print("\nJogos por time (ordenado crescente):")
games_per_team = df.groupby('team').size().sort_values()
print(games_per_team.to_string())

under_3 = games_per_team[games_per_team < 3]
print(f"\nTimes com menos de 3 jogos registrados: {len(under_3)}")
print(under_3.to_string())

print("\nArquivo salvo em data/processed/team_stats_statsbomb.csv")
