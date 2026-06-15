import pandas as pd
from statsbombpy import sb

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=== 1. COMPETIÇÕES DISPONÍVEIS ===")
comps = sb.competitions()
print(comps[['competition_id', 'competition_name', 'season_name']].to_string())

print("\n=== 2. COMPETIÇÕES DE SELEÇÕES ===")
NATIONAL_KEYWORDS = [
    'World Cup', 'Euro', 'Copa America', 'African Cup', 'Africa Cup',
    'Qualif', 'UEFA Nations League', 'Nations League', 'Copa Amer',
]
mask = comps['competition_name'].str.contains('|'.join(NATIONAL_KEYWORDS), case=False, na=False)
national_comps = comps[mask]
print(national_comps[['competition_id', 'competition_name', 'season_name']].to_string())

print("\n=== 3. COPA DO MUNDO MAIS RECENTE ===")
world_cups = comps[comps['competition_name'] == 'FIFA World Cup']
print(world_cups[['competition_id', 'competition_name', 'season_name', 'season_id']].to_string())

# Pega a edição mais recente (maior season_id)
latest_wc = world_cups.sort_values('season_id', ascending=False).iloc[0]
comp_id = int(latest_wc['competition_id'])
season_id = int(latest_wc['season_id'])
print(f"\nSelecionada: {latest_wc['competition_name']} - {latest_wc['season_name']} "
      f"(competition_id={comp_id}, season_id={season_id})")

print("\n=== 3a. PARTIDAS ===")
matches = sb.matches(competition_id=comp_id, season_id=season_id)
print(f"Total de partidas: {len(matches)}")
cols_matches = [c for c in ['match_id', 'match_date', 'home_team', 'away_team',
                             'home_score', 'away_score', 'competition_stage'] if c in matches.columns]
print(matches[cols_matches].to_string())

match_id = int(matches.iloc[0]['match_id'])
print(f"\nPartida escolhida para análise de eventos: match_id={match_id} "
      f"({matches.iloc[0]['home_team']} x {matches.iloc[0]['away_team']})")

print("\n=== 3b. EVENTOS DA PARTIDA ===")
events = sb.events(match_id=match_id)
print(f"Total de eventos: {len(events)}")

print("\n--- Colunas disponíveis ---")
for col in events.columns:
    print(f"  {col}")

print("\n--- value_counts() de 'type' ---")
print(events['type'].value_counts().to_string())

print("\n=== 3c. EVENTOS 'Shot' ===")
shot_cols = [c for c in events.columns if c.startswith('shot')]
shots = events[events['type'] == 'Shot']
print(f"Total de eventos Shot: {len(shots)}")
print(f"\nColunas shot.*: {shot_cols}")
if not shots.empty:
    print("\nNulos por coluna shot.* (na amostra de Shot):")
    print(shots[shot_cols].isnull().sum().to_string())
    print("\nExemplo de 1 evento Shot (colunas shot.* não-nulas):")
    sample = shots.iloc[0][shot_cols].dropna()
    print(sample.to_string())

print("\n=== 3d. EVENTOS 'Duel'/'Tackle' ===")
duel_cols = [c for c in events.columns if c.startswith('duel')]
duels = events[events['type'].isin(['Duel', 'Tackle'])]
print(f"Total de eventos Duel/Tackle: {len(duels)}")
print(f"\nColunas duel.*: {duel_cols}")
if not duels.empty:
    print("\nNulos por coluna duel.* (na amostra de Duel/Tackle):")
    print(duels[duel_cols].isnull().sum().to_string())
    print("\nExemplo de 1 evento Duel/Tackle (colunas duel.* não-nulas):")
    sample = duels.iloc[0][duel_cols].dropna()
    print(sample.to_string())
