import pandas as pd

fm23 = pd.read_csv('data/raw/merged_players (1).csv')
mapping = pd.read_csv('data/processed/fm23_player_mapping.csv')

proxies = [
    ('Lamine Yamal',      'Espanha',       19024412),
    ('Cubarsí',           'Espanha',       85085378),
    ('Marcus Rashford',   'Inglaterra',    28054109),
    ('Kobbie Mainoo',     'Inglaterra',    29232937),
    ('Elliot Anderson',   'Inglaterra',    29113879),
    ('Folarin Balogun',   'Estados Unidos',29110623),
    ('Julián Quiñones',   'México',        51042313),
    ('Lamine Camara',     'Senegal',       85032335),
    ('Habib Diarra',      'Senegal',       18022052),
    ('Assane Diao',       'Senegal',       48042766),
    ('Facundo Pellistri', 'Uruguai',       78074594),
    ('Koki Ogawa',        'Japão',         45064891),
    ('Lennart Karl',      'Alemanha',      92012109),
]

for player, team, uid in proxies:
    proxy_row = fm23[fm23['UID'] == uid]
    if proxy_row.empty:
        print(f"UID não encontrado: {uid} ({player})")
        continue
    proxy_name = proxy_row.iloc[0]['Name']
    mask = (mapping['supabase_name'] == player) & (mapping['supabase_team'] == team)
    mapping.loc[mask, 'fm23_name'] = proxy_name
    mapping.loc[mask, 'fm23_uid'] = uid
    mapping.loc[mask, 'status'] = 'proxy'
    print(f"OK: {player} → {proxy_name}")

mapping.to_csv('data/processed/fm23_player_mapping.csv', index=False)

matched = (mapping['status'] == 'matched').sum()
proxy = (mapping['status'] == 'proxy').sum()
unmatched = (mapping['status'] == 'unmatched').sum()
print(f"\nMatched: {matched} | Proxy: {proxy} | Unmatched: {unmatched}")