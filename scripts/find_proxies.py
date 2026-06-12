import pandas as pd
from rapidfuzz import fuzz

fm23 = pd.read_csv('data/raw/merged_players (1).csv')

proxies = [
    ('Lamine Yamal',     'Espanha',    'Neymar',            'BRA'),
    ('Cubarsí',          'Espanha',    'Aymeric Laporte',   'ESP'),
    ('Marcus Rashford',  'Inglaterra', 'Raheem Sterling',   'ENG'),
    ('Kobbie Mainoo',    'Inglaterra', 'Jude Bellingham',   'ENG'),
    ('Elliot Anderson',  'Inglaterra', 'Kalvin Phillips',   'ENG'),
    ('Folarin Balogun',  'Estados Unidos', 'Ollie Watkins', 'ENG'),
    ('Julián Quiñones',  'México',     'Hirving Lozano',    'MEX'),
    ('Lamine Camara',    'Senegal',    'Idrissa Gueye',     'SEN'),
    ('Habib Diarra',     'Senegal',    'Cheikhou Kouyaté',  'SEN'),
    ('Assane Diao',      'Senegal',    'Ismaila Sarr',      'SEN'),
    ('Facundo Pellistri','Uruguai',    'Federico Valverde', 'URU'),
    ('Koki Ogawa',       'Japão',      'Takumi Minamino',   'JPN'),
    ('Lennart Karl',     'Alemanha',   'Bernd Leno',        'GER'),
]

for player, team, proxy_name, proxy_nat in proxies:
    candidates = fm23.copy()
    candidates['score'] = candidates['Name'].apply(lambda x: fuzz.WRatio(proxy_name, str(x)))
    top3 = candidates.nlargest(3, 'score')[['Name', 'Nat', 'score', 'UID']]
    print(f"\n{'='*50}")
    print(f"{player} ({team}) → proxy: {proxy_name}")
    print(top3.to_string(index=False))