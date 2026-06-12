import pandas as pd
from rapidfuzz import fuzz

fm23 = pd.read_csv('data/raw/merged_players (1).csv')
mapping = pd.read_csv('data/processed/fm23_player_mapping.csv')

craques = [
    ('Lamine Yamal', 'Espanha', 'Spain'),
    ('Cubarsí', 'Espanha', 'Spain'),
    ('Marcus Rashford', 'Inglaterra', 'England'),
    ('Kobbie Mainoo', 'Inglaterra', 'England'),
    ('Brahim Díaz', 'Marrocos', 'Morocco'),
    ('Facundo Pellistri', 'Uruguai', 'Uruguay'),
    ('Folarin Balogun', 'Estados Unidos', 'USA'),
    ('Lamine Camara', 'Senegal', 'Senegal'),
    ('Habib Diarra', 'Senegal', 'Senegal'),
    ('Assane Diao', 'Senegal', 'Senegal'),
    ('Julián Quiñones', 'México', 'Mexico'),
    ('Koki Ogawa', 'Japão', 'Japan'),
]

for supabase_name, supabase_team, nat in craques:
    candidates = fm23[fm23['Nat'].str.contains(nat[:3], case=False, na=False)].copy()
    if candidates.empty:
        candidates = fm23.copy()
    candidates['score'] = candidates['Name'].apply(lambda x: fuzz.WRatio(supabase_name, str(x)))
    top5 = candidates.nlargest(5, 'score')[['Name', 'Nat', 'score', 'UID']]
    print(f"\n{'='*50}")
    print(f"{supabase_name} ({supabase_team})")
    print(top5.to_string(index=False))