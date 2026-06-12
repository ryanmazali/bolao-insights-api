import pandas as pd

df = pd.read_csv('data/processed/fm23_player_mapping.csv')

selecoes_relevantes = [
    'Brasil', 'Argentina', 'França', 'Inglaterra', 'Alemanha', 
    'Portugal', 'Espanha', 'Holanda', 'Bélgica', 'Itália',
    'Croácia', 'Uruguai', 'México', 'Estados Unidos', 'Japão',
    'Coreia do Sul', 'Marrocos', 'Senegal', 'Egito'
]

unmatched = df[
    (df['status'] == 'unmatched') & 
    (df['supabase_team'].isin(selecoes_relevantes))
][['supabase_name', 'supabase_team']].sort_values('supabase_team')

print(unmatched.to_string())
print(f"\nTotal: {len(unmatched)}")