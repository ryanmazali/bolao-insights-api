import pandas as pd

PATH = 'data/raw/players_data-2024_2025.csv'

df = None
for encoding in ('utf-8', 'utf-8-sig', 'latin-1'):
    try:
        df = pd.read_csv(PATH, encoding=encoding)
        print(f"Encoding usado: {encoding}")
        break
    except UnicodeDecodeError:
        continue

print(f"\nShape: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("\n=== COLUNAS ===")
for col in df.columns:
    print(f"  {col}")

print("\n=== df['Pos'].value_counts() ===")
print(df['Pos'].value_counts())

print("\n=== df['Comp'].value_counts() ===")
print(df['Comp'].value_counts())

print("\n=== NULOS POR COLUNA (decrescente) ===")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0].sort_values(ascending=False)
print(nulls.to_string())

print("\n=== 1 EXEMPLO POR POSICAO (FW, MF, DF, GK) ===")
cols_sample = ['Player', 'Squad', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'Tkl', 'Int', 'KP', 'PrgP', 'PrgC']
cols_sample = [c for c in cols_sample if c in df.columns]
for pos in ['FW', 'MF', 'DF', 'GK']:
    row = df[df['Pos'] == pos].head(1)
    print(f"\n-- {pos} --")
    if row.empty:
        print("  (nenhuma linha encontrada)")
    else:
        print(row[cols_sample].to_string(index=False))

print("\n=== df['Player'].head(30) ===")
print(df['Player'].head(30).to_string())
