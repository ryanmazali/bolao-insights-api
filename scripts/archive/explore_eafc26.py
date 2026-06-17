import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', None)

# ── Copa 2026 — mesma lista do coverage_check.py ────────────────────────────
# Mapeamento: nome PT → nome EA FC26 (nationality no CSV)
WC2026_EA = {
    'México':           'Mexico',
    'África do Sul':    'South Africa',
    'Coreia do Sul':    'Korea Republic',
    'Tchéquia':         'Czech Republic',
    'Canadá':           'Canada',
    'Bósnia':           'Bosnia and Herzegovina',
    'Catar':            'Qatar',
    'Suíça':            'Switzerland',
    'Brasil':           'Brazil',
    'Marrocos':         'Morocco',
    'Haiti':            'Haiti',
    'Escócia':          'Scotland',
    'Estados Unidos':   'United States',
    'Paraguai':         'Paraguay',
    'Austrália':        'Australia',
    'Turquia':          'Turkey',
    'Alemanha':         'Germany',
    'Curaçao':          'Curaçao',
    'Costa do Marfim':  "Côte d'Ivoire",
    'Equador':          'Ecuador',
    'Holanda':          'Holland',
    'Japão':            'Japan',
    'Suécia':           'Sweden',
    'Tunísia':          'Tunisia',
    'Bélgica':          'Belgium',
    'Egito':            'Egypt',
    'Irã':              'Iran',
    'Nova Zelândia':    'New Zealand',
    'Espanha':          'Spain',
    'Cabo Verde':       'Cape Verde Islands',
    'Arábia Saudita':   'Saudi Arabia',
    'Uruguai':          'Uruguay',
    'França':           'France',
    'Senegal':          'Senegal',
    'Iraque':           'Iraq',
    'Noruega':          'Norway',
    'Argentina':        'Argentina',
    'Argélia':          'Algeria',
    'Áustria':          'Austria',
    'Jordânia':         'Jordan',
    'Portugal':         'Portugal',
    'Rep. D. Congo':    'Congo DR',
    'Uzbequistão':      'Uzbekistan',
    'Colômbia':         'Colombia',
    'Inglaterra':       'England',
    'Croácia':          'Croatia',
    'Gana':             'Ghana',
    'Panamá':           'Panama',
}

SEP = '=' * 70


def print_section(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)


def explore_file(path, label):
    print_section(f'{label}  —  {path}')
    df = pd.read_csv(path)

    print(f'\nShape: {df.shape}')

    print(f'\nColunas ({len(df.columns)}):')
    for c in df.columns:
        print(f'  {c}')

    print('\nPrimeiros 3 jogadores:')
    name_col = 'commonName' if 'commonName' in df.columns else df.columns[3]
    print(df[[c for c in ['commonName', 'position', 'nationality', 'team', 'overallRating'] if c in df.columns].copy()].head(3).to_string(index=False))

    return df


# ════════════════════════════════════════════════════════════════════════════
# 1. OUTFIELD
# ════════════════════════════════════════════════════════════════════════════
df_out = explore_file('data/raw/ea_fc26_outfield.csv', 'OUTFIELD')

print_section('OUTFIELD — Position (value_counts)')
print(df_out['position'].value_counts().to_string())

print_section('OUTFIELD — League / leagueName (top 20)')
print(df_out['leagueName'].value_counts().head(20).to_string())

print_section('OUTFIELD — Nationality (top 20)')
print(df_out['nationality'].value_counts().head(20).to_string())

# ── Colunas de shooting / finishing ─────────────────────────────────────────
print_section('OUTFIELD — Colunas relacionadas a shooting/finishing')
shoot_kw = ['shoot', 'finish', 'shot', 'volley', 'header']
shoot_cols = [c for c in df_out.columns if any(kw in c.lower() for kw in shoot_kw)]
print(shoot_cols)

# ── Colunas de defending / tackling ─────────────────────────────────────────
print_section('OUTFIELD — Colunas relacionadas a defending/tackling')
def_kw = ['defend', 'tackl', 'intercept', 'block', 'mark']
def_cols = [c for c in df_out.columns if any(kw in c.lower() for kw in def_kw)]
print(def_cols)

# ── Nulos ────────────────────────────────────────────────────────────────────
print_section('OUTFIELD — Nulos por coluna (apenas colunas com nulos)')
nulls = df_out.isnull().sum()
nulls = nulls[nulls > 0]
if nulls.empty:
    print('  Nenhuma coluna com valores nulos.')
else:
    print(nulls.to_string())

# ── Exemplo 1 jogador por posição (numéricos) ────────────────────────────────
num_cols = df_out.select_dtypes(include='number').columns.tolist()
id_cols = ['commonName', 'position', 'nationality', 'team']

for pos_group, pos_filter in [('FW', ['ST', 'CF', 'LW', 'RW', 'LS', 'RS', 'LF', 'RF']),
                               ('MF', ['CM', 'CAM', 'CDM', 'LM', 'RM']),
                               ('DF', ['CB', 'LB', 'RB', 'LWB', 'RWB'])]:
    print_section(f'OUTFIELD — Exemplo de jogador {pos_group} (atributos numéricos)')
    sample = df_out[df_out['position'].isin(pos_filter)].sort_values('overallRating', ascending=False).head(1)
    if sample.empty:
        # fallback: busca pela coluna positionType
        sample = df_out[df_out['positionType'].str.upper() == pos_group].sort_values('overallRating', ascending=False).head(1)
    if not sample.empty:
        row = sample.iloc[0]
        print(f"  {row.get('commonName', row.get('firstName',''))} | {row.get('position','')} | {row.get('team','')} | {row.get('nationality','')}")
        print()
        for c in num_cols:
            print(f"    {c:<30} {row[c]}")
    else:
        print(f'  Nenhum jogador encontrado para {pos_group}')

# ════════════════════════════════════════════════════════════════════════════
# 2. GOALKEEPERS
# ════════════════════════════════════════════════════════════════════════════
df_gk = explore_file('data/raw/ea_fc26_goalkeepers.csv', 'GOALKEEPERS')

print_section('GOALKEEPERS — Colunas disponíveis')
print(list(df_gk.columns))

print_section('GOALKEEPERS — Exemplo de 1 goleiro (todos os atributos)')
best_gk = df_gk.sort_values('overallRating', ascending=False).head(1).iloc[0]
for c in df_gk.columns:
    print(f"  {c:<30} {best_gk[c]}")

# ════════════════════════════════════════════════════════════════════════════
# 3. COBERTURA COPA 2026
# ════════════════════════════════════════════════════════════════════════════
print_section('COBERTURA COPA 2026 — jogadores EA FC26 por seleção')

# Combina outfield + goalkeepers
df_all = pd.concat([
    df_out[['commonName', 'nationality', 'position']],
    df_gk[['commonName', 'nationality', 'position']],
], ignore_index=True)

print(f'\nTotal de jogadores no dataset (outfield + GK): {len(df_all)}')

rows = []
for pt_name, ea_nat in WC2026_EA.items():
    count = (df_all['nationality'] == ea_nat).sum()
    rows.append({'selecao': pt_name, 'ea_nationality': ea_nat, 'jogadores': count})

df_cov = pd.DataFrame(rows).sort_values('jogadores', ascending=False).reset_index(drop=True)

print('\nTop 10 seleções com mais jogadores:')
print(df_cov.head(10).to_string(index=False))

print('\nTodas as seleções (ordenado por cobertura):')
print(df_cov.to_string(index=False))

menos5 = df_cov[df_cov['jogadores'] < 5]
print(f'\nSeleções com menos de 5 jogadores ({len(menos5)}):')
if menos5.empty:
    print('  Nenhuma.')
else:
    print(menos5.to_string(index=False))

zero = df_cov[df_cov['jogadores'] == 0]
print(f'\nSeleções com 0 jogadores ({len(zero)}):')
if zero.empty:
    print('  Nenhuma.')
else:
    print(zero.to_string(index=False))
