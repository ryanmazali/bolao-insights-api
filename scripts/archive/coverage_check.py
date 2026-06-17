import pandas as pd
from statsbombpy import sb

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# As 48 selecoes da Copa 2026 (nomes em portugues, conforme tabela `teams`
# do Supabase), com o nome em ingles usado pelo StatsBomb e o codigo de
# 3 letras usado pelo FBref na coluna `Nation` (ex: "br BRA" -> "BRA").
WC2026_TEAMS = {
    'México':           {'en': 'Mexico',          'fbref': 'MEX'},
    'África do Sul':    {'en': 'South Africa',    'fbref': 'RSA'},
    'Coreia do Sul':    {'en': 'South Korea',     'fbref': 'KOR'},
    'Tchéquia':         {'en': 'Czech Republic',  'fbref': 'CZE'},
    'Canadá':           {'en': 'Canada',          'fbref': 'CAN'},
    'Bósnia':           {'en': 'Bosnia and Herzegovina', 'fbref': 'BIH'},
    'Catar':            {'en': 'Qatar',           'fbref': 'QAT'},
    'Suíça':            {'en': 'Switzerland',     'fbref': 'SUI'},
    'Brasil':           {'en': 'Brazil',          'fbref': 'BRA'},
    'Marrocos':         {'en': 'Morocco',         'fbref': 'MAR'},
    'Haiti':            {'en': 'Haiti',           'fbref': 'HAI'},
    'Escócia':          {'en': 'Scotland',        'fbref': 'SCO'},
    'Estados Unidos':   {'en': 'United States',   'fbref': 'USA'},
    'Paraguai':         {'en': 'Paraguay',        'fbref': 'PAR'},
    'Austrália':        {'en': 'Australia',       'fbref': 'AUS'},
    'Turquia':          {'en': 'Turkey',          'fbref': 'TUR'},
    'Alemanha':         {'en': 'Germany',         'fbref': 'GER'},
    'Curaçao':          {'en': 'Curacao',         'fbref': 'CUW'},
    'Costa do Marfim':  {'en': 'Ivory Coast',     'fbref': 'CIV'},
    'Equador':          {'en': 'Ecuador',         'fbref': 'ECU'},
    'Holanda':          {'en': 'Netherlands',     'fbref': 'NED'},
    'Japão':            {'en': 'Japan',           'fbref': 'JPN'},
    'Suécia':           {'en': 'Sweden',          'fbref': 'SWE'},
    'Tunísia':          {'en': 'Tunisia',         'fbref': 'TUN'},
    'Bélgica':          {'en': 'Belgium',         'fbref': 'BEL'},
    'Egito':            {'en': 'Egypt',           'fbref': 'EGY'},
    'Irã':              {'en': 'Iran',            'fbref': 'IRN'},
    'Nova Zelândia':    {'en': 'New Zealand',     'fbref': 'NZL'},
    'Espanha':          {'en': 'Spain',           'fbref': 'ESP'},
    'Cabo Verde':       {'en': 'Cape Verde',      'fbref': 'CPV'},
    'Arábia Saudita':   {'en': 'Saudi Arabia',    'fbref': 'KSA'},
    'Uruguai':          {'en': 'Uruguay',         'fbref': 'URU'},
    'França':           {'en': 'France',          'fbref': 'FRA'},
    'Senegal':          {'en': 'Senegal',         'fbref': 'SEN'},
    'Iraque':           {'en': 'Iraq',            'fbref': 'IRQ'},
    'Noruega':          {'en': 'Norway',          'fbref': 'NOR'},
    'Argentina':        {'en': 'Argentina',       'fbref': 'ARG'},
    'Argélia':          {'en': 'Algeria',         'fbref': 'ALG'},
    'Áustria':          {'en': 'Austria',         'fbref': 'AUT'},
    'Jordânia':         {'en': 'Jordan',          'fbref': 'JOR'},
    'Portugal':         {'en': 'Portugal',        'fbref': 'POR'},
    'Rep. D. Congo':    {'en': 'DR Congo',        'fbref': 'COD'},
    'Uzbequistão':      {'en': 'Uzbekistan',      'fbref': 'UZB'},
    'Colômbia':         {'en': 'Colombia',        'fbref': 'COL'},
    'Inglaterra':       {'en': 'England',         'fbref': 'ENG'},
    'Croácia':          {'en': 'Croatia',         'fbref': 'CRO'},
    'Gana':             {'en': 'Ghana',           'fbref': 'GHA'},
    'Panamá':           {'en': 'Panama',          'fbref': 'PAN'},
}

print(f"Total de selecoes Copa 2026: {len(WC2026_TEAMS)}")

# ── 1. StatsBomb: Euro 2024 + Copa America 2024 ─────────────────────────
print("\n=== 1. STATSBOMB — EURO 2024 + COPA AMERICA 2024 ===")

euro_matches = sb.matches(competition_id=55, season_id=282)
copa_matches = sb.matches(competition_id=223, season_id=282)

euro_teams = set(euro_matches['home_team']) | set(euro_matches['away_team'])
copa_teams = set(copa_matches['home_team']) | set(copa_matches['away_team'])

print(f"Times na Euro 2024: {len(euro_teams)}")
print(f"Times na Copa America 2024: {len(copa_teams)}")

team_to_tournament = {}
for t in euro_teams:
    team_to_tournament.setdefault(t, []).append('Euro 2024')
for t in copa_teams:
    team_to_tournament.setdefault(t, []).append('Copa America 2024')

# ── 2 e 3. Cruzamento com as 48 selecoes da Copa 2026 ───────────────────
print("\n=== 2/3. COBERTURA STATSBOMB x COPA 2026 ===")

com_dados = []
sem_dados = []

for pt_name, info in WC2026_TEAMS.items():
    en_name = info['en']
    if en_name in team_to_tournament:
        com_dados.append((pt_name, en_name, team_to_tournament[en_name]))
    else:
        sem_dados.append((pt_name, en_name))

print(f"\n-- Selecoes COM dados StatsBomb ({len(com_dados)}) --")
for pt_name, en_name, torneios in com_dados:
    print(f"  {pt_name:<18} ({en_name}) — {', '.join(torneios)}")

print(f"\n-- Selecoes SEM dados StatsBomb ({len(sem_dados)}) --")
for pt_name, en_name in sem_dados:
    print(f"  {pt_name:<18} ({en_name})")

pct = 100 * len(com_dados) / len(WC2026_TEAMS)
print(f"\nCobertura StatsBomb: {len(com_dados)}/{len(WC2026_TEAMS)} ({pct:.1f}%)")

# ── 4. FBref ─────────────────────────────────────────────────────────
print("\n=== 4. FBREF — players_data-2024_2025.csv ===")

df = pd.read_csv('data/raw/players_data-2024_2025.csv')

print(f"\nValores unicos de Nation ({df['Nation'].nunique()}):")
print(sorted(df['Nation'].dropna().unique().tolist()))

df['nation_code'] = df['Nation'].str.split().str[-1]
df['Min_num'] = pd.to_numeric(
    df['Min'].astype(str).str.replace(',', '', regex=False), errors='coerce'
)

df_450 = df[df['Min_num'] >= 450]
print(f"\nJogadores com Min >= 450: {len(df_450)} de {len(df)}")

print("\n-- Jogadores (Min>=450) por selecao da Copa 2026 --")
rows = []
for pt_name, info in WC2026_TEAMS.items():
    code = info['fbref']
    count = (df_450['nation_code'] == code).sum()
    rows.append((pt_name, code, count))

rows.sort(key=lambda x: x[2], reverse=True)
for pt_name, code, count in rows:
    print(f"  {pt_name:<18} ({code}) — {count} jogadores")

zero = [r for r in rows if r[2] == 0]
print(f"\nSelecoes com 0 jogadores no FBref (Min>=450): {len(zero)}")
for pt_name, code, _ in zero:
    print(f"  {pt_name} ({code})")
