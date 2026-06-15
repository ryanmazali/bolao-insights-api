import json

import numpy as np
import pandas as pd

RAW_PATH = 'data/raw/players_data-2024_2025.csv'
OUT_CSV = 'data/processed/fbref_players_per90.csv'
OUT_MEDIANS = 'data/processed/fbref_medians.json'

pd.set_option('display.width', 200)

# ── 1. Leitura ───────────────────────────────────────────────────────────
df = None
for encoding in ('utf-8-sig', 'latin-1'):
    try:
        df = pd.read_csv(RAW_PATH, encoding=encoding)
        print(f"Encoding usado: {encoding}")
        break
    except UnicodeDecodeError:
        continue

df['position'] = df['Pos'].str.split(',').str[0]

# ── 3. Resolução de colunas duplicadas com sufixo ───────────────────────
# Preferência: sem sufixo > _stats_shooting > demais tabelas (na ordem em
# que aparecem no CSV consolidado).
SUFFIX_PRIORITY = [
    '', '_stats_shooting', '_stats_passing', '_stats_passing_types',
    '_stats_gca', '_stats_defense', '_stats_possession',
    '_stats_playing_time', '_stats_misc', '_stats_keeper', '_stats_keeper_adv',
]


def resolve_column(frame: pd.DataFrame, base: str, overrides=None):
    candidates = overrides if overrides else [base + suf for suf in SUFFIX_PRIORITY]
    existing = [c for c in candidates if c in frame.columns]
    if not existing:
        return pd.Series(np.nan, index=frame.index), []
    result = frame[existing[0]].copy()
    for c in existing[1:]:
        result = result.fillna(frame[c])
    return result, existing


# Metricas explicitamente pedidas para log de resolucao, com overrides quando
# o nome "canonico" nao deve ser o primeiro da lista de sufixos padrao.
LOGGED_METRICS = {
    'Sh': None,
    'SoT': None,
    'Fls': None,
    'Recov': None,
    'Won%': None,
    'Blocks': ['Blocks_stats_defense', 'Blocks'],
}

# Demais metricas usadas no pipeline (resolvidas silenciosamente, exceto se
# ambiguas).
OTHER_METRICS = [
    'Tkl', 'TklW', 'Int', 'Clr', 'KP', 'PrgP', 'PrgC', 'Touches', 'Carries',
    'CrdY', 'CrdR', 'xG', 'xAG', 'npxG', 'xA', 'Save%', 'CS%', 'CS', 'GA',
    'Gls', 'Ast', 'Min', 'MP', 'Starts',
]

print("\n=== RESOLUÇÃO DE COLUNAS (métricas pedidas) ===")
for metric, overrides in LOGGED_METRICS.items():
    resolved, used = resolve_column(df, metric, overrides)
    df[f'_{metric}'] = resolved
    print(f"  {metric:<8} -> {used}")

for metric in OTHER_METRICS:
    resolved, used = resolve_column(df, metric)
    df[f'_{metric}'] = resolved
    if len(used) > 1:
        print(f"  {metric:<8} -> {used}  (ambíguo)")

# ── 2. Deduplicação (Player + posição primária) ─────────────────────────
SUM_COLS = [
    '_Min', '_MP', '_Starts', '_Gls', '_Ast', '_Tkl', '_TklW', '_Blocks',
    '_Int', '_Clr', '_KP', '_PrgP', '_PrgC', '_Touches', '_Carries',
    '_CrdY', '_CrdR', '_Recov', '_Sh', '_SoT', '_Fls', '_CS', '_GA',
]
WAVG_COLS = ['_xG', '_xAG', '_npxG', '_xA', '_Save%', '_CS%', '_Won%']


def weighted_avg(group: pd.DataFrame, col: str, weight_col: str = '_Min'):
    valid = group[[col, weight_col]].dropna()
    if valid.empty or valid[weight_col].sum() == 0:
        return np.nan
    return (valid[col] * valid[weight_col]).sum() / valid[weight_col].sum()


def clean_league(comp):
    if pd.isna(comp) or comp == 'Multiple':
        return comp
    parts = comp.split(' ', 1)
    return parts[1] if len(parts) > 1 else comp


records = []
for (player, pos), g in df.groupby(['Player', 'position'], sort=False):
    rec = {'Player': player, 'Pos': pos}

    for col in SUM_COLS:
        rec[col] = g[col].fillna(0).sum()

    for col in WAVG_COLS:
        rec[col] = weighted_avg(g, col)

    top_row = g.loc[g['_Min'].idxmax()]
    rec['Squad'] = top_row['Squad']
    rec['Nation'] = top_row['Nation']
    rec['Age'] = top_row['Age']

    comps = g['Comp'].dropna().unique()
    rec['Comp'] = 'Multiple' if len(comps) > 1 else (comps[0] if len(comps) else np.nan)

    records.append(rec)

agg = pd.DataFrame(records)
agg['Comp'] = agg['Comp'].apply(clean_league)
agg['Min'] = agg['_Min']

# ── 4. Filtro de minutos mínimos ─────────────────────────────────────────
line_mask = agg['Pos'].isin(['FW', 'MF', 'DF'])
gk_mask = agg['Pos'] == 'GK'

keep = (line_mask & (agg['Min'] >= 450)) | (gk_mask & (agg['Min'] >= 270))
agg = agg[keep].copy()


def sample_size(row):
    if row['Min'] >= 900:
        return 'large'
    if row['Pos'] == 'GK' and row['Min'] < 450:
        return 'small'
    return 'medium'


agg['sample_size'] = agg.apply(sample_size, axis=1)

# ── 5. Normalização por 90 minutos ──────────────────────────────────────
agg['90s'] = agg['Min'] / 90.0

P90_SOURCE = {
    'Gls_p90': '_Gls', 'Ast_p90': '_Ast', 'xG_p90': '_xG', 'xAG_p90': '_xAG', 'npxG_p90': '_npxG',
    'Sh_p90': '_Sh', 'SoT_p90': '_SoT',
    'KP_p90': '_KP', 'PrgP_p90': '_PrgP', 'PrgC_p90': '_PrgC',
    'Tkl_p90': '_Tkl', 'TklW_p90': '_TklW', 'Int_p90': '_Int', 'Blocks_p90': '_Blocks', 'Clr_p90': '_Clr',
    'CrdY_p90': '_CrdY', 'Fls_p90': '_Fls', 'Recov_p90': '_Recov',
    'Touches_p90': '_Touches', 'Carries_p90': '_Carries',
}

for out_col, src_col in P90_SOURCE.items():
    agg[out_col] = agg[src_col] / agg['90s']

agg['GA_p90'] = agg['_GA'] / agg['90s']
agg['Save%'] = agg['_Save%']
agg['CS%'] = agg['_CS%']
agg['CS'] = agg['_CS']

# ── 5b. Diagnóstico Sh_p90 — Salah e Marmoush ───────────────────────────
print("\n=== DIAGNÓSTICO Sh_p90 — Salah e Marmoush ===")
for diag_name in ['Mohamed Salah', 'Omar Marmoush']:
    pre_rows = df[df['Player'] == diag_name]
    post_row = agg[agg['Player'] == diag_name]
    print(f"  {diag_name}:")
    print(f"    Linhas antes da deduplicação: {len(pre_rows)}")
    if post_row.empty:
        print("    Não encontrado em 'agg' (removido pelo filtro de minutos mínimos)")
        continue
    r = post_row.iloc[0]
    print(f"    Min total (pós soma): {r['Min']:.0f}")
    print(f"    90s calculado (Min/90): {r['90s']:.4f}")
    print(f"    Sh total (pós soma): {r['_Sh']:.0f}")
    print(f"    Sh_p90 = Sh/90s = {r['Sh_p90']:.4f}")

# ── 5c. Validação de sanidade dos _p90 ──────────────────────────────────
SH_P90_THRESHOLDS = {'FW': 6, 'MF': 4, 'DF': 2}
XG_P90_THRESHOLD = 1.5

print("\n=== VALIDAÇÃO DE SANIDADE (_p90) ===")
n_suspicious = 0
for _, row in agg.iterrows():
    pos = row['Pos']
    sh_threshold = SH_P90_THRESHOLDS.get(pos)
    if sh_threshold is not None and row['Sh_p90'] > sh_threshold:
        print(
            f"  WARNING: {row['Player']} ({row['Squad']}, {pos}) — "
            f"Sh_p90={row['Sh_p90']:.2f} > {sh_threshold} (suspeito para {pos})"
        )
        n_suspicious += 1
    if row['xG_p90'] > XG_P90_THRESHOLD:
        print(
            f"  WARNING: {row['Player']} ({row['Squad']}, {pos}) — "
            f"xG_p90={row['xG_p90']:.2f} > {XG_P90_THRESHOLD}"
        )
        n_suspicious += 1

if n_suspicious == 0:
    print("  Nenhum valor suspeito encontrado.")
else:
    print(f"\n  Total de valores suspeitos: {n_suspicious}")

# ── 6. Medianas por liga/posição (fallback Camada B) ────────────────────
P90_COLS = list(P90_SOURCE.keys()) + ['GA_p90']
GK_EXTRA = ['Save%', 'CS%']
LEAGUES = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'Multiple']
POSITIONS = ['FW', 'MF', 'DF', 'GK']


def median_or_none(series: pd.Series):
    if series.empty or series.notna().sum() == 0:
        return None
    return float(series.median())


medians = {}
for league in LEAGUES:
    sub = agg[agg['Comp'] == league]
    medians[league] = {}
    for pos in POSITIONS:
        pos_sub = sub[sub['Pos'] == pos]
        cols = P90_COLS + (GK_EXTRA if pos == 'GK' else [])
        medians[league][pos] = {c: median_or_none(pos_sub[c]) for c in cols}

medians['global'] = {}
for pos in POSITIONS:
    pos_sub = agg[agg['Pos'] == pos]
    cols = P90_COLS + (GK_EXTRA if pos == 'GK' else [])
    medians['global'][pos] = {c: median_or_none(pos_sub[c]) for c in cols}

with open(OUT_MEDIANS, 'w') as f:
    json.dump(medians, f, indent=2)

# ── 7. Saída principal ───────────────────────────────────────────────────
OUTPUT_COLS = (
    ['Player', 'Pos', 'Squad', 'Comp', 'Nation', 'Age', 'Min', '90s', 'sample_size']
    + list(P90_SOURCE.keys())
    + ['GA_p90', 'Save%', 'CS%', 'CS']
)

out = agg[OUTPUT_COLS].copy()
out.to_csv(OUT_CSV, index=False)

print("\n=== RESUMO ===")
print(f"Shape final: {out.shape[0]} linhas x {out.shape[1]} colunas")

print("\nJogadores por posição após filtro:")
print(out['Pos'].value_counts().to_string())

print("\n-- Top 5 Sh_p90 (FW) --")
print(out[out['Pos'] == 'FW'].nlargest(5, 'Sh_p90')[['Player', 'Squad', 'Min', 'Sh_p90']].to_string(index=False))

print("\n-- Top 5 xG_p90 (FW) --")
print(out[out['Pos'] == 'FW'].nlargest(5, 'xG_p90')[['Player', 'Squad', 'Min', 'xG_p90']].to_string(index=False))

print("\n-- Top 5 Tkl_p90 (DF) --")
print(out[out['Pos'] == 'DF'].nlargest(5, 'Tkl_p90')[['Player', 'Squad', 'Min', 'Tkl_p90']].to_string(index=False))

print("\n-- Top 5 Save% (GK) --")
print(out[out['Pos'] == 'GK'].nlargest(5, 'Save%')[['Player', 'Squad', 'Min', 'Save%']].to_string(index=False))

print("\n-- Medianas globais por posição (métricas principais) --")
for pos in POSITIONS:
    g = medians['global'][pos]
    if pos == 'GK':
        print(f"  {pos}: Save%={g['Save%']:.1f} | CS%={g['CS%']:.1f} | GA_p90={g['GA_p90']:.2f}")
    else:
        print(
            f"  {pos}: Gls_p90={g['Gls_p90']:.2f} | xG_p90={g['xG_p90']:.2f} | "
            f"Ast_p90={g['Ast_p90']:.2f} | Sh_p90={g['Sh_p90']:.2f} | "
            f"Tkl_p90={g['Tkl_p90']:.2f} | Int_p90={g['Int_p90']:.2f}"
        )

print(f"\nArquivos salvos:\n  {OUT_CSV}\n  {OUT_MEDIANS}")
