"""ETL: EA FC 26 → features de elenco por seleção (Copa 2026).

Fonte de jogadores: Supabase `players` + `teams` (convocados Copa 2026)
Fonte de ratings:   data/raw/ea_fc26_official_male.csv
Saída:              data/processed/eafc26_team_features.csv

Matching (por seleção):
  1. Exato normalizado (sem acento, lowercase)
  2. Compacto sem espaços  → resolve nomes coreanos/asiáticos com espaçamento diferente
  3. Alias manual          → apelidos, nomes parciais, romanizações alternativas
  4. Fuzzy token_set_ratio ≥ 85 + validação de posição para scores 85–92

Correções:
  1. Shrinkage (floor=20): seleções com < 20 matched regridem para a média das
     vizinhas no ranking FIFA (±10 posições, threshold ≥ FIFA_STABLE_MIN matched).
     Fallback para média global se vizinhos estáveis insuficientes.
  2. Fallback Catar: média das seleções FIFA vizinhas (rank ±10, n ≥ 15).

Uso (raiz do projeto, venv ativado):
    python scripts/etl_eafc26_team_features.py
"""

import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from supabase import create_client

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ── Caminhos ──────────────────────────────────────────────────────────────────
OFFICIAL_MALE_PATH = Path("data/raw/ea_fc26_official_male.csv")
FIFA_RANKING_PATH  = Path("data/raw/fifa_ranking-2024-04-04.csv")
OUTPUT_PATH        = Path("data/processed/eafc26_team_features.csv")

# ── Parâmetros ────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD       = 85   # score mínimo para match fuzzy
FUZZY_POS_CHECK_MAX   = 92   # acima disso aceita sem checar posição
SAMPLE_WEIGHT_FLOOR   = 20   # mínimo para peso = 1.0
MIN_PLAYERS_FALLBACK  = 11   # abaixo disso: shrinkage forte
QATAR_SAMPLE_WEIGHT   = 0.3
FIFA_NEIGHBOR_WINDOW  = 10
FIFA_STABLE_MIN       = 15   # n mínimo de matched para ser vizinho estável
FIFA_MIN_NBRS         = 3    # mínimo de vizinhos estáveis; abaixo → fallback global

# ── Mapeamentos de seleção ─────────────────────────────────────────────────────
SUPABASE_TO_FIFA = {
    "México":          "MEX", "África do Sul":  "RSA", "Coreia do Sul": "KOR",
    "Tchéquia":        "CZE", "Canadá":         "CAN", "Bósnia":        "BIH",
    "Catar":           "QAT", "Suíça":          "SUI", "Brasil":        "BRA",
    "Marrocos":        "MAR", "Haiti":          "HAI", "Escócia":       "SCO",
    "Estados Unidos":  "USA", "Paraguai":       "PAR", "Austrália":     "AUS",
    "Turquia":         "TUR", "Alemanha":       "GER", "Curaçao":       "CUW",
    "Costa do Marfim": "CIV", "Equador":        "ECU", "Holanda":       "NED",
    "Japão":           "JPN", "Suécia":         "SWE", "Tunísia":       "TUN",
    "Bélgica":         "BEL", "Egito":          "EGY", "Irã":           "IRN",
    "Nova Zelândia":   "NZL", "Espanha":        "ESP", "Cabo Verde":    "CPV",
    "Arábia Saudita":  "KSA", "Uruguai":        "URU", "França":        "FRA",
    "Senegal":         "SEN", "Iraque":         "IRQ", "Noruega":       "NOR",
    "Argentina":       "ARG", "Argélia":        "ALG", "Áustria":       "AUT",
    "Jordânia":        "JOR", "Portugal":       "POR", "Rep. D. Congo": "COD",
    "Uzbequistão":     "UZB", "Colômbia":       "COL", "Inglaterra":    "ENG",
    "Croácia":         "CRO", "Gana":           "GHA", "Panamá":        "PAN",
}
FIFA_TO_SUPABASE = {v: k for k, v in SUPABASE_TO_FIFA.items()}
ALL_48_TEAMS     = set(SUPABASE_TO_FIFA.keys())

# ── Aliases manuais: norm(supabase) → norm(ea_name) ──────────────────────────
# None = forçar miss (nome ambíguo ou jogador não presente no EA)
MANUAL_ALIASES: dict[str, str | None] = {
    # Argentina
    "cuti romero":          "cristian romero",
    "otamendi":             "nicolas otamendi",
    # Brasil
    "gabriel magalhaes":    "gabriel",          # EA usa nome único (CB do Arsenal)
    "danilo santos":        None,               # ambíguo — skip
    # Jordânia (pontuação alta mas posição diverge no EA)
    "musa al taamari":      "musa al tamari",   # al-taamari → al tamari
    # Iraque — transliterações diferentes
    "aymen hussein":        "aimen hussein",
    # Turquia — grafias distintas
    "orkun kokcu":          "orkun kokcu",      # norm já resolve cedilha
    # México
    "joah vasquez":         "johan vasquez",    # variação de nome próprio
    "guillermo ochoa":      None,               # EA tem David Ochoa (diferente)
}

# Placeholder de gol contra no Supabase — ignorar
SKIP_NAMES = {"gol contra", "gol-contra", "own goal"}

SEP = "=" * 72


# ── Normalização ──────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("-", " ")       # hífens → espaço (nomes coreanos/árabes hífen vs espaço)
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm_compact(s: str) -> str:
    """Versão sem espaços — resolve nomes asiáticos com espaçamento diferente."""
    return norm(s).replace(" ", "")


# ── Features ──────────────────────────────────────────────────────────────────

def top_n_mean(series: pd.Series, n: int) -> float:
    vals = series.nlargest(n)
    return float(vals.mean()) if len(vals) > 0 else float("nan")


def calc_team_features(group: pd.DataFrame) -> dict:
    fw  = group[group["pos_category"] == "FW"]["overall"]
    mf  = group[group["pos_category"] == "MF"]["overall"]
    df_ = group[group["pos_category"] == "DF"]["overall"]
    gk  = group[group["pos_category"] == "GK"]["overall"]
    overall = group["overall"]

    return {
        "eafc_atk":      float(fw.mean())    if len(fw)  > 0 else float("nan"),
        "eafc_best_atk": float(fw.max())     if len(fw)  > 0 else float("nan"),
        "eafc_top3_atk": top_n_mean(fw, 3),

        "eafc_mid":      float(mf.mean())    if len(mf)  > 0 else float("nan"),
        "eafc_best_mid": float(mf.max())     if len(mf)  > 0 else float("nan"),
        "eafc_top3_mid": top_n_mean(mf, 3),

        "eafc_def":      float(df_.mean())   if len(df_) > 0 else float("nan"),
        "eafc_best_def": float(df_.max())    if len(df_) > 0 else float("nan"),
        "eafc_top5_def": top_n_mean(df_, 5),

        "eafc_gk":       float(gk.max())     if len(gk)  > 0 else float("nan"),
        "eafc_gk_avg":   float(gk.mean())    if len(gk)  > 0 else float("nan"),

        "eafc_squad":        float(overall.mean()),
        "eafc_best_overall": float(overall.max()),
        "eafc_top11":        top_n_mean(overall, 11),
        "eafc_depth":        float(overall[overall >= 75].mean())
                             if (overall >= 75).any() else float("nan"),
        "eafc_std":          float(overall.std()),
    }


EAFC_FEATURES = [
    "eafc_atk", "eafc_best_atk", "eafc_top3_atk",
    "eafc_mid", "eafc_best_mid", "eafc_top3_mid",
    "eafc_def", "eafc_best_def", "eafc_top5_def",
    "eafc_gk", "eafc_gk_avg",
    "eafc_squad", "eafc_best_overall", "eafc_top11", "eafc_depth", "eafc_std",
]


# ── Matching ──────────────────────────────────────────────────────────────────

def build_ea_index(df_ea: pd.DataFrame, team_name: str):
    """Retorna (rows_list, norm_list, compact_list, ea_df_team)."""
    ea_t = df_ea[df_ea["team_name"] == team_name].copy()
    ea_t["name_norm"]    = ea_t["name"].apply(norm)
    ea_t["name_compact"] = ea_t["name"].apply(norm_compact)
    rows = ea_t.to_dict("records")
    norms    = [r["name_norm"]    for r in rows]
    compacts = [r["name_compact"] for r in rows]
    return rows, norms, compacts


def match_player(
    sb_name: str,
    sb_pos: str,
    ea_rows: list,
    ea_norms: list[str],
    ea_compacts: list[str],
) -> tuple[dict | None, str]:
    """
    Retorna (ea_row | None, método_str).
    Método: 'exact' | 'compact' | 'alias' | 'fuzzy' | 'miss'
    """
    pn      = norm(sb_name)
    pc      = norm_compact(sb_name)

    # 0. Pular placeholder
    if pn in SKIP_NAMES:
        return None, "skip"

    # 1. Exato
    if pn in ea_norms:
        return ea_rows[ea_norms.index(pn)], "exact"

    # 2. Compacto (resolve nomes asiáticos com espaçamento diferente)
    if pc in ea_compacts:
        return ea_rows[ea_compacts.index(pc)], "compact"

    # 3. Alias manual
    alias = MANUAL_ALIASES.get(pn)
    if alias is not None:
        an = norm(alias)
        if an in ea_norms:
            return ea_rows[ea_norms.index(an)], "alias"
        ac = norm_compact(alias)
        if ac in ea_compacts:
            return ea_rows[ea_compacts.index(ac)], "alias"
    elif alias is None and pn in MANUAL_ALIASES:
        return None, "alias-skip"

    # 4. Fuzzy dentro da mesma seleção
    if not ea_norms:
        return None, "miss"

    m = process.extractOne(pn, ea_norms, scorer=fuzz.token_set_ratio)
    if m and m[1] >= FUZZY_THRESHOLD:
        score = m[1]
        idx   = ea_norms.index(m[0])
        ea_r  = ea_rows[idx]
        # Para scores intermediários exige posição compatível
        if score < FUZZY_POS_CHECK_MAX:
            ea_pos = ea_r.get("pos_category", "")
            if ea_pos and sb_pos and ea_pos != sb_pos:
                return None, "miss-pos"
        return ea_r, f"fuzzy({score})"

    return None, "miss"


# ── Ranking FIFA ──────────────────────────────────────────────────────────────

def load_fifa_latest() -> tuple[pd.DataFrame, dict[str, int]]:
    """Retorna (df_latest, rank_lookup) onde rank_lookup: team_name → rank FIFA."""
    df_fifa = pd.read_csv(FIFA_RANKING_PATH)
    latest  = df_fifa[df_fifa["rank_date"] == df_fifa["rank_date"].max()]
    rank_lookup: dict[str, int] = {}
    for team_name, abrv in SUPABASE_TO_FIFA.items():
        row = latest[latest["country_abrv"] == abrv]
        if not row.empty:
            rank_lookup[team_name] = int(row.iloc[0]["rank"])
    return latest, rank_lookup


# ── Shrinkage e fallback ───────────────────────────────────────────────────────

def _neighbor_mean(
    team_name: str,
    rank_lookup: dict[str, int],
    result_raw: pd.DataFrame,
    global_means: dict[str, float],
) -> dict[str, float]:
    """Neighbor mean (raw values) para team_name, ou global_means se insuficiente."""
    if team_name not in rank_lookup:
        return global_means

    team_rank = rank_lookup[team_name]
    lo = team_rank - FIFA_NEIGHBOR_WINDOW
    hi = team_rank + FIFA_NEIGHBOR_WINDOW

    candidate_names = {
        t for t in rank_lookup
        if t != team_name and lo <= rank_lookup[t] <= hi
    }

    stable_nbrs = result_raw[
        result_raw["team"].isin(candidate_names)
        & (result_raw["n_matched"] >= FIFA_STABLE_MIN)
    ]

    if len(stable_nbrs) >= FIFA_MIN_NBRS:
        return {f: float(stable_nbrs[f].mean()) for f in EAFC_FEATURES}

    return global_means


def apply_shrinkage(
    result_raw: pd.DataFrame,
    rank_lookup: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[str, float]]]:
    """
    Aplica shrinkage em todas as features eafc_*.

    Se rank_lookup fornecido: regride para neighbor_mean (±FIFA_NEIGHBOR_WINDOW rank).
    Caso contrário: regride para média global (comportamento anterior).

    Retorna (result, global_means, neighbor_means_by_team).
    """
    result = result_raw.copy()
    result["eafc_sample_weight"] = (
        result["n_matched"] / SAMPLE_WEIGHT_FLOOR
    ).clip(upper=1.0)

    stable = result[result["n_matched"] >= SAMPLE_WEIGHT_FLOOR]
    global_means = {f: float(stable[f].mean()) for f in EAFC_FEATURES}

    # Determina fallback por equipe
    neighbor_means_by_team: dict[str, dict[str, float]] = {}
    for team_name in result["team"]:
        if rank_lookup is not None:
            neighbor_means_by_team[team_name] = _neighbor_mean(
                team_name, rank_lookup, result_raw, global_means
            )
        else:
            neighbor_means_by_team[team_name] = global_means

    # Aplica shrinkage: w * raw + (1-w) * fallback
    raw_vals = {f: result[f].values.copy() for f in EAFC_FEATURES}
    teams    = list(result["team"])
    weights  = result["eafc_sample_weight"].values.copy()

    for feat in EAFC_FEATURES:
        new_col = [
            weights[i] * raw_vals[feat][i]
            + (1 - weights[i]) * neighbor_means_by_team[teams[i]][feat]
            for i in range(len(teams))
        ]
        result[feat] = new_col

    return result, global_means, neighbor_means_by_team


def build_qatar_row(
    result: pd.DataFrame,
    df_fifa_latest: pd.DataFrame,
    rank_lookup: dict[str, int],
) -> dict:
    qatar_rank = rank_lookup.get("Catar")
    if qatar_rank is None:
        qt = df_fifa_latest[df_fifa_latest["country_abrv"] == "QAT"]
        qatar_rank = int(qt.iloc[0]["rank"]) if not qt.empty else 50

    low, high = qatar_rank - FIFA_NEIGHBOR_WINDOW, qatar_rank + FIFA_NEIGHBOR_WINDOW

    neighbor_abrvs = set(
        df_fifa_latest[
            (df_fifa_latest["rank"] >= low) & (df_fifa_latest["rank"] <= high)
        ]["country_abrv"]
    )
    neighbor_names = {
        FIFA_TO_SUPABASE[a] for a in neighbor_abrvs
        if a in FIFA_TO_SUPABASE and a != "QAT"
    }
    stable_nbrs = result[
        result["team"].isin(neighbor_names)
        & (result["n_matched"] >= FIFA_STABLE_MIN)
    ]

    print(f"\n  Catar FIFA rank: {qatar_rank} (janela {low}–{high})")
    print(f"  Vizinhos Copa 2026 estáveis ({len(stable_nbrs)}):")
    for t in sorted(stable_nbrs["team"]):
        n = int(stable_nbrs.loc[stable_nbrs["team"] == t, "n_matched"].iloc[0])
        print(f"    {t} (n={n})")

    return {
        "team": "Catar",
        "n_squad": 0, "n_matched": 0,
        "n_fw": 0, "n_mf": 0, "n_df": 0, "n_gk": 0,
        "eafc_sample_weight": QATAR_SAMPLE_WEIGHT,
        **{feat: float(stable_nbrs[feat].mean()) for feat in EAFC_FEATURES},
    }


# ── Supabase ──────────────────────────────────────────────────────────────────

def load_supabase_squads() -> dict[str, list[dict]]:
    """Retorna {team_name: [{'name': ..., 'position': ...}]}."""
    sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )

    # Carrega players em páginas de 1000
    players: list[dict] = []
    start = 0
    while True:
        r = sb.table("players").select("name,position,team_id").range(start, start + 999).execute()
        players.extend(r.data)
        if len(r.data) < 1000:
            break
        start += 1000

    teams_r = sb.table("teams").select("id,name").execute()
    teams_map = {t["id"]: t["name"] for t in teams_r.data}

    squads: dict[str, list[dict]] = {}
    for p in players:
        team_name = teams_map.get(p["team_id"], "")
        if not team_name:
            continue
        squads.setdefault(team_name, []).append(
            {"name": p["name"], "position": p["position"]}
        )
    return squads


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  ETL — EA FC 26 × Supabase (convocados Copa 2026) → features")
    print(SEP)

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    print(f"\n[1] Carregando EA FC 26: {OFFICIAL_MALE_PATH}")
    df_ea = pd.read_csv(OFFICIAL_MALE_PATH)
    df_ea = df_ea[df_ea["overall"].notna()].copy()
    print(f"    {len(df_ea):,} jogadores ({df_ea['team_name'].nunique()} seleções)")

    print(f"\n[2] Carregando convocados do Supabase...")
    squads = load_supabase_squads()
    total_sb = sum(len(v) for v in squads.values())
    print(f"    {total_sb} convocados em {len(squads)} seleções")

    print(f"\n[3] Carregando ranking FIFA: {FIFA_RANKING_PATH}")
    df_fifa_latest, rank_lookup = load_fifa_latest()
    print(f"    {len(rank_lookup)}/48 seleções Copa 2026 mapeadas no ranking")

    # ── 2. Matching + features por seleção ────────────────────────────────────
    print("\n[4] Matching convocados × EA FC 26 (por seleção)...")
    print()
    header = f"{'Seleção':<20} {'squad':>5} {'match':>5} {'pct':>5}  {'method breakdown'}"
    print(header)
    print("-" * 72)

    rows = []
    for team_name in sorted(ALL_48_TEAMS - {"Catar"}):
        squad = squads.get(team_name, [])
        if not squad:
            print(f"  {'[AVISO] sem squad Supabase:':<20} {team_name}")
            continue

        ea_rows, ea_norms, ea_compacts = build_ea_index(df_ea, team_name)

        matched_rows: list[dict] = []
        method_counts: dict[str, int] = {}
        miss_names: list[str] = []

        for p in squad:
            ea_r, method = match_player(
                p["name"], p["position"], ea_rows, ea_norms, ea_compacts
            )
            method_counts[method] = method_counts.get(method, 0) + 1
            if ea_r is not None:
                matched_rows.append(ea_r)
            elif method not in ("skip", "alias-skip"):
                miss_names.append(p["name"])

        n_squad   = len(squad)
        n_matched = len(matched_rows)
        pct       = 100 * n_matched / n_squad if n_squad else 0

        methods_str = "  ".join(
            f"{k}:{v}" for k, v in sorted(method_counts.items())
            if k not in ("miss", "miss-pos", "alias-skip", "skip")
        )
        extras = []
        for k in ("miss-pos", "alias-skip", "miss"):
            if method_counts.get(k):
                extras.append(f"{k}:{method_counts[k]}")
        extra_str = "  ".join(extras)

        print(
            f"  {team_name:<20} {n_squad:>5} {n_matched:>5} {pct:>4.0f}%  "
            f"skip:{method_counts.get('skip',0)}  {methods_str}  {extra_str}"
        )

        if matched_rows:
            df_matched = pd.DataFrame(matched_rows)
            feats = calc_team_features(df_matched)
        else:
            feats = {f: float("nan") for f in EAFC_FEATURES}

        rows.append({
            "team":      team_name,
            "n_squad":   n_squad,
            "n_matched": n_matched,
            "n_fw":  sum(1 for r in matched_rows if r.get("pos_category") == "FW"),
            "n_mf":  sum(1 for r in matched_rows if r.get("pos_category") == "MF"),
            "n_df":  sum(1 for r in matched_rows if r.get("pos_category") == "DF"),
            "n_gk":  sum(1 for r in matched_rows if r.get("pos_category") == "GK"),
            **feats,
        })

    result_raw = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

    # Preenche NaN posicionais com mediana global (antes do shrinkage)
    for feat in EAFC_FEATURES:
        if result_raw[feat].isna().any():
            result_raw[feat] = result_raw[feat].fillna(result_raw[feat].median())

    # ── 3. Shrinkage ──────────────────────────────────────────────────────────
    print(f"\n[5] Aplicando shrinkage por vizinhança FIFA (floor={SAMPLE_WEIGHT_FLOOR}, "
          f"window=±{FIFA_NEIGHBOR_WINDOW}, stable_min={FIFA_STABLE_MIN})...")

    # Shrinkage global (antes) — apenas para comparação
    result_global, global_means, _ = apply_shrinkage(result_raw, rank_lookup=None)

    # Shrinkage por vizinhança (novo)
    result, global_means, neighbor_means = apply_shrinkage(result_raw, rank_lookup=rank_lookup)

    # Antes/depois: seleções com sample_weight < 0.7
    sw_col     = result["eafc_sample_weight"]
    low_mask   = sw_col < 0.7

    low_teams = result[low_mask][["team", "n_matched", "eafc_sample_weight"]].copy()
    low_teams["squad_antes"]  = result_global.loc[low_mask, "eafc_squad"].values
    low_teams["squad_depois"] = result.loc[low_mask, "eafc_squad"].values
    low_teams["delta"]        = low_teams["squad_depois"] - low_teams["squad_antes"]
    low_teams["fifa_rank"]    = low_teams["team"].map(rank_lookup).fillna(999).astype(int)
    low_teams = low_teams.sort_values("fifa_rank")

    print(f"\n  Comparação shrinkage global → vizinhança (sample_weight < 0.7):")
    print(f"  {'Seleção':<22} {'rank':>5} {'n':>4} {'w':>5}  "
          f"{'squad_antes':>11} {'squad_depois':>12} {'delta':>7}  {'fallback'}")
    print("  " + "-" * 78)
    for _, r in low_teams.iterrows():
        team = r["team"]
        nbm  = neighbor_means.get(team, {})
        used_global = (nbm == {f: global_means[f] for f in EAFC_FEATURES})
        fallback_tag = "global" if used_global else f"viz(rank {r['fifa_rank']}±{FIFA_NEIGHBOR_WINDOW})"
        print(
            f"  {team:<22} {int(r['fifa_rank']):>5} {int(r['n_matched']):>4} "
            f"{r['eafc_sample_weight']:>5.2f}  "
            f"{r['squad_antes']:>11.2f} {r['squad_depois']:>12.2f} "
            f"{r['delta']:>+7.2f}  {fallback_tag}"
        )

    # Confirma que seleções com peso = 1.0 não mudaram
    full_weight = result[result["eafc_sample_weight"] >= 1.0]
    delta_full  = (result.loc[full_weight.index, "eafc_squad"]
                   - result_global.loc[full_weight.index, "eafc_squad"]).abs().max()
    print(f"\n  Seleções com peso=1.0: delta_max eafc_squad = {delta_full:.6f}  "
          f"({'OK — sem mudança' if delta_full < 1e-9 else 'AVISO: houve mudança'})")

    low_cov = result_raw[result_raw["n_matched"] < MIN_PLAYERS_FALLBACK].sort_values("n_matched")
    if not low_cov.empty:
        print(f"\n  Seleções com < {MIN_PLAYERS_FALLBACK} matched (shrinkage forte):")
        for _, r in low_cov.iterrows():
            w = float(r["n_matched"]) / SAMPLE_WEIGHT_FLOOR
            print(f"    {r['team']:<22} n={int(r['n_matched'])}  w={w:.2f}")

    # ── 4. Fallback Catar ─────────────────────────────────────────────────────
    print(f"\n[6] Calculando fallback FIFA para Catar...")
    qatar_row = build_qatar_row(result, df_fifa_latest, rank_lookup)
    result = pd.concat([result, pd.DataFrame([qatar_row])], ignore_index=True)
    result = result.sort_values("team").reset_index(drop=True)

    # ── 5. Salva ──────────────────────────────────────────────────────────────
    col_order = (
        ["team", "n_squad", "n_matched", "n_fw", "n_mf", "n_df", "n_gk",
         "eafc_sample_weight"] + EAFC_FEATURES
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result[col_order].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n[7] Salvo: {OUTPUT_PATH}  ({len(result)} seleções × {len(col_order)} colunas)")

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RELATÓRIO")
    print(SEP)

    covered = set(result["team"])
    missing = ALL_48_TEAMS - covered
    total_matched = int(result[result["team"] != "Catar"]["n_matched"].sum())
    total_squad   = int(result[result["team"] != "Catar"]["n_squad"].sum())
    print(f"\nCobertura global: {total_matched}/{total_squad} convocados matched "
          f"({100*total_matched/total_squad:.1f}%)")
    print(f"Seleções: {len(covered)}/{len(ALL_48_TEAMS)} (incluindo Catar via fallback)")
    if missing:
        print(f"  Sem dados: {sorted(missing)}")

    print(f"\nTop 8 por eafc_squad (após shrinkage):")
    top8 = result.nlargest(8, "eafc_squad")[
        ["team", "n_matched", "eafc_sample_weight", "eafc_squad", "eafc_top11", "eafc_atk", "eafc_def", "eafc_gk"]
    ]
    print(top8.to_string(index=False))

    print(f"\nTabela completa — eafc_squad (pós shrinkage por vizinhança):")
    display = result.sort_values("eafc_squad", ascending=False)[
        ["team", "n_squad", "n_matched", "eafc_sample_weight",
         "eafc_squad", "eafc_atk", "eafc_mid", "eafc_def", "eafc_gk"]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
