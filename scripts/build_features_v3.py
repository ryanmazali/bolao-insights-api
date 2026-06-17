"""build_features_v3.py — Pipeline de features para treino dos modelos v3.

Diferenças em relação ao build_features.py (v2):
  - Features de elenco: fm23_* substituídas por eafc_* (EA FC 26, convocados)
  - Novas features: eafc_mid/best_mid/top3_mid, eafc_gk_avg (sem equivalente FM23)
  - Recency weight: curva por degraus em vez de exponencial contínuo
  - Targets: goals_home e goals_away separados (em vez de result + total_goals)
  - sample_weight_final = recency_weight × tournament_weight × elo_weight

Saída:
  data/processed/training_data_v3.csv
  data/processed/feature_columns_v3.json

Uso (raiz do projeto, venv ativado):
    python scripts/build_features_v3.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Caminhos ──────────────────────────────────────────────────────────────────
RESULTS_PATH      = Path("data/raw/results.csv")
EAFC_FEATURES_PATH = Path("data/processed/eafc26_team_features.csv")
FORMER_NAMES_PATH = Path("data/raw/former_names.csv")
ELO_PATH          = Path("data/raw/eloratings.csv")
FIFA_RANKING_FILES = [
    Path("data/raw/fifa_ranking-2023-07-20.csv"),
    Path("data/raw/fifa_ranking-2024-04-04.csv"),
    Path("data/raw/fifa_ranking-2024-06-20.csv"),
]
OUTPUT_CSV        = Path("data/processed/training_data_v3.csv")
OUTPUT_JSON       = Path("data/processed/feature_columns_v3.json")

# ── Configurações ──────────────────────────────────────────────────────────────
CUTOFF_DATE    = "2006-01-01"
REFERENCE_DATE = pd.Timestamp("2026-06-10")
GLOBAL_AVG_ELO = 1750.0

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                       1.00,
    "UEFA Euro":                            0.90,
    "Copa América":                         0.90,
    "Africa Cup of Nations":                0.85,
    "UEFA Nations League":                  0.80,
    "CONCACAF Gold Cup":                    0.75,
    "AFC Asian Cup":                        0.75,
    "FIFA Confederations Cup":              0.75,
    "FIFA World Cup qualification":         0.70,
    "UEFA Euro qualification":              0.65,
    "Copa América qualification":           0.65,
    "African Cup of Nations qualification": 0.60,
    "CONCACAF Nations League":              0.60,
    "Friendly":                             0.25,
}

NAME_MAPPING = {
    "IR Iran":                    "Iran",
    "Korea Republic":             "South Korea",
    "Korea DPR":                  "North Korea",
    "Côte d'Ivoire":              "Ivory Coast",
    "Cote d'Ivoire":              "Ivory Coast",
    "Türkiye":                    "Turkey",
    "China PR":                   "China",
    "Bosnia and Herzegovina":     "Bosnia-Herzegovina",
    "United States":              "USA",
    "Czechia":                    "Czech Republic",
    "Congo DR":                   "DR Congo",
    "Cabo Verde":                 "Cape Verde",
    "Curaçao":                    "Curacao",
    "Democratic Republic of Congo": "DR Congo",
    "North Macedonia":            "North Macedonia",
    "Saint Kitts and Nevis":      "St Kitts and Nevis",
    "São Tomé and Príncipe":      "Sao Tome and Principe",
}

# Supabase PT → EN (mesmo mapeamento do build_features.py v2)
SUPABASE_PT_TO_EN = {
    "Alemanha":       "Germany",        "Argentina":    "Argentina",
    "Argélia":        "Algeria",        "Arábia Saudita": "Saudi Arabia",
    "Austrália":      "Australia",      "Brasil":       "Brazil",
    "Bélgica":        "Belgium",        "Bósnia":       "Bosnia-Herzegovina",
    "Cabo Verde":     "Cape Verde",     "Canadá":       "Canada",
    "Catar":          "Qatar",          "Colômbia":     "Colombia",
    "Coreia do Sul":  "South Korea",    "Costa do Marfim": "Ivory Coast",
    "Croácia":        "Croatia",        "Curaçao":      "Curacao",
    "Egito":          "Egypt",          "Equador":      "Ecuador",
    "Escócia":        "Scotland",       "Espanha":      "Spain",
    "Estados Unidos": "USA",            "França":       "France",
    "Gana":           "Ghana",          "Haiti":        "Haiti",
    "Holanda":        "Netherlands",    "Inglaterra":   "England",
    "Iraque":         "Iraq",           "Irã":          "Iran",
    "Japão":          "Japan",          "Jordânia":     "Jordan",
    "Marrocos":       "Morocco",        "México":       "Mexico",
    "Noruega":        "Norway",         "Nova Zelândia": "New Zealand",
    "Panamá":         "Panama",         "Paraguai":     "Paraguay",
    "Portugal":       "Portugal",       "Rep. D. Congo": "DR Congo",
    "Senegal":        "Senegal",        "Suécia":       "Sweden",
    "Suíça":          "Switzerland",    "Tchéquia":     "Czech Republic",
    "Tunísia":        "Tunisia",        "Turquia":      "Turkey",
    "Uruguai":        "Uruguay",        "Uzbequistão":  "Uzbekistan",
    "África do Sul":  "South Africa",   "Áustria":      "Austria",
}

EAFC_METRICS = [
    "eafc_atk", "eafc_best_atk", "eafc_top3_atk",
    "eafc_mid", "eafc_best_mid", "eafc_top3_mid",
    "eafc_def", "eafc_best_def", "eafc_top5_def",
    "eafc_gk", "eafc_gk_avg",
    "eafc_squad", "eafc_best_overall", "eafc_top11", "eafc_depth", "eafc_std",
]

SEP = "=" * 70


# ── Normalização ──────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return NAME_MAPPING.get(name, name)


# ── Recency weight v3 (curva por degraus) ─────────────────────────────────────

def recency_weight_v3(match_date: pd.Timestamp, reference: pd.Timestamp) -> float:
    days_ago = (reference - match_date).days
    if days_ago <= 365:
        return 1.00
    if days_ago <= 730:
        return 0.85
    if days_ago <= 1460:
        return 0.65
    if days_ago <= 2920:
        return 0.40
    return 0.20


def get_tournament_weight(tournament: str) -> float:
    t_lower = tournament.lower()
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t_lower:
            return w
    return 0.50


# ── Carga de dados ────────────────────────────────────────────────────────────

def load_elo_ratings():
    df = pd.read_csv(ELO_PATH)
    df["rank_date"]    = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["country_full"] = df["team"].apply(normalize_name)
    df = df.rename(columns={"rating": "total_points"})
    df = df.dropna(subset=["total_points"]).sort_values("rank_date")
    current_elo = df.groupby("country_full")["total_points"].last().to_dict()
    print(f"  [Elo] {len(df):,} registros históricos")
    return df[["country_full", "rank_date", "total_points"]], current_elo


def load_fifa_rankings():
    dfs = [pd.read_csv(f, parse_dates=["rank_date"])
           for f in FIFA_RANKING_FILES if f.exists()]
    rankings = pd.concat(dfs, ignore_index=True)
    rankings["country_full"] = rankings["country_full"].apply(normalize_name)
    rankings = rankings.sort_values("rank_date")

    current_rankings = {
        "Argentina": 1877.27, "France": 1870.70, "Spain": 1874.71,
        "England": 1827.05, "Brazil": 1765.86, "Belgium": 1742.24,
        "Netherlands": 1753.57, "Portugal": 1766.18, "Germany": 1735.77,
        "Croatia": 1714.87, "Italy": 1704.73, "Colombia": 1698.35,
        "Mexico": 1687.48, "Senegal": 1684.07, "Uruguay": 1673.07,
        "USA": 1671.23, "Japan": 1661.58, "Switzerland": 1650.06,
        "Iran": 1619.58, "Denmark": 1619.47, "Turkey": 1605.73,
        "Ecuador": 1598.52, "Austria": 1597.40, "South Korea": 1591.63,
        "Nigeria": 1586.69, "Australia": 1579.34, "Algeria": 1571.03,
        "Egypt": 1562.37, "Canada": 1559.48, "Norway": 1557.44,
        "Ukraine": 1549.29, "Ivory Coast": 1540.87, "Panama": 1539.16,
        "Russia": 1529.60, "Poland": 1526.18, "Wales": 1516.95,
        "Sweden": 1509.79, "Hungary": 1506.39, "Czech Republic": 1505.74,
        "Paraguay": 1505.35, "Scotland": 1503.34, "Serbia": 1502.13,
        "Cameroon": 1481.24, "Tunisia": 1476.41, "DR Congo": 1474.43,
        "Slovakia": 1473.66, "Greece": 1473.19, "Venezuela": 1469.18,
        "Uzbekistan": 1458.73, "Chile": 1458.20, "Peru": 1457.69,
        "Costa Rica": 1457.00, "Romania": 1455.89, "Mali": 1455.59,
        "Qatar": 1450.31, "Iraq": 1446.28, "Morocco": 1755.10,
        "South Africa": 1428.38, "Saudi Arabia": 1423.88,
        "Ghana": 1346.88, "Haiti": 1293.10, "Curacao": 1294.77,
        "Cape Verde": 1371.11, "New Zealand": 1275.58,
        "Bosnia-Herzegovina": 1387.22, "Jordan": 1387.74,
    }
    print(f"  [FIFA] {len(rankings):,} registros históricos")
    return rankings, current_rankings


def load_results(elo_df, current_elo):
    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"] >= CUTOFF_DATE].copy()
    df["home_team"] = df["home_team"].apply(normalize_name)
    df["away_team"] = df["away_team"].apply(normalize_name)

    df["result"]      = np.where(df["home_score"] > df["away_score"], "H",
                        np.where(df["home_score"] < df["away_score"], "A", "D"))
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["btts"]        = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)

    df["tournament_weight"] = df["tournament"].apply(get_tournament_weight)
    df["recency_weight"]    = df["date"].apply(
        lambda d: recency_weight_v3(d, REFERENCE_DATE)
    )

    df["home_elo"] = df.apply(
        lambda r: get_elo_at_date(elo_df, current_elo, r["home_team"], r["date"]), axis=1
    )
    df["away_elo"] = df.apply(
        lambda r: get_elo_at_date(elo_df, current_elo, r["away_team"], r["date"]), axis=1
    )
    df["avg_elo"]    = (df["home_elo"] + df["away_elo"]) / 2
    df["elo_weight"] = df["avg_elo"] / GLOBAL_AVG_ELO
    df["match_weight"]   = df["tournament_weight"] * df["elo_weight"]
    df["sample_weight"]  = df["match_weight"] * df["recency_weight"]

    print(f"  [Results] {len(df):,} partidas  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  [Results] {df['tournament'].nunique()} torneios únicos")
    return df.sort_values("date").reset_index(drop=True)


def load_eafc_features():
    df = pd.read_csv(EAFC_FEATURES_PATH)
    df["team_en"] = df["team"].map(SUPABASE_PT_TO_EN).fillna(df["team"])

    former = pd.read_csv(FORMER_NAMES_PATH)
    former_to_current = {
        row["former"]: normalize_name(row["current"])
        for _, row in former.iterrows()
    }

    eafc_lookup  = df.set_index("team_en")[EAFC_METRICS].to_dict("index")
    eafc_defaults = df[EAFC_METRICS].mean().to_dict()

    covered = set(eafc_lookup.keys())
    print(f"  [EAFC] {len(covered)} seleções com features de elenco")
    return eafc_lookup, eafc_defaults, former_to_current


def get_eafc(team, eafc_lookup, eafc_defaults, former_to_current):
    if team in eafc_lookup:
        return eafc_lookup[team], True
    resolved = former_to_current.get(team)
    if resolved in eafc_lookup:
        return eafc_lookup[resolved], True
    return eafc_defaults, False


# ── Índice de rankings ─────────────────────────────────────────────────────────

_RANKING_INDEX_CACHE: dict = {}


def _get_ranking_index(rankings_df: pd.DataFrame) -> dict:
    key = id(rankings_df)
    if key not in _RANKING_INDEX_CACHE:
        idx = {}
        for team, grp in rankings_df.groupby("country_full"):
            g = grp.sort_values("rank_date")
            idx[team] = (
                g["rank_date"].values.astype("datetime64[ns]"),
                g["total_points"].values,
            )
        _RANKING_INDEX_CACHE[key] = idx
    return _RANKING_INDEX_CACHE[key]


def get_ranking_at_date(rankings_df, current_rankings, team, date):
    idx = _get_ranking_index(rankings_df)
    entry = idx.get(team)
    if entry is not None:
        dates, points = entry
        pos = np.searchsorted(dates, np.datetime64(date), side="right")
        if pos > 0:
            return points[pos - 1]
    return current_rankings.get(team, 1200.0)


def get_elo_at_date(elo_df, current_elo, team, date):
    idx = _get_ranking_index(elo_df)
    entry = idx.get(team)
    if entry is not None:
        dates, points = entry
        pos = np.searchsorted(dates, np.datetime64(date), side="right")
        if pos > 0:
            return points[pos - 1]
    return current_elo.get(team, GLOBAL_AVG_ELO)


# ── Features por time ─────────────────────────────────────────────────────────

def compute_team_features(df, team, before_date, rankings_df, current_rankings,
                           window=20):
    home_m = df[(df["home_team"] == team) & (df["date"] < before_date)].copy()
    home_m["is_home"]       = True
    home_m["goals_for"]     = home_m["home_score"]
    home_m["goals_against"] = home_m["away_score"]
    home_m["won"]           = (home_m["result"] == "H").astype(int)
    home_m["drew"]          = (home_m["result"] == "D").astype(int)
    home_m["opp_team"]      = home_m["away_team"]

    away_m = df[(df["away_team"] == team) & (df["date"] < before_date)].copy()
    away_m["is_home"]       = False
    away_m["goals_for"]     = away_m["away_score"]
    away_m["goals_against"] = away_m["home_score"]
    away_m["won"]           = (away_m["result"] == "A").astype(int)
    away_m["drew"]          = (away_m["result"] == "D").astype(int)
    away_m["opp_team"]      = away_m["home_team"]

    all_m = pd.concat([home_m, away_m]).sort_values("date")
    if len(all_m) == 0:
        return None

    all_m["opp_points"] = all_m.apply(
        lambda r: get_ranking_at_date(rankings_df, current_rankings,
                                      r["opp_team"], r["date"]), axis=1
    )
    all_m["rec_w"]   = all_m["date"].apply(
        lambda d: recency_weight_v3(d, before_date)
    )
    all_m["total_w"] = all_m["rec_w"] * all_m["tournament_weight"]

    w = all_m["total_w"].values
    opp_avg = np.average(all_m["opp_points"], weights=w)

    def sos_adjust(avg, opp_pts):
        norm = (opp_pts - 1200) / 700
        return avg * (0.6 + norm * 0.8)

    goals_for_avg     = np.average(all_m["goals_for"], weights=w)
    goals_against_avg = np.average(all_m["goals_against"], weights=w)
    win_rate          = np.average(all_m["won"], weights=w)
    draw_rate         = np.average(all_m["drew"], weights=w)
    btts_rate         = np.average(
        ((all_m["goals_for"] > 0) & (all_m["goals_against"] > 0)).astype(int), weights=w
    )
    clean_sheet_rate  = np.average((all_m["goals_against"] == 0).astype(int), weights=w)

    recent = all_m.tail(window)
    w_r    = recent["total_w"].values
    form_gf   = np.average(recent["goals_for"],     weights=w_r) if len(recent) > 0 else goals_for_avg
    form_ga   = np.average(recent["goals_against"], weights=w_r) if len(recent) > 0 else goals_against_avg
    form_wr   = np.average(recent["won"],           weights=w_r) if len(recent) > 0 else win_rate

    last5  = all_m.tail(5)
    w5     = last5["total_w"].values
    form5_pts = np.average(last5["won"] * 3 + last5["drew"], weights=w5) / 3.0 \
                if len(last5) > 0 else 0.5

    hmatch  = all_m[all_m["is_home"]]
    amatch  = all_m[~all_m["is_home"]]
    nmatch  = all_m[all_m["neutral"]]
    wr_home    = np.average(hmatch["won"], weights=hmatch["total_w"]) if len(hmatch) > 0 else win_rate
    wr_away    = np.average(amatch["won"], weights=amatch["total_w"]) if len(amatch) > 0 else win_rate
    wr_neutral = np.average(nmatch["won"], weights=nmatch["total_w"]) if len(nmatch) > 0 else win_rate

    fifa_pts = get_ranking_at_date(rankings_df, current_rankings, team, before_date)

    return {
        "n_matches":        len(all_m),
        "fifa_points":      fifa_pts,
        "goals_for_avg":    goals_for_avg,
        "goals_against_avg": goals_against_avg,
        "goal_diff_avg":    goals_for_avg - goals_against_avg,
        "win_rate":         win_rate,
        "draw_rate":        draw_rate,
        "btts_rate":        btts_rate,
        "clean_sheet_rate": clean_sheet_rate,
        "form_goals_for":   form_gf,
        "form_goals_against": form_ga,
        "form_win_rate":    form_wr,
        "form5_pts":        form5_pts,
        "win_rate_home":    wr_home,
        "win_rate_away":    wr_away,
        "win_rate_neutral": wr_neutral,
        "avg_opp_points":   opp_avg,
        "sos_goals_for":    sos_adjust(goals_for_avg, opp_avg),
        "sos_goals_against": sos_adjust(goals_against_avg, opp_avg),
        "sos_form_goals":   sos_adjust(form_gf, opp_avg),
    }


def compute_h2h(df, home_team, away_team, before_date, n=10):
    h2h = df[
        (((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
         ((df["home_team"] == away_team) & (df["away_team"] == home_team))) &
        (df["date"] < before_date)
    ].tail(n)

    if len(h2h) == 0:
        return {"h2h_home_wins": 0.33, "h2h_draws": 0.33,
                "h2h_away_wins": 0.33, "h2h_goals_avg": 2.5, "h2h_n": 0}

    hw = dw = aw = 0
    for _, r in h2h.iterrows():
        if r["home_team"] == home_team:
            if r["result"] == "H":   hw += 1
            elif r["result"] == "D": dw += 1
            else:                    aw += 1
        else:
            if r["result"] == "A":   hw += 1
            elif r["result"] == "D": dw += 1
            else:                    aw += 1

    n_g = len(h2h)
    return {
        "h2h_home_wins": hw / n_g,
        "h2h_draws":     dw / n_g,
        "h2h_away_wins": aw / n_g,
        "h2h_goals_avg": h2h["total_goals"].mean(),
        "h2h_n":         n_g,
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_features(df, rankings_df, current_rankings,
                   eafc_lookup, eafc_defaults, former_to_current):
    print(f"  Construindo features para {len(df):,} partidas...")
    no_eafc = set()
    rows = []

    for i, match in df.iterrows():
        if i % 1000 == 0:
            print(f"    {i:,}/{len(df):,}...")

        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        hf = compute_team_features(df, home, date, rankings_df, current_rankings)
        af = compute_team_features(df, away, date, rankings_df, current_rankings)
        if hf is None or af is None:
            continue

        h2h = compute_h2h(df, home, away, date)

        h_eafc, h_found = get_eafc(home, eafc_lookup, eafc_defaults, former_to_current)
        a_eafc, a_found = get_eafc(away, eafc_lookup, eafc_defaults, former_to_current)
        if not h_found: no_eafc.add(home)
        if not a_found: no_eafc.add(away)

        row = {
            # Identificação (não entra no feature set)
            "date":       date,
            "home_team":  home,
            "away_team":  away,
            "tournament": match["tournament"],
            "is_neutral": int(match["neutral"]),

            # Pesos
            "tournament_weight":  match["tournament_weight"],
            "recency_weight":     match["recency_weight"],
            "elo_weight":         match["elo_weight"],
            "sample_weight":      match["sample_weight"],
            "home_elo":           match["home_elo"],
            "away_elo":           match["away_elo"],
            "elo_diff_abs":       abs(match["home_elo"] - match["away_elo"]),

            # ── Features HOME históricas ──────────────────────────────
            "home_fifa_points":        hf["fifa_points"],
            "home_goals_for":          hf["goals_for_avg"],
            "home_goals_against":      hf["goals_against_avg"],
            "home_goal_diff":          hf["goal_diff_avg"],
            "home_win_rate":           hf["win_rate"],
            "home_draw_rate":          hf["draw_rate"],
            "home_btts_rate":          hf["btts_rate"],
            "home_clean_sheet":        hf["clean_sheet_rate"],
            "home_form_goals_for":     hf["form_goals_for"],
            "home_form_goals_against": hf["form_goals_against"],
            "home_form_win_rate":      hf["form_win_rate"],
            "home_form5_pts":          hf["form5_pts"],
            "home_win_rate_home":      hf["win_rate_home"],
            "home_win_rate_neutral":   hf["win_rate_neutral"],
            "home_avg_opp_points":     hf["avg_opp_points"],
            "home_sos_goals_for":      hf["sos_goals_for"],
            "home_sos_goals_against":  hf["sos_goals_against"],
            "home_sos_form":           hf["sos_form_goals"],

            # ── Features HOME eafc ────────────────────────────────────
            "home_eafc_atk":          h_eafc["eafc_atk"],
            "home_eafc_best_atk":     h_eafc["eafc_best_atk"],
            "home_eafc_top3_atk":     h_eafc["eafc_top3_atk"],
            "home_eafc_mid":          h_eafc["eafc_mid"],
            "home_eafc_best_mid":     h_eafc["eafc_best_mid"],
            "home_eafc_top3_mid":     h_eafc["eafc_top3_mid"],
            "home_eafc_def":          h_eafc["eafc_def"],
            "home_eafc_best_def":     h_eafc["eafc_best_def"],
            "home_eafc_top5_def":     h_eafc["eafc_top5_def"],
            "home_eafc_gk":           h_eafc["eafc_gk"],
            "home_eafc_gk_avg":       h_eafc["eafc_gk_avg"],
            "home_eafc_squad":        h_eafc["eafc_squad"],
            "home_eafc_best_overall": h_eafc["eafc_best_overall"],
            "home_eafc_top11":        h_eafc["eafc_top11"],
            "home_eafc_depth":        h_eafc["eafc_depth"],
            "home_eafc_std":          h_eafc["eafc_std"],

            # ── Features AWAY históricas ──────────────────────────────
            "away_fifa_points":        af["fifa_points"],
            "away_goals_for":          af["goals_for_avg"],
            "away_goals_against":      af["goals_against_avg"],
            "away_goal_diff":          af["goal_diff_avg"],
            "away_win_rate":           af["win_rate"],
            "away_draw_rate":          af["draw_rate"],
            "away_btts_rate":          af["btts_rate"],
            "away_clean_sheet":        af["clean_sheet_rate"],
            "away_form_goals_for":     af["form_goals_for"],
            "away_form_goals_against": af["form_goals_against"],
            "away_form_win_rate":      af["form_win_rate"],
            "away_form5_pts":          af["form5_pts"],
            "away_win_rate_away":      af["win_rate_away"],
            "away_win_rate_neutral":   af["win_rate_neutral"],
            "away_avg_opp_points":     af["avg_opp_points"],
            "away_sos_goals_for":      af["sos_goals_for"],
            "away_sos_goals_against":  af["sos_goals_against"],
            "away_sos_form":           af["sos_form_goals"],

            # ── Features AWAY eafc ────────────────────────────────────
            "away_eafc_atk":          a_eafc["eafc_atk"],
            "away_eafc_best_atk":     a_eafc["eafc_best_atk"],
            "away_eafc_top3_atk":     a_eafc["eafc_top3_atk"],
            "away_eafc_mid":          a_eafc["eafc_mid"],
            "away_eafc_best_mid":     a_eafc["eafc_best_mid"],
            "away_eafc_top3_mid":     a_eafc["eafc_top3_mid"],
            "away_eafc_def":          a_eafc["eafc_def"],
            "away_eafc_best_def":     a_eafc["eafc_best_def"],
            "away_eafc_top5_def":     a_eafc["eafc_top5_def"],
            "away_eafc_gk":           a_eafc["eafc_gk"],
            "away_eafc_gk_avg":       a_eafc["eafc_gk_avg"],
            "away_eafc_squad":        a_eafc["eafc_squad"],
            "away_eafc_best_overall": a_eafc["eafc_best_overall"],
            "away_eafc_top11":        a_eafc["eafc_top11"],
            "away_eafc_depth":        a_eafc["eafc_depth"],
            "away_eafc_std":          a_eafc["eafc_std"],

            # ── Diferenciais ──────────────────────────────────────────
            "diff_fifa_points":      hf["fifa_points"]    - af["fifa_points"],
            "diff_goals_for":        hf["goals_for_avg"]  - af["goals_for_avg"],
            "diff_goals_against":    hf["goals_against_avg"] - af["goals_against_avg"],
            "diff_win_rate":         hf["win_rate"]       - af["win_rate"],
            "diff_form_win_rate":    hf["form_win_rate"]  - af["form_win_rate"],
            "diff_form5":            hf["form5_pts"]      - af["form5_pts"],
            "diff_sos_goals":        hf["sos_goals_for"]  - af["sos_goals_for"],
            "diff_sos_form":         hf["sos_form_goals"] - af["sos_form_goals"],
            "diff_avg_opp":          hf["avg_opp_points"] - af["avg_opp_points"],
            "eafc_atk_diff":         h_eafc["eafc_atk"]        - a_eafc["eafc_atk"],
            "eafc_def_diff":         h_eafc["eafc_def"]        - a_eafc["eafc_def"],
            "eafc_mid_diff":         h_eafc["eafc_mid"]        - a_eafc["eafc_mid"],
            "eafc_gk_diff":          h_eafc["eafc_gk"]         - a_eafc["eafc_gk"],
            "eafc_squad_diff":       h_eafc["eafc_squad"]      - a_eafc["eafc_squad"],
            "eafc_best_overall_diff": h_eafc["eafc_best_overall"] - a_eafc["eafc_best_overall"],
            "eafc_top3_atk_diff":    h_eafc["eafc_top3_atk"]   - a_eafc["eafc_top3_atk"],
            "eafc_top5_def_diff":    h_eafc["eafc_top5_def"]   - a_eafc["eafc_top5_def"],

            # ── H2H ───────────────────────────────────────────────────
            "h2h_home_wins": h2h["h2h_home_wins"],
            "h2h_draws":     h2h["h2h_draws"],
            "h2h_away_wins": h2h["h2h_away_wins"],
            "h2h_goals_avg": h2h["h2h_goals_avg"],
            "h2h_n":         h2h["h2h_n"],

            # ── Targets ───────────────────────────────────────────────
            "goals_home":  int(match["home_score"]),
            "goals_away":  int(match["away_score"]),
            # Mantidos para retrocompatibilidade / análise
            "result":      match["result"],
            "total_goals": match["total_goals"],
            "btts":        match["btts"],
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print(f"  {len(result_df):,} partidas com features completas")
    if no_eafc:
        print(f"  [AVISO] {len(no_eafc)} times sem eafc — usaram defaults: "
              f"{sorted(no_eafc)[:10]}{'...' if len(no_eafc) > 10 else ''}")
    return result_df, no_eafc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  build_features_v3.py — Pipeline de features Copa 2026")
    print(SEP)

    print("\n[1] Carregando dados base...")
    elo_df, current_elo       = load_elo_ratings()
    rankings_df, current_rnk  = load_fifa_rankings()
    df                        = load_results(elo_df, current_elo)
    eafc_lookup, eafc_def, former_to_current = load_eafc_features()

    print("\n[2] Construindo features...")
    features, no_eafc = build_features(
        df, rankings_df, current_rnk,
        eafc_lookup, eafc_def, former_to_current,
    )

    # ── Feature columns (excluindo targets e metadados) ───────────────────────
    META_COLS   = {"date", "home_team", "away_team", "tournament",
                   "tournament_weight", "recency_weight", "elo_weight",
                   "sample_weight", "home_elo", "away_elo", "elo_diff_abs"}
    TARGET_COLS = {"goals_home", "goals_away", "result", "total_goals", "btts"}
    FEATURE_COLS = [c for c in features.columns
                    if c not in META_COLS and c not in TARGET_COLS
                    and c not in ("is_neutral",)]
    FEATURE_COLS = ["is_neutral"] + FEATURE_COLS   # is_neutral entra no feature set

    # ── Salva ─────────────────────────────────────────────────────────────────
    print("\n[3] Salvando...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  CSV: {OUTPUT_CSV}  ({features.shape[0]:,} linhas × {features.shape[1]} colunas)")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {OUTPUT_JSON}  ({len(FEATURE_COLS)} features)")

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RELATÓRIO FINAL")
    print(SEP)

    print(f"\nShape dataset: {features.shape}")
    print(f"Total features (sem targets/metadados): {len(FEATURE_COLS)}")

    eafc_feats = [c for c in FEATURE_COLS if "eafc" in c]
    hist_feats  = [c for c in FEATURE_COLS if c not in eafc_feats and c not in ("is_neutral", "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_goals_avg", "h2h_n")]
    diff_feats  = [c for c in FEATURE_COLS if c.startswith("diff_") or c.endswith("_diff")]
    print(f"  Históricas (Elo/FIFA/forma/SOS): {len(hist_feats)}")
    print(f"  EAFC (home+away+diff):           {len(eafc_feats)}")
    print(f"  H2H:                             {len([c for c in FEATURE_COLS if c.startswith('h2h')])}")
    print(f"  Contexto (is_neutral):           1")

    print("\n── Distribuição de goals_home ──")
    print(f"  média={features['goals_home'].mean():.3f}  "
          f"std={features['goals_home'].std():.3f}  "
          f"max={features['goals_home'].max():.0f}")
    vc_h = features["goals_home"].value_counts().sort_index()
    for g, n in vc_h[vc_h.index <= 6].items():
        print(f"  {g} gols: {n:,} partidas ({100*n/len(features):.1f}%)")

    print("\n── Distribuição de goals_away ──")
    print(f"  média={features['goals_away'].mean():.3f}  "
          f"std={features['goals_away'].std():.3f}  "
          f"max={features['goals_away'].max():.0f}")
    vc_a = features["goals_away"].value_counts().sort_index()
    for g, n in vc_a[vc_a.index <= 6].items():
        print(f"  {g} gols: {n:,} partidas ({100*n/len(features):.1f}%)")

    print("\n── Partidas e sample_weight médio por período ──")
    features["year"] = features["date"].dt.year
    per = features.groupby("year").agg(
        n=("goals_home", "count"),
        sw_mean=("sample_weight", "mean"),
        sw_max=("sample_weight", "max"),
    )
    per["período"] = per.index.astype(str)
    for yr, row in per.iterrows():
        print(f"  {yr}: {int(row['n']):>5} partidas  sw_mean={row['sw_mean']:.3f}  sw_max={row['sw_max']:.3f}")

    print("\n── Recency weight v3 — distribuição ──")
    rw_bins = {
        "≤1 ano (1.00)":  (features["recency_weight"] == 1.00).sum(),
        "1-2 anos (0.85)": (features["recency_weight"] == 0.85).sum(),
        "2-4 anos (0.65)": (features["recency_weight"] == 0.65).sum(),
        "4-8 anos (0.40)": (features["recency_weight"] == 0.40).sum(),
        ">8 anos (0.20)":  (features["recency_weight"] == 0.20).sum(),
    }
    for label, cnt in rw_bins.items():
        print(f"  {label}: {cnt:,} partidas")

    print("\n── Top 10 features (primeiras por nome) ──")
    eafc_sample = [c for c in FEATURE_COLS if "eafc" in c][:10]
    for c in eafc_sample:
        print(f"  {c}")

    print("\n── Confirmação: todas as eafc_* presentes ──")
    for m in EAFC_METRICS:
        h_col = f"home_{m}"
        a_col = f"away_{m}"
        ok_h = "✓" if h_col in FEATURE_COLS else "✗"
        ok_a = "✓" if a_col in FEATURE_COLS else "✗"
        print(f"  {ok_h} {h_col:<30}  {ok_a} {a_col}")

    if no_eafc:
        print(f"\n── Times sem eafc (usaram média global, n={len(no_eafc)}) ──")
        for t in sorted(no_eafc)[:20]:
            print(f"  {t}")
        if len(no_eafc) > 20:
            print(f"  ... e mais {len(no_eafc)-20}")

    # Partidas perdidas por falta de histórico (home_f ou away_f = None)
    total_raw = len(df)
    lost = total_raw - len(features)
    print(f"\n── Partidas perdidas por falta de histórico ──")
    print(f"  Partidas brutas: {total_raw:,}")
    print(f"  Com features:    {len(features):,}")
    print(f"  Perdidas:        {lost:,}  ({100*lost/total_raw:.1f}%)")

    print(f"\n{SEP}")
    print("  CONCLUÍDO")
    print(SEP)


if __name__ == "__main__":
    main()
