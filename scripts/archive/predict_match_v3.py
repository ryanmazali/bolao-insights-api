"""predict_match_v3.py — Predição completa para uma partida usando pipeline v3.

Roda a predição completa:
  - Features históricas via compute_team_features (ELO, FIFA, forma, SOS, H2H)
  - Features EAFC26 dos convocados
  - Modelos XGBoost Poisson v3 (goals_home + goals_away)
  - Correção Dixon-Coles (rho=-0.20)
  - Probabilidades de resultado + mercados

Uso:
    python scripts/predict_match_v3.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Adiciona raiz do projeto ao path para importar de scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_features_v3 import (
    NAME_MAPPING,
    SUPABASE_PT_TO_EN,
    TOURNAMENT_WEIGHTS,
    EAFC_METRICS,
    REFERENCE_DATE,
    GLOBAL_AVG_ELO,
    normalize_name,
    recency_weight_v3,
    get_tournament_weight,
    load_elo_ratings,
    load_fifa_rankings,
    load_results,
    load_eafc_features,
    compute_team_features,
    compute_h2h,
    get_eafc,
    get_elo_at_date,
)

SEP = "=" * 70

# ── Configuração da partida ───────────────────────────────────────────────────
HOME_TEAM  = "Austria"
AWAY_TEAM  = "Jordan"
MATCH_DATE = pd.Timestamp("2026-06-17")  # hoje / data do jogo
TOURNAMENT = "FIFA World Cup"
IS_NEUTRAL = 1  # Copa 2026 em campo neutro (USA/Canada/Mexico)

# ── Caminhos de modelos ───────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "models"
MODEL_HOME   = MODELS_DIR / "model_goals_home_v3.pkl"
MODEL_AWAY   = MODELS_DIR / "model_goals_away_v3.pkl"
FEAT_JSON    = MODELS_DIR / "feature_columns_v3.json"
METRICS_JSON = MODELS_DIR / "metrics_v3.json"
EAFC_CSV     = ROOT / "data/processed/eafc26_team_features.csv"


# ── Dixon-Coles ───────────────────────────────────────────────────────────────

def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1.0 - lh * la * rho
    if x == 0 and y == 1: return 1.0 + lh * rho
    if x == 1 and y == 0: return 1.0 + la * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0


def predict_match_dc(lh, la, rho=-0.20, mg=8):
    lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
    hp = poisson.pmf(range(mg + 1), lh)
    ap = poisson.pmf(range(mg + 1), la)
    sm = np.outer(hp, ap)
    for i in range(min(2, mg + 1)):
        for j in range(min(2, mg + 1)):
            sm[i, j] *= _tau(i, j, lh, la, rho)
    sm /= sm.sum()

    p_home = float(sm[np.tril_indices(mg + 1, -1)].sum())
    p_draw = float(np.trace(sm))
    p_away = float(sm[np.triu_indices(mg + 1,  1)].sum())
    p_over = float(sum(sm[i, j] for i in range(mg+1) for j in range(mg+1) if i+j > 2))
    p_btts = float((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la)))
    most   = np.unravel_index(sm.argmax(), sm.shape)

    # Top 5 placares mais prováveis
    flat = [(sm[i, j], i, j) for i in range(mg+1) for j in range(mg+1)]
    top5 = sorted(flat, reverse=True)[:5]

    return dict(
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        p_over_25=p_over, p_btts=p_btts,
        most_likely_score=f"{most[0]}-{most[1]}",
        most_likely_prob=float(sm[most]),
        top5_scores=top5,
        score_matrix=sm,
    )


def decide(ph, pd_, pa, thr):
    if thr > 0 and pd_ >= thr: return "Empate"
    return "Casa (Austria)" if ph >= pa else "Fora (Jordan)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print(f"  Predição: {HOME_TEAM} vs {AWAY_TEAM}  [{MATCH_DATE.date()}]")
    print(f"  Torneio : {TOURNAMENT}  |  Neutro: {'Sim' if IS_NEUTRAL else 'Não'}")
    print(SEP)

    # ── [1] Features EAFC ────────────────────────────────────────────────
    print("\n[1] Features EAFC26 (convocados Copa 2026)")
    eafc_raw = pd.read_csv(EAFC_CSV)
    eafc_raw["team_en"] = eafc_raw["team"].map(SUPABASE_PT_TO_EN).fillna(eafc_raw["team"])

    print(f"\n  {'Campo':<22} {'Austria':>10} {'Jordan':>10}")
    print("  " + "-" * 44)
    for col in ["n_squad", "n_matched", "eafc_sample_weight",
                "eafc_squad", "eafc_atk", "eafc_best_atk", "eafc_top3_atk",
                "eafc_mid", "eafc_best_mid", "eafc_top3_mid",
                "eafc_def", "eafc_best_def", "eafc_top5_def",
                "eafc_gk", "eafc_gk_avg", "eafc_best_overall",
                "eafc_top11", "eafc_depth", "eafc_std"]:
        r_aut = eafc_raw[eafc_raw["team_en"] == HOME_TEAM]
        r_jor = eafc_raw[eafc_raw["team_en"] == AWAY_TEAM]
        v_aut = r_aut[col].values[0] if len(r_aut) else "N/A"
        v_jor = r_jor[col].values[0] if len(r_jor) else "N/A"
        fmt = ".1f" if isinstance(v_aut, float) else ""
        print(f"  {col:<22} {v_aut:>10{fmt}} {v_jor:>10{fmt}}")

    # Alerta de confiança
    r_jor = eafc_raw[eafc_raw["team_en"] == AWAY_TEAM].iloc[0]
    r_aut = eafc_raw[eafc_raw["team_en"] == HOME_TEAM].iloc[0]
    print(f"\n  [!] Jordan: {int(r_jor['n_matched'])}/{int(r_jor['n_squad'])} jogadores matchados "
          f"(eafc_sample_weight={r_jor['eafc_sample_weight']:.2f}) "
          f"→ BAIXA CONFIANÇA nas features EAFC de Jordan")
    print(f"  [✓] Austria: {int(r_aut['n_matched'])}/{int(r_aut['n_squad'])} jogadores matchados "
          f"(eafc_sample_weight={r_aut['eafc_sample_weight']:.2f}) "
          f"→ ALTA CONFIANÇA nas features EAFC de Austria")

    # ── [2] Carrega dados históricos ──────────────────────────────────────
    print("\n[2] Carregando dados históricos...")
    elo_df, current_elo             = load_elo_ratings()
    rankings_df, current_rankings   = load_fifa_rankings()
    df_results                      = load_results(elo_df, current_elo)
    eafc_lookup, eafc_def, former   = load_eafc_features()

    # ── [3] Features históricas ───────────────────────────────────────────
    print(f"\n[3] Computando features históricas (antes de {MATCH_DATE.date()})...")
    hf = compute_team_features(df_results, HOME_TEAM, MATCH_DATE, rankings_df, current_rankings)
    af = compute_team_features(df_results, AWAY_TEAM, MATCH_DATE, rankings_df, current_rankings)
    h2h = compute_h2h(df_results, HOME_TEAM, AWAY_TEAM, MATCH_DATE)

    print(f"\n  {'Feature':<28} {'Austria':>10} {'Jordan':>10}")
    print("  " + "-" * 50)
    hist_display = [
        ("n_matches",         hf["n_matches"],          af["n_matches"]),
        ("fifa_points",       hf["fifa_points"],         af["fifa_points"]),
        ("goals_for_avg",     hf["goals_for_avg"],       af["goals_for_avg"]),
        ("goals_against_avg", hf["goals_against_avg"],   af["goals_against_avg"]),
        ("win_rate",          hf["win_rate"],             af["win_rate"]),
        ("draw_rate",         hf["draw_rate"],            af["draw_rate"]),
        ("form_goals_for",    hf["form_goals_for"],       af["form_goals_for"]),
        ("form_goals_against",hf["form_goals_against"],  af["form_goals_against"]),
        ("form_win_rate",     hf["form_win_rate"],        af["form_win_rate"]),
        ("form5_pts",         hf["form5_pts"],            af["form5_pts"]),
        ("win_rate_home",     hf["win_rate_home"],        None),
        ("win_rate_away",     None,                       af["win_rate_away"]),
        ("win_rate_neutral",  hf["win_rate_neutral"],     af["win_rate_neutral"]),
        ("avg_opp_points",    hf["avg_opp_points"],       af["avg_opp_points"]),
        ("sos_goals_for",     hf["sos_goals_for"],        af["sos_goals_for"]),
    ]
    for name, v_h, v_a in hist_display:
        sh = f"{v_h:.3f}" if v_h is not None else "  —"
        sa = f"{v_a:.3f}" if v_a is not None else "  —"
        print(f"  {name:<28} {sh:>10} {sa:>10}")

    # ELO ao vivo
    home_elo = get_elo_at_date(elo_df, current_elo, HOME_TEAM, MATCH_DATE)
    away_elo = get_elo_at_date(elo_df, current_elo, AWAY_TEAM, MATCH_DATE)
    elo_diff = home_elo - away_elo
    print(f"\n  ELO Austria  : {home_elo:.1f}")
    print(f"  ELO Jordan   : {away_elo:.1f}")
    print(f"  ELO diff     : {elo_diff:+.1f}")

    # H2H
    print(f"\n  H2H (últimas 10 partidas Austria vs Jordan):")
    print(f"    home_wins={h2h['h2h_home_wins']:.3f}  draws={h2h['h2h_draws']:.3f}  "
          f"away_wins={h2h['h2h_away_wins']:.3f}  "
          f"goals_avg={h2h['h2h_goals_avg']:.2f}  n={h2h['h2h_n']}")

    # ── [4] Monta vetor de features ───────────────────────────────────────
    print(f"\n[4] Montando vetor de features...")
    with open(FEAT_JSON, encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(METRICS_JSON, encoding="utf-8") as f:
        metrics = json.load(f)
    rho = metrics["rho_dixon_coles"]

    h_eafc, h_found = get_eafc(HOME_TEAM, eafc_lookup, eafc_def, former)
    a_eafc, a_found = get_eafc(AWAY_TEAM, eafc_lookup, eafc_def, former)

    t_weight  = get_tournament_weight(TOURNAMENT)
    r_weight  = recency_weight_v3(MATCH_DATE, REFERENCE_DATE)
    avg_elo   = (home_elo + away_elo) / 2
    elo_w     = avg_elo / GLOBAL_AVG_ELO

    row = {
        "is_neutral":              IS_NEUTRAL,
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
        "home_eafc_atk":           h_eafc["eafc_atk"],
        "home_eafc_best_atk":      h_eafc["eafc_best_atk"],
        "home_eafc_top3_atk":      h_eafc["eafc_top3_atk"],
        "home_eafc_mid":           h_eafc["eafc_mid"],
        "home_eafc_best_mid":      h_eafc["eafc_best_mid"],
        "home_eafc_top3_mid":      h_eafc["eafc_top3_mid"],
        "home_eafc_def":           h_eafc["eafc_def"],
        "home_eafc_best_def":      h_eafc["eafc_best_def"],
        "home_eafc_top5_def":      h_eafc["eafc_top5_def"],
        "home_eafc_gk":            h_eafc["eafc_gk"],
        "home_eafc_gk_avg":        h_eafc["eafc_gk_avg"],
        "home_eafc_squad":         h_eafc["eafc_squad"],
        "home_eafc_best_overall":  h_eafc["eafc_best_overall"],
        "home_eafc_top11":         h_eafc["eafc_top11"],
        "home_eafc_depth":         h_eafc["eafc_depth"],
        "home_eafc_std":           h_eafc["eafc_std"],
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
        "away_eafc_atk":           a_eafc["eafc_atk"],
        "away_eafc_best_atk":      a_eafc["eafc_best_atk"],
        "away_eafc_top3_atk":      a_eafc["eafc_top3_atk"],
        "away_eafc_mid":           a_eafc["eafc_mid"],
        "away_eafc_best_mid":      a_eafc["eafc_best_mid"],
        "away_eafc_top3_mid":      a_eafc["eafc_top3_mid"],
        "away_eafc_def":           a_eafc["eafc_def"],
        "away_eafc_best_def":      a_eafc["eafc_best_def"],
        "away_eafc_top5_def":      a_eafc["eafc_top5_def"],
        "away_eafc_gk":            a_eafc["eafc_gk"],
        "away_eafc_gk_avg":        a_eafc["eafc_gk_avg"],
        "away_eafc_squad":         a_eafc["eafc_squad"],
        "away_eafc_best_overall":  a_eafc["eafc_best_overall"],
        "away_eafc_top11":         a_eafc["eafc_top11"],
        "away_eafc_depth":         a_eafc["eafc_depth"],
        "away_eafc_std":           a_eafc["eafc_std"],
        "diff_fifa_points":        hf["fifa_points"]       - af["fifa_points"],
        "diff_goals_for":          hf["goals_for_avg"]     - af["goals_for_avg"],
        "diff_goals_against":      hf["goals_against_avg"] - af["goals_against_avg"],
        "diff_win_rate":           hf["win_rate"]          - af["win_rate"],
        "diff_form_win_rate":      hf["form_win_rate"]     - af["form_win_rate"],
        "diff_form5":              hf["form5_pts"]         - af["form5_pts"],
        "diff_sos_goals":          hf["sos_goals_for"]     - af["sos_goals_for"],
        "diff_sos_form":           hf["sos_form_goals"]    - af["sos_form_goals"],
        "diff_avg_opp":            hf["avg_opp_points"]    - af["avg_opp_points"],
        "eafc_atk_diff":           h_eafc["eafc_atk"]          - a_eafc["eafc_atk"],
        "eafc_def_diff":           h_eafc["eafc_def"]          - a_eafc["eafc_def"],
        "eafc_mid_diff":           h_eafc["eafc_mid"]          - a_eafc["eafc_mid"],
        "eafc_gk_diff":            h_eafc["eafc_gk"]           - a_eafc["eafc_gk"],
        "eafc_squad_diff":         h_eafc["eafc_squad"]        - a_eafc["eafc_squad"],
        "eafc_best_overall_diff":  h_eafc["eafc_best_overall"] - a_eafc["eafc_best_overall"],
        "eafc_top3_atk_diff":      h_eafc["eafc_top3_atk"]    - a_eafc["eafc_top3_atk"],
        "eafc_top5_def_diff":      h_eafc["eafc_top5_def"]    - a_eafc["eafc_top5_def"],
        "h2h_home_wins":           h2h["h2h_home_wins"],
        "h2h_draws":               h2h["h2h_draws"],
        "h2h_away_wins":           h2h["h2h_away_wins"],
        "h2h_goals_avg":           h2h["h2h_goals_avg"],
        "h2h_n":                   h2h["h2h_n"],
    }

    # Verifica que todas as features estão presentes
    missing = [c for c in feat_cols if c not in row]
    if missing:
        print(f"  [ERRO] Features faltando: {missing}")
        return

    X = np.array([[row[c] for c in feat_cols]])
    print(f"  Vetor de features montado: {X.shape[1]} features  ✓")
    print(f"  EAFC Austria encontrado: {'✓' if h_found else '✗ (fallback)'}  "
          f"Jordan: {'✓' if a_found else '✗ (fallback)'}")

    # ── [5] Predição ──────────────────────────────────────────────────────
    print(f"\n[5] Predição com modelos XGBoost Poisson v3...")
    model_h = joblib.load(MODEL_HOME)
    model_a = joblib.load(MODEL_AWAY)

    lh = float(model_h.predict(X)[0])
    la = float(model_a.predict(X)[0])

    print(f"\n  lambda_home (Austria) : {lh:.4f} gols esperados")
    print(f"  lambda_away (Jordan)  : {la:.4f} gols esperados")
    print(f"  rho Dixon-Coles       : {rho}")

    # Poisson puro (sem DC) para comparar
    from scipy.stats import poisson as spois
    def poisson_pure(lhv, lav, mg=8):
        hp = spois.pmf(range(mg+1), lhv)
        ap = spois.pmf(range(mg+1), lav)
        sm = np.outer(hp, ap)
        return (sm[np.tril_indices(mg+1,-1)].sum(),
                np.trace(sm),
                sm[np.triu_indices(mg+1,1)].sum())

    ph0, pd0, pa0 = poisson_pure(lh, la)
    result_dc = predict_match_dc(lh, la, rho=rho)

    print(f"\n  {'':30} {'Poisson puro':>14} {'+ Dixon-Coles':>14}")
    print("  " + "-" * 60)
    print(f"  {'P(Austria vence)':30} {ph0*100:>13.2f}%  {result_dc['p_home']*100:>13.2f}%")
    print(f"  {'P(Empate)':30} {pd0*100:>13.2f}%  {result_dc['p_draw']*100:>13.2f}%")
    print(f"  {'P(Jordan vence)':30} {pa0*100:>13.2f}%  {result_dc['p_away']*100:>13.2f}%")
    print(f"  {'P(Over 2.5)':30} {'—':>14}  {result_dc['p_over_25']*100:>13.2f}%")
    print(f"  {'P(BTTS)':30} {'—':>14}  {result_dc['p_btts']*100:>13.2f}%")

    print(f"\n  Top 5 placares mais prováveis (com DC):")
    for prob, i, j in result_dc["top5_scores"]:
        print(f"    {i}-{j}  →  {prob*100:.2f}%")

    # ── [6] Decisão por threshold ──────────────────────────────────────────
    print(f"\n[6] Decisão por threshold (com Dixon-Coles):")
    ph = result_dc["p_home"]
    pd_ = result_dc["p_draw"]
    pa = result_dc["p_away"]
    for thr in [0.33, 0.38]:
        dec = decide(ph, pd_, pa, thr)
        triggered = "p_draw >= thr ✓" if thr > 0 and pd_ >= thr else f"argmax(p_home={ph:.3f}, p_away={pa:.3f})"
        print(f"  thr={thr}  →  {dec:<20}  [{triggered}]")

    # ── [7] Análise de confiança ───────────────────────────────────────────
    print(f"\n[7] Análise de confiança da predição:")
    print(f"\n  Jordan (away):")
    print(f"    Convocados no squad      : {int(r_jor['n_squad'])}")
    print(f"    Matchados no EA FC 26    : {int(r_jor['n_matched'])}  ({100*r_jor['n_matched']/r_jor['n_squad']:.1f}%)")
    print(f"    eafc_sample_weight       : {r_jor['eafc_sample_weight']:.2f}  (escala 0-1, ideal ≥ 0.7)")
    print(f"    → As features eafc_* de Jordan são ESTIMADAS (média global),")
    print(f"      não refletem a força real do elenco convocado.")
    print(f"    → Nos dados de treino, Jordan usou fallback em TODAS as partidas.")

    print(f"\n  Austria (home):")
    print(f"    Convocados no squad      : {int(r_aut['n_squad'])}")
    print(f"    Matchados no EA FC 26    : {int(r_aut['n_matched'])}  ({100*r_aut['n_matched']/r_aut['n_squad']:.1f}%)")
    print(f"    eafc_sample_weight       : {r_aut['eafc_sample_weight']:.2f}  (escala 0-1, ideal ≥ 0.7)")
    print(f"    → Features EAFC de Austria são confiáveis.")

    print(f"\n  ELO diff Austria - Jordan: {elo_diff:+.1f} pontos")
    print(f"  Austria FIFA pts         : {hf['fifa_points']:.1f}")
    print(f"  Jordan FIFA pts          : {af['fifa_points']:.1f}")
    print(f"  diff_fifa_points         : {row['diff_fifa_points']:+.1f}")
    print(f"  Confiança geral          : MÉDIA (Jordan EAFC não confiável)")

    print(f"\n{SEP}")
    print(f"  RESUMO FINAL — Austria vs Jordan")
    print(SEP)
    print(f"  λ_Austria = {lh:.3f}  |  λ_Jordan = {la:.3f}")
    print(f"  P(Austria) = {ph*100:.1f}%  P(Empate) = {pd_*100:.1f}%  P(Jordan) = {pa*100:.1f}%")
    print(f"  Placar mais provável : {result_dc['most_likely_score']} ({result_dc['most_likely_prob']*100:.2f}%)")
    print(f"  Over 2.5             : {result_dc['p_over_25']*100:.1f}%  |  BTTS: {result_dc['p_btts']*100:.1f}%")
    print(f"  Previsão (thr=0.33)  : {decide(ph, pd_, pa, 0.33)}")
    print(f"  Previsão (thr=0.38)  : {decide(ph, pd_, pa, 0.38)}")
    print(SEP)


if __name__ == "__main__":
    main()
