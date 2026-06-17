"""predict_v3.py — Predição v3 via Poisson Bivariada + Dixon-Coles.

Correção Dixon-Coles: ajusta a massa de probabilidade nos placares baixos
(0-0, 1-0, 0-1, 1-1) via parâmetro rho estimado por MLE nos dados reais.
rho < 0 → mais 0-0 e 1-1 (empates), menos 1-0 e 0-1 (vitórias curtas).

Uso standalone:
    python -m src.models.predict_v3
    python src/models/predict_v3.py
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Caminhos ──────────────────────────────────────────────────────────────────
_MODELS_DIR   = Path("models")
_MODEL_HOME   = _MODELS_DIR / "model_goals_home_v3.pkl"
_MODEL_AWAY   = _MODELS_DIR / "model_goals_away_v3.pkl"
_FEAT_JSON    = _MODELS_DIR / "feature_columns_v3.json"
_METRICS_JSON = _MODELS_DIR / "metrics_v3.json"
_DATA_CSV     = Path("data/processed/training_data_v3.csv")
_EAFC_CSV     = Path("data/processed/eafc26_team_features.csv")

SPLIT_DATE = "2023-12-31"
SEP = "=" * 68

SUPABASE_PT_TO_EN = {
    "Alemanha": "Germany", "Argentina": "Argentina", "Argélia": "Algeria",
    "Arábia Saudita": "Saudi Arabia", "Austrália": "Australia", "Brasil": "Brazil",
    "Bélgica": "Belgium", "Bósnia": "Bosnia-Herzegovina", "Cabo Verde": "Cape Verde",
    "Canadá": "Canada", "Catar": "Qatar", "Colômbia": "Colombia",
    "Coreia do Sul": "South Korea", "Costa do Marfim": "Ivory Coast",
    "Croácia": "Croatia", "Curaçao": "Curacao", "Egito": "Egypt",
    "Equador": "Ecuador", "Escócia": "Scotland", "Espanha": "Spain",
    "Estados Unidos": "USA", "França": "France", "Gana": "Ghana",
    "Haiti": "Haiti", "Holanda": "Netherlands", "Inglaterra": "England",
    "Iraque": "Iraq", "Irã": "Iran", "Japão": "Japan", "Jordânia": "Jordan",
    "Marrocos": "Morocco", "México": "Mexico", "Noruega": "Norway",
    "Nova Zelândia": "New Zealand", "Panamá": "Panama", "Paraguai": "Paraguay",
    "Portugal": "Portugal", "Rep. D. Congo": "DR Congo", "Senegal": "Senegal",
    "Suécia": "Sweden", "Suíça": "Switzerland", "Tchéquia": "Czech Republic",
    "Tunísia": "Tunisia", "Turquia": "Turkey", "Uruguai": "Uruguay",
    "Uzbequistão": "Uzbekistan", "África do Sul": "South Africa", "Áustria": "Austria",
}


# ── Dixon-Coles: função tau ───────────────────────────────────────────────────

def tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Fator de correção Dixon-Coles para placares baixos."""
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Núcleo: Poisson Bivariada + Dixon-Coles ───────────────────────────────────

def predict_match(lambda_home: float, lambda_away: float,
                  max_goals: int = 8, rho: float = 0.0) -> dict:
    """Matriz de placares Poisson com correção Dixon-Coles opcional (rho≤0).

    rho=0.0 → Poisson puro (sem correção).
    rho<0.0 → mais massa em 0-0 e 1-1, menos em 1-0 e 0-1.
    """
    lh = max(float(lambda_home), 1e-6)
    la = max(float(lambda_away), 1e-6)

    hp = poisson.pmf(range(max_goals + 1), lh)
    ap = poisson.pmf(range(max_goals + 1), la)
    score_matrix = np.outer(hp, ap)

    # Aplica correção Dixon-Coles nos placares baixos
    if rho != 0.0:
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                score_matrix[i, j] *= tau(i, j, lh, la, rho)
        score_matrix /= score_matrix.sum()  # renormaliza

    p_home = float(score_matrix[np.tril_indices(max_goals + 1, -1)].sum())
    p_draw = float(np.trace(score_matrix))
    p_away = float(score_matrix[np.triu_indices(max_goals + 1,  1)].sum())

    p_over_25 = float(sum(
        score_matrix[i, j]
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
        if i + j > 2
    ))

    p_btts = float(
        (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
    )

    most_likely = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

    return {
        "p_home":            round(p_home, 4),
        "p_draw":            round(p_draw, 4),
        "p_away":            round(p_away, 4),
        "goals_home":        round(lh, 2),
        "goals_away":        round(la, 2),
        "p_over_25":         round(p_over_25, 4),
        "p_btts":            round(p_btts, 4),
        "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
        "most_likely_prob":  round(float(score_matrix[most_likely]), 4),
    }


# ── Critério de decisão com threshold ────────────────────────────────────────

def predict_result(p_home: float, p_draw: float, p_away: float,
                   threshold_draw: float = 0.0) -> str:
    """Hierárquico: empate se p_draw >= threshold (>0), senão argmax H/A."""
    if threshold_draw > 0.0 and p_draw >= threshold_draw:
        return "D"
    return "H" if p_home >= p_away else "A"


# ── Estimação de rho por MLE ──────────────────────────────────────────────────

def _log_likelihood_rho(lh_arr, la_arr, y_h, y_a, rho, mg=8):
    """Log-likelihood total dado rho para um conjunto de partidas."""
    ll = 0.0
    for lh, la, h, a in zip(lh_arr, la_arr, y_h, y_a):
        lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
        hp = poisson.pmf(range(mg + 1), lh)
        ap = poisson.pmf(range(mg + 1), la)
        sm = np.outer(hp, ap)

        # Aplica tau e renormaliza
        for i in range(min(2, mg + 1)):
            for j in range(min(2, mg + 1)):
                sm[i, j] *= tau(i, j, lh, la, rho)
        sm /= sm.sum()

        # Log-prob do placar observado (clipa para max_goals)
        h_c = min(int(h), mg)
        a_c = min(int(a), mg)
        prob = max(sm[h_c, a_c], 1e-12)
        ll += np.log(prob)
    return ll


def estimate_rho(lh_arr, la_arr, y_h, y_a,
                 rho_min=-0.5, rho_max=0.0, rho_step=0.01) -> tuple:
    """Busca o rho que maximiza o log-likelihood nos dados fornecidos.

    Returns (best_rho, tabela_df)
    """
    rho_values = np.round(np.arange(rho_min, rho_max + rho_step / 2, rho_step), 3)
    records = []
    for rho in rho_values:
        ll = _log_likelihood_rho(lh_arr, la_arr, y_h, y_a, rho)
        records.append({"rho": rho, "log_likelihood": round(ll, 4)})

    table = pd.DataFrame(records)
    best_idx = table["log_likelihood"].idxmax()
    best_rho = float(table.loc[best_idx, "rho"])
    return best_rho, table


# ── Carga de modelos ──────────────────────────────────────────────────────────

def load_models_v3():
    for p in (_MODEL_HOME, _MODEL_AWAY, _FEAT_JSON):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} não encontrado. Rode scripts/train_models_v3.py primeiro."
            )
    model_h = joblib.load(_MODEL_HOME)
    model_a = joblib.load(_MODEL_AWAY)
    with open(_FEAT_JSON, encoding="utf-8") as f:
        feat_cols = json.load(f)
    return model_h, model_a, feat_cols


def predict_from_row(model_h, model_a, feat_cols: list, row: pd.Series,
                     threshold_draw: float = 0.0, rho: float = 0.0) -> dict:
    X  = row[feat_cols].values.reshape(1, -1)
    lh = float(model_h.predict(X)[0])
    la = float(model_a.predict(X)[0])
    return predict_match(lh, la, rho=rho)


# ── Construção do DataFrame de predições ──────────────────────────────────────

def build_preds_df(model_h, model_a, feat_cols, val_df: pd.DataFrame,
                   rho: float = 0.0) -> pd.DataFrame:
    X_val  = val_df[feat_cols].values
    lh_all = model_h.predict(X_val)
    la_all = model_a.predict(X_val)

    rows = [predict_match(lh_all[i], la_all[i], rho=rho) for i in range(len(val_df))]
    preds = pd.DataFrame(rows)

    val = val_df.reset_index(drop=True)
    preds["lh"]              = lh_all
    preds["la"]              = la_all
    preds["true_result"]     = np.where(
        val["goals_home"] > val["goals_away"], "H",
        np.where(val["goals_home"] < val["goals_away"], "A", "D")
    )
    preds["goals_home_true"] = val["goals_home"].values
    preds["goals_away_true"] = val["goals_away"].values
    preds["btts_true"]       = (
        (val["goals_home"] > 0) & (val["goals_away"] > 0)
    ).astype(int).values
    preds["tournament"]      = val["tournament"].values
    return preds


# ── Calibração de threshold ───────────────────────────────────────────────────

def calibrate_threshold(preds: pd.DataFrame) -> tuple:
    """Varre 0.25-0.40 e retorna (best_thr, tabela). Critério: max acurácia."""
    thresholds = np.round(np.arange(0.25, 0.405, 0.01), 2)
    n_true_draw = (preds["true_result"] == "D").sum()
    n = len(preds)
    records = []
    for thr in thresholds:
        pred_res = np.array([
            predict_result(r.p_home, r.p_draw, r.p_away, thr)
            for r in preds.itertuples()
        ])
        correct = (pred_res == preds["true_result"].values).sum()
        acc = correct / n
        n_pred_draw = (pred_res == "D").sum()
        tp_draw = ((pred_res == "D") & (preds["true_result"].values == "D")).sum()
        prec = tp_draw / n_pred_draw if n_pred_draw > 0 else 0.0
        rec  = tp_draw / n_true_draw if n_true_draw > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        records.append({
            "threshold": thr, "acc": round(acc, 4),
            "n_draw_pred": int(n_pred_draw),
            "draw_prec": round(prec, 4),
            "draw_rec": round(rec, 4),
            "draw_f1": round(f1, 4),
        })
    table = pd.DataFrame(records)
    # Maximiza F1 de empate entre os thresholds até 3pp abaixo da acurácia
    # máxima — evita o colapso de recall que a acurácia pura produz quando
    # a precisão da classe D é baixa (acc pura tende a zerar empates).
    candidates = table[table["acc"] >= table["acc"].max() - 0.03]
    best_idx = candidates.sort_values("draw_f1", ascending=False).index[0]
    return float(table.loc[best_idx, "threshold"]), table


# ── Avaliação completa ────────────────────────────────────────────────────────

def evaluate_with_threshold(preds: pd.DataFrame, threshold_draw: float,
                            label: str = "") -> dict:
    n = len(preds)
    pred_res = np.array([
        predict_result(r.p_home, r.p_draw, r.p_away, threshold_draw)
        for r in preds.itertuples()
    ])
    correct  = pred_res == preds["true_result"].values
    acc      = correct.mean()
    total_true = preds["goals_home_true"] + preds["goals_away_true"]
    ou_acc   = ((total_true > 2.5) == (preds["p_over_25"] > 0.5)).mean()
    btts_acc = (preds["btts_true"] == (preds["p_btts"] > 0.5)).mean()
    mae_h    = mean_absolute_error(preds["goals_home_true"], preds["lh"])
    mae_a    = mean_absolute_error(preds["goals_away_true"], preds["la"])
    mae_t    = mean_absolute_error(total_true, preds["lh"] + preds["la"])

    if label:
        hdr = f"  {'Resultado':<12} {'Real':>5} {'R%':>5}  {'Prev':>5} {'P%':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
        print(f"\n  {label} — threshold={threshold_draw}  n={n:,}")
        print(f"  Acurácia: {acc*100:.2f}%  MAE_t={mae_t:.4f}  O/U={ou_acc*100:.2f}%  BTTS={btts_acc*100:.2f}%")
        print(f"  p_draw: min={preds['p_draw'].min():.3f}  mean={preds['p_draw'].mean():.3f}  max={preds['p_draw'].max():.3f}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for res, lbl in [("H", "Home win"), ("D", "Empate"), ("A", "Away win")]:
            nt = (preds["true_result"] == res).sum()
            np_ = (pred_res == res).sum()
            tp  = ((preds["true_result"].values == res) & (pred_res == res)).sum()
            prec = tp / np_ if np_ > 0 else 0.0
            rec  = tp / nt  if nt  > 0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
            print(f"  {lbl:<12} {nt:>5} {100*nt/n:>4.1f}%  "
                  f"{np_:>5} {100*np_/n:>4.1f}%  "
                  f"{prec*100:>5.1f}%  {rec*100:>5.1f}%  {f1:.3f}")

    n_draw_pred = (pred_res == "D").sum()
    n_draw_true = (preds["true_result"] == "D").sum()
    tp_draw = ((preds["true_result"].values == "D") & (pred_res == "D")).sum()
    draw_rec = tp_draw / n_draw_true if n_draw_true > 0 else 0.0

    return dict(acc=acc, mae_h=mae_h, mae_a=mae_a, mae_t=mae_t,
                ou_acc=ou_acc, btts_acc=btts_acc,
                n_draw_pred=int(n_draw_pred), draw_rec=draw_rec)


# ── Predição completa de uma partida (mercados full) ───────────────────────────
# Lógica compartilhada entre scripts/predict_match_cli.py e
# api/routes/full_match.py — única fonte de verdade para os mercados.

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from build_features_v3 import (  # noqa: E402
    SUPABASE_PT_TO_EN as _SUPABASE_PT_TO_EN,
    normalize_name as _normalize_name,
    load_elo_ratings as _load_elo_ratings,
    load_fifa_rankings as _load_fifa_rankings,
    load_results as _load_results,
    load_eafc_features as _load_eafc_features,
    compute_team_features as _compute_team_features,
    compute_h2h as _compute_h2h,
    get_eafc as _get_eafc,
)

_FULL_MATCH_CTX: dict = {}


def resolve_team_name(raw: str) -> str:
    """PT (Supabase) ou EN -> nome canônico usado no pipeline histórico."""
    name = _SUPABASE_PT_TO_EN.get(raw, raw)
    return _normalize_name(name)


def load_full_match_context(force: bool = False) -> None:
    """Carrega (uma única vez) tudo que predict_full_match() precisa:
    histórico Elo/FIFA/resultados/EAFC26 + modelos v3 + rho/threshold."""
    if _FULL_MATCH_CTX and not force:
        return
    elo_df, current_elo           = _load_elo_ratings()
    rankings_df, current_rankings = _load_fifa_rankings()
    df_results                    = _load_results(elo_df, current_elo)
    eafc_lookup, eafc_def, former = _load_eafc_features()
    eafc_raw = pd.read_csv(_EAFC_CSV)
    eafc_raw["team_en"] = eafc_raw["team"].map(_SUPABASE_PT_TO_EN).fillna(eafc_raw["team"])

    model_h, model_a, feat_cols = load_models_v3()
    with open(_METRICS_JSON, encoding="utf-8") as f:
        metrics = json.load(f)

    _FULL_MATCH_CTX.update(dict(
        elo_df=elo_df, current_elo=current_elo,
        rankings_df=rankings_df, current_rankings=current_rankings,
        df_results=df_results,
        eafc_lookup=eafc_lookup, eafc_def=eafc_def, former=former, eafc_raw=eafc_raw,
        model_h=model_h, model_a=model_a, feat_cols=feat_cols,
        rho=metrics["rho_dixon_coles"], threshold_draw=metrics["threshold_draw"],
    ))


def build_score_matrix(lh: float, la: float, rho: float, mg: int = 8) -> np.ndarray:
    lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
    hp = poisson.pmf(range(mg + 1), lh)
    ap = poisson.pmf(range(mg + 1), la)
    sm = np.outer(hp, ap)
    if rho != 0.0:
        for i in range(min(2, mg + 1)):
            for j in range(min(2, mg + 1)):
                sm[i, j] *= tau(i, j, lh, la, rho)
        sm /= sm.sum()
    return sm


def _result_probs(sm: np.ndarray) -> tuple:
    mg = sm.shape[0] - 1
    p_home = float(sm[np.tril_indices(mg + 1, -1)].sum())
    p_draw = float(np.trace(sm))
    p_away = float(sm[np.triu_indices(mg + 1, 1)].sum())
    return p_home, p_draw, p_away


def _over_under_table(sm: np.ndarray, lines=(0.5, 1.5, 2.5, 3.5, 4.5)) -> list:
    mg = sm.shape[0] - 1
    totals = np.add.outer(np.arange(mg + 1), np.arange(mg + 1))
    return [
        {"line": line,
         "p_over": float(sm[totals > line].sum()),
         "p_under": float(sm[totals < line].sum())}
        for line in lines
    ]


def _btts_probs(sm: np.ndarray) -> tuple:
    p_yes = float(sm[1:, 1:].sum())
    return p_yes, 1.0 - p_yes


def _top_n_scores(sm: np.ndarray, n: int = 5) -> list:
    mg = sm.shape[0] - 1
    flat = [(float(sm[i, j]), i, j) for i in range(mg + 1) for j in range(mg + 1)]
    top = sorted(flat, reverse=True)[:n]
    return [{"score": f"{i}-{j}", "probability": prob} for prob, i, j in top]


def _asian_handicap_table(sm: np.ndarray, lines=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)) -> list:
    """Handicap a favor do time da casa. adjusted = (gols_casa - gols_fora) + line."""
    mg = sm.shape[0] - 1
    margin = np.subtract.outer(np.arange(mg + 1), np.arange(mg + 1))
    out = []
    for line in lines:
        adjusted = margin + line
        p_home_covers = float(sm[adjusted > 0].sum())
        p_push        = float(sm[adjusted == 0].sum())
        p_away_covers = float(sm[adjusted < 0].sum())
        out.append({
            "line": line,
            "home_covers": p_home_covers,
            "push": p_push if p_push > 0 else None,
            "away_covers": p_away_covers,
        })
    return out


def _build_feature_row(hf: dict, af: dict, h_eafc: dict, a_eafc: dict,
                       h2h: dict, is_neutral: int) -> dict:
    return {
        "is_neutral":              is_neutral,
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


def predict_full_match(home: str, away: str, is_neutral: bool = True,
                       match_date: "pd.Timestamp | None" = None) -> dict:
    """Predição completa (todos os mercados) para uma partida.

    home/away aceitam nomes em português (Supabase) ou inglês — são
    resolvidos via resolve_team_name() antes de tudo. Única fonte de
    verdade para os mercados: usada por scripts/predict_match_cli.py
    e api/routes/full_match.py.
    """
    load_full_match_context()
    ctx = _FULL_MATCH_CTX
    home = resolve_team_name(home)
    away = resolve_team_name(away)
    snapshot_date = match_date or pd.Timestamp.today().normalize()

    hf = _compute_team_features(ctx["df_results"], home, snapshot_date,
                                ctx["rankings_df"], ctx["current_rankings"])
    af = _compute_team_features(ctx["df_results"], away, snapshot_date,
                                ctx["rankings_df"], ctx["current_rankings"])
    h2h = _compute_h2h(ctx["df_results"], home, away, snapshot_date)
    h_eafc, h_found = _get_eafc(home, ctx["eafc_lookup"], ctx["eafc_def"], ctx["former"])
    a_eafc, a_found = _get_eafc(away, ctx["eafc_lookup"], ctx["eafc_def"], ctx["former"])

    row = _build_feature_row(hf, af, h_eafc, a_eafc, h2h, int(is_neutral))
    feat_cols = ctx["feat_cols"]
    missing = [c for c in feat_cols if c not in row]
    if missing:
        raise ValueError(f"Features faltando no vetor: {missing}")

    X = np.array([[row[c] for c in feat_cols]])
    lh = float(ctx["model_h"].predict(X)[0])
    la = float(ctx["model_a"].predict(X)[0])

    rho = ctx["rho"]
    threshold_draw = ctx["threshold_draw"]
    sm = build_score_matrix(lh, la, rho)
    p_home, p_draw, p_away = _result_probs(sm)
    predicted = predict_result(p_home, p_draw, p_away, threshold_draw)
    predicted_label = {"H": "home_win", "D": "draw", "A": "away_win"}[predicted]
    p_btts_yes, p_btts_no = _btts_probs(sm)

    def eafc_info(team, eafc_row, found):
        r = ctx["eafc_raw"][ctx["eafc_raw"]["team_en"] == team]
        if len(r) == 0:
            return {"sample_weight": None, "matched": None, "found": found}
        r = r.iloc[0]
        return {
            "sample_weight": float(r["eafc_sample_weight"]),
            "matched": f"{int(r['n_matched'])}/{int(r['n_squad'])}",
            "found": found,
        }

    home_conf = eafc_info(home, h_eafc, h_found)
    away_conf = eafc_info(away, a_eafc, a_found)
    low_confidence = any(
        c["sample_weight"] is not None and c["sample_weight"] < 0.7
        for c in (home_conf, away_conf)
    )

    h2h_block = {"has_data": h2h["h2h_n"] > 0}
    if h2h["h2h_n"] > 0:
        h2h_block.update({
            "n": h2h["h2h_n"],
            "home_wins": h2h["h2h_home_wins"],
            "draws": h2h["h2h_draws"],
            "away_wins": h2h["h2h_away_wins"],
            "goals_avg": h2h["h2h_goals_avg"],
        })

    return {
        "home_team": home, "away_team": away,
        "lambda_home": lh, "lambda_away": la,
        "result_1x2": {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "predicted": predicted_label, "threshold_used": threshold_draw,
        },
        "expected_goals": {"home": lh, "away": la, "total": lh + la},
        "over_under": _over_under_table(sm),
        "btts": {"p_yes": p_btts_yes, "p_no": p_btts_no},
        "exact_score_top5": _top_n_scores(sm, 5),
        "asian_handicap": _asian_handicap_table(sm),
        "confidence": {
            "home_sample_weight": home_conf["sample_weight"],
            "away_sample_weight": away_conf["sample_weight"],
            "home_matched": home_conf["matched"],
            "away_matched": away_conf["matched"],
            "low_confidence": low_confidence,
        },
        "h2h": h2h_block,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  predict_v3.py — Dixon-Coles + Calibração Final")
    print(SEP)

    # Carrega
    print("\n[1] Carregando modelos e dados...")
    model_h, model_a, feat_cols = load_models_v3()
    df = pd.read_csv(_DATA_CSV, parse_dates=["date"])
    val = df[df["date"] > SPLIT_DATE].copy().reset_index(drop=True)

    eafc = pd.read_csv(_EAFC_CSV)
    eafc["team_en"] = eafc["team"].map(SUPABASE_PT_TO_EN).fillna(eafc["team"])
    squad_mean = eafc["eafc_squad"].mean()
    TOL = 0.01

    mask_real = (
        (np.abs(val["home_eafc_squad"] - squad_mean) > TOL) &
        (np.abs(val["away_eafc_squad"] - squad_mean) > TOL)
    )
    sub = val[mask_real].copy().reset_index(drop=True)

    print(f"  val total  : {len(val):,}  |  EAFC real: {len(sub):,} ({100*len(sub)/len(val):.1f}%)")

    # Prediz lambdas para o subconjunto EAFC real (usado na estimação de rho)
    X_sub  = sub[feat_cols].values
    lh_sub = model_h.predict(X_sub)
    la_sub = model_a.predict(X_sub)
    y_h_sub = sub["goals_home"].values
    y_a_sub = sub["goals_away"].values

    # ── Estimação de rho ──────────────────────────────────────────────────
    print(f"\n[2] Estimando rho por MLE no subconjunto EAFC real ({len(sub)} partidas)...")
    best_rho, rho_table = estimate_rho(lh_sub, la_sub, y_h_sub, y_a_sub)

    # Baseline (rho=0)
    ll_zero = rho_table.loc[rho_table["rho"] == 0.0, "log_likelihood"].values[0]

    print(f"\n  {'rho':>7}  {'log-likelihood':>16}  {'delta vs rho=0':>16}")
    print(f"  {'─'*44}")
    for _, row in rho_table.iterrows():
        delta = row["log_likelihood"] - ll_zero
        marker = " <-- ÓTIMO" if row["rho"] == best_rho else ""
        print(f"  {row['rho']:>7.3f}  {row['log_likelihood']:>16.2f}  {delta:>+16.2f}{marker}")

    print(f"\n  Rho ótimo : {best_rho}")
    print(f"  ΔlogL (vs rho=0): {rho_table.loc[rho_table['rho']==best_rho,'log_likelihood'].values[0] - ll_zero:+.2f}")

    # ── Constrói DataFrames de predições ──────────────────────────────────
    print(f"\n[3] Construindo predições (sem DC e com DC rho={best_rho})...")
    preds_val_nodc = build_preds_df(model_h, model_a, feat_cols, val, rho=0.0)
    preds_val_dc   = build_preds_df(model_h, model_a, feat_cols, val, rho=best_rho)
    preds_sub_nodc = build_preds_df(model_h, model_a, feat_cols, sub, rho=0.0)
    preds_sub_dc   = build_preds_df(model_h, model_a, feat_cols, sub, rho=best_rho)

    # ── Threshold scan com DC no val completo ─────────────────────────────
    print(f"\n[4] Calibrando threshold com Dixon-Coles (val completo)...")
    best_thr_dc, thr_table_dc = calibrate_threshold(preds_val_dc)

    print(f"\n  {'thr':>5}  {'acc':>6}  {'pred_D':>7}  {'prec_D':>7}  {'rec_D':>7}  {'F1_D':>7}")
    print(f"  {'─'*52}")
    for _, row in thr_table_dc.iterrows():
        marker = " <-- ÓTIMO" if row["threshold"] == best_thr_dc else ""
        print(f"  {row['threshold']:.2f}   {row['acc']*100:5.1f}%  "
              f"{int(row['n_draw_pred']):>7}  "
              f"{row['draw_prec']*100:6.1f}%  "
              f"{row['draw_rec']*100:6.1f}%  "
              f"{row['draw_f1']:.4f}{marker}")

    # Usa threshold anterior (0.33) para comparação consistente,
    # e também o novo threshold ótimo com DC
    THR_OLD = 0.33

    # ── Avaliação detalhada ───────────────────────────────────────────────
    print(f"\n[5] Avaliação detalhada...")
    r_val_nodc   = evaluate_with_threshold(preds_val_nodc, THR_OLD,     "Val geral    | sem DC       ")
    r_val_dc033  = evaluate_with_threshold(preds_val_dc,   THR_OLD,     "Val geral    | DC rho={:.2f} thr=0.33".format(best_rho))
    r_val_dc_opt = evaluate_with_threshold(preds_val_dc,   best_thr_dc, "Val geral    | DC rho={:.2f} thr={:.2f} (ótimo)".format(best_rho, best_thr_dc))
    r_sub_nodc   = evaluate_with_threshold(preds_sub_nodc, THR_OLD,     "EAFC real    | sem DC       ")
    r_sub_dc     = evaluate_with_threshold(preds_sub_dc,   best_thr_dc, "EAFC real    | DC rho={:.2f} thr={:.2f}".format(best_rho, best_thr_dc))

    # ── Tabela comparativa ────────────────────────────────────────────────
    print(f"\n[6] Tabela comparativa")
    col_w = 18
    print(f"\n  {'Métrica':<22} {'Sem DC (0.33)':>{col_w}} {'DC thr=0.33':>{col_w}} {'DC thr={:.2f} (ótimo)'.format(best_thr_dc):>{col_w}}")
    print("  " + "─" * (22 + col_w * 3 + 6))

    def _row(name, key, mult=1, fmt=".2f", unit=""):
        v0  = r_val_nodc[key]  * mult
        v1  = r_val_dc033[key] * mult
        v2  = r_val_dc_opt[key]* mult
        d1  = v1 - v0
        d2  = v2 - v0
        s1  = f"{v1:{fmt}}{unit} ({d1:+.2f})"
        s2  = f"{v2:{fmt}}{unit} ({d2:+.2f})"
        print(f"  {name:<22} {v0:{fmt}}{unit}{'':>{col_w - len(f'{v0:{fmt}}{unit}')}}  {s1:>{col_w}}  {s2:>{col_w}}")

    _row("Acurácia geral",    "acc",          100, ".2f", "%")
    _row("Empates previstos", "n_draw_pred",    1, ".0f", "")
    _row("Recall empate",     "draw_rec",      100, ".1f", "%")
    _row("MAE total_goals",   "mae_t",           1, ".4f", "")
    _row("Over/Under 2.5",    "ou_acc",        100, ".2f", "%")
    _row("BTTS accuracy",     "btts_acc",      100, ".2f", "%")

    print(f"\n  --- Subconjunto EAFC real (n={len(sub)}) ---")
    print(f"  {'Métrica':<22} {'Sem DC':>{col_w}} {'DC thr={:.2f}'.format(best_thr_dc):>{col_w}}")
    print("  " + "─" * (22 + col_w * 2 + 3))
    for name, key, mult, fmt, unit in [
        ("Acurácia geral",    "acc",       100, ".2f", "%"),
        ("Empates previstos", "n_draw_pred", 1, ".0f", ""),
        ("Recall empate",     "draw_rec",  100, ".1f", "%"),
        ("MAE total_goals",   "mae_t",       1, ".4f", ""),
        ("Over/Under 2.5",    "ou_acc",    100, ".2f", "%"),
        ("BTTS accuracy",     "btts_acc",  100, ".2f", "%"),
    ]:
        v0 = r_sub_nodc[key] * mult
        v1 = r_sub_dc[key]   * mult
        d  = v1 - v0
        s0 = f"{v0:{fmt}}{unit}"
        s1 = f"{v1:{fmt}}{unit} ({d:+.2f})"
        print(f"  {name:<22} {s0:>{col_w}}  {s1:>{col_w}}")

    # ── Salva métricas ────────────────────────────────────────────────────
    print(f"\n[7] Salvando metrics_v3.json...")
    metrics = {
        "threshold_draw":      round(best_thr_dc, 2),
        "rho_dixon_coles":     round(best_rho, 3),
        "val_accuracy":        round(r_val_dc_opt["acc"], 4),
        "val_mae_home":        round(r_val_dc_opt["mae_h"], 4),
        "val_mae_away":        round(r_val_dc_opt["mae_a"], 4),
        "val_over25_accuracy": round(r_val_dc_opt["ou_acc"], 4),
        "val_btts_accuracy":   round(r_val_dc_opt["btts_acc"], 4),
        "eafc_real_accuracy":  round(r_sub_dc["acc"], 4),
        "generated_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  {json.dumps(metrics, indent=4)}")

    print(f"\n{SEP}")
    print("  CONCLUÍDO")
    print(SEP)


if __name__ == "__main__":
    main()
