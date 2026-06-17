"""Avalia modelos v3 no subconjunto do val onde ambos os times têm dados EAFC reais."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SEP = "=" * 60

# ── Carrega ───────────────────────────────────────────────────────────────────
with open("models/feature_columns_v3.json") as f:
    feat_cols = json.load(f)
with open("models/metrics_v3.json") as f:
    metrics = json.load(f)

model_h = joblib.load("models/model_goals_home_v3.pkl")
model_a = joblib.load("models/model_goals_away_v3.pkl")
THRESHOLD = metrics["threshold_draw"]

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

eafc = pd.read_csv("data/processed/eafc26_team_features.csv")
eafc["team_en"] = eafc["team"].map(SUPABASE_PT_TO_EN).fillna(eafc["team"])
squad_mean = eafc["eafc_squad"].mean()

print(SEP)
print("  Avaliação v3 — subconjunto EAFC real")
print(SEP)
print(f"\n  Media global eafc_squad = {squad_mean:.6f}")
print(f"  threshold_draw          = {THRESHOLD}")

# ── Val set + filtro ──────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/training_data_v3.csv", parse_dates=["date"])
val = df[df["date"] > "2023-12-31"].copy().reset_index(drop=True)

TOL = 0.01
mask_real = (
    (np.abs(val["home_eafc_squad"] - squad_mean) > TOL) &
    (np.abs(val["away_eafc_squad"] - squad_mean) > TOL)
)
sub = val[mask_real].copy().reset_index(drop=True)

print(f"\n  Val total   : {len(val):,} partidas")
print(f"  EAFC real   : {len(sub):,} partidas ({100*len(sub)/len(val):.1f}%)")
print(f"  Fallback    : {len(val)-len(sub):,} partidas ({100*(1-len(sub)/len(val)):.1f}%)")

# ── Seleções ──────────────────────────────────────────────────────────────────
teams = sorted(set(sub["home_team"].tolist() + sub["away_team"].tolist()))
print(f"\n  Seleções com dados EAFC reais no val ({len(teams)} únicas):")
for i in range(0, len(teams), 5):
    print("    " + "  |  ".join(f"{t:<20}" for t in teams[i:i+5]))

# ── Torneios no subconjunto ───────────────────────────────────────────────────
print(f"\n  Torneios no subconjunto EAFC real:")
for t, cnt in sub["tournament"].value_counts().items():
    print(f"    {t:<45} {cnt:>4}")


# ── Funções de predição ───────────────────────────────────────────────────────
def predict_match_poisson(lh, la, mg=8):
    lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
    hp = poisson.pmf(range(mg + 1), lh)
    ap = poisson.pmf(range(mg + 1), la)
    sm = np.outer(hp, ap)
    p_home = float(sm[np.tril_indices(mg + 1, -1)].sum())
    p_draw = float(np.trace(sm))
    p_away = float(sm[np.triu_indices(mg + 1, 1)].sum())
    p_over = float(sum(sm[i][j] for i in range(mg + 1) for j in range(mg + 1) if i + j > 2))
    p_btts = float((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la)))
    return p_home, p_draw, p_away, p_over, p_btts


def decide(ph, pd_, pa, thr):
    if thr > 0 and pd_ >= thr:
        return "D"
    return "H" if ph >= pa else "A"


# ── Avaliação ─────────────────────────────────────────────────────────────────
def evaluate(subset, label):
    X = subset[feat_cols].values
    lh = model_h.predict(X)
    la = model_a.predict(X)

    y_h = subset["goals_home"].values
    y_a = subset["goals_away"].values
    true_res = np.where(y_h > y_a, "H", np.where(y_h < y_a, "A", "D"))
    total_true = y_h + y_a
    btts_true = ((y_h > 0) & (y_a > 0)).astype(int)

    ph_all, pd_all, pa_all, po_all, pb_all, pred_res = [], [], [], [], [], []
    for i in range(len(subset)):
        ph, pd_, pa, po, pb = predict_match_poisson(lh[i], la[i])
        ph_all.append(ph); pd_all.append(pd_); pa_all.append(pa)
        po_all.append(po); pb_all.append(pb)
        pred_res.append(decide(ph, pd_, pa, THRESHOLD))

    ph_all = np.array(ph_all)
    pd_all = np.array(pd_all)
    po_all = np.array(po_all)
    pb_all = np.array(pb_all)
    pred_res = np.array(pred_res)

    correct = pred_res == true_res
    acc = correct.mean()
    mae_h = mean_absolute_error(y_h, lh)
    mae_a = mean_absolute_error(y_a, la)
    mae_t = mean_absolute_error(total_true, lh + la)
    ou_acc = ((total_true > 2.5) == (po_all > 0.5)).mean()
    btts_acc = (btts_true == (pb_all > 0.5)).mean()

    n = len(subset)
    print(f"\n{SEP}")
    print(f"  {label}  (n={n:,})")
    print(SEP)
    print(f"  Acurácia resultado : {acc*100:.2f}%  ({correct.sum()}/{n})")
    print(f"  MAE goals_home     : {mae_h:.4f}")
    print(f"  MAE goals_away     : {mae_a:.4f}")
    print(f"  MAE total_goals    : {mae_t:.4f}")
    print(f"  Over/Under 2.5     : {ou_acc*100:.2f}%")
    print(f"  BTTS accuracy      : {btts_acc*100:.2f}%")
    print(f"  p_draw range       : min={pd_all.min():.3f}  mean={pd_all.mean():.3f}  max={pd_all.max():.3f}")

    print(f"\n  Distribuição resultado (threshold_draw={THRESHOLD}):")
    hdr = f"  {'Resultado':<12} {'Real':>6} {'Real%':>6}  {'Previsto':>8} {'Prev%':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for res, lbl in [("H", "Home win"), ("D", "Empate"), ("A", "Away win")]:
        n_true = (true_res == res).sum()
        n_pred = (pred_res == res).sum()
        tp = ((true_res == res) & (pred_res == res)).sum()
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec = tp / n_true if n_true > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {lbl:<12} {n_true:>6} {100*n_true/n:>5.1f}%  "
              f"{n_pred:>8} {100*n_pred/n:>5.1f}%  "
              f"{prec*100:>5.1f}%  {rec*100:>6.1f}%  {f1:.3f}")

    return dict(acc=acc, mae_h=mae_h, mae_a=mae_a, mae_t=mae_t,
                ou_acc=ou_acc, btts_acc=btts_acc)


r_all = evaluate(val, "GERAL — val completo (2024+)")
r_sub = evaluate(sub, "SUBCONJUNTO — apenas EAFC real (ambos os times)")

# ── Tabela comparativa ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  COMPARATIVO")
print(SEP)
print(f"  {'Métrica':<22} {'Geral (2517)':>13} {'EAFC real (304)':>16}  {'Delta':>8}")
print("  " + "-" * 62)
comparisons = [
    ("Acurácia resultado", r_all["acc"] * 100,      r_sub["acc"] * 100,      "%",  ".2f"),
    ("MAE goals_home",     r_all["mae_h"],           r_sub["mae_h"],          "",   ".4f"),
    ("MAE goals_away",     r_all["mae_a"],           r_sub["mae_a"],          "",   ".4f"),
    ("MAE total_goals",    r_all["mae_t"],           r_sub["mae_t"],          "",   ".4f"),
    ("Over/Under 2.5",     r_all["ou_acc"] * 100,   r_sub["ou_acc"] * 100,   "%",  ".2f"),
    ("BTTS accuracy",      r_all["btts_acc"] * 100, r_sub["btts_acc"] * 100, "%",  ".2f"),
]
for name, v_all, v_sub, unit, fmt in comparisons:
    delta = v_sub - v_all
    sign = "+" if delta >= 0 else ""
    print(f"  {name:<22} {v_all:>12{fmt}}{unit}  {v_sub:>14{fmt}}{unit}  "
          f"{sign}{delta:{fmt}}{unit}")
print("  " + "-" * 62)

print(f"\n{SEP}")
print("  CONCLUIDO")
print(SEP)
