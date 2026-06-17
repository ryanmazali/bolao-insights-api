"""Avaliação v3 corrigido com Dixon-Coles + threshold=0.33."""
import json
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SEP = "=" * 64

with open("models/feature_columns_v3.json") as f:
    feat_cols = json.load(f)
with open("models/metrics_v3.json") as f:
    metrics = json.load(f)

model_h = joblib.load("models/model_goals_home_v3.pkl")
model_a = joblib.load("models/model_goals_away_v3.pkl")
THRESHOLD = metrics["threshold_draw"]    # 0.33
RHO       = metrics["rho_dixon_coles"]  # -0.0686

eafc = pd.read_csv("data/processed/eafc26_team_features.csv")
squad_mean = eafc["eafc_squad"].mean()
TOL = 0.01

df  = pd.read_csv("data/processed/training_data_v3.csv", parse_dates=["date"])
val = df[df["date"] > "2023-12-31"].copy().reset_index(drop=True)
mask_real = (
    (np.abs(val["home_eafc_squad"] - squad_mean) > TOL) &
    (np.abs(val["away_eafc_squad"] - squad_mean) > TOL)
)
sub = val[mask_real].copy().reset_index(drop=True)


def _tau(x, y, lhv, lav, rho):
    if x == 0 and y == 0: return 1.0 - lhv * lav * rho
    if x == 0 and y == 1: return 1.0 + lhv * rho
    if x == 1 and y == 0: return 1.0 + lav * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0


def predict_dc(lhv, lav, rho, mg=8):
    lhv, lav = max(float(lhv), 1e-6), max(float(lav), 1e-6)
    hp = poisson.pmf(range(mg + 1), lhv)
    ap = poisson.pmf(range(mg + 1), lav)
    sm = np.outer(hp, ap)
    for i in range(min(2, mg + 1)):
        for j in range(min(2, mg + 1)):
            sm[i, j] *= _tau(i, j, lhv, lav, rho)
    sm /= sm.sum()
    ph = float(sm[np.tril_indices(mg + 1, -1)].sum())
    pd_ = float(np.trace(sm))
    pa = float(sm[np.triu_indices(mg + 1, 1)].sum())
    po = float(sum(sm[i, j] for i in range(mg+1) for j in range(mg+1) if i+j > 2))
    pb = float((1 - poisson.pmf(0, lhv)) * (1 - poisson.pmf(0, lav)))
    return ph, pd_, pa, po, pb


def decide(ph, pd_, pa, thr):
    if thr > 0 and pd_ >= thr:
        return "D"
    return "H" if ph >= pa else "A"


def evaluate(subset, label):
    X = subset[feat_cols].values
    lh = model_h.predict(X)
    la = model_a.predict(X)
    y_h = subset["goals_home"].values
    y_a = subset["goals_away"].values
    total_true = y_h + y_a
    true_res = np.where(y_h > y_a, "H", np.where(y_h < y_a, "A", "D"))

    ph_all, pd_all, pa_all, po_all, pb_all, pred_res = [], [], [], [], [], []
    for i in range(len(subset)):
        ph, pd_, pa, po, pb = predict_dc(lh[i], la[i], RHO)
        ph_all.append(ph); pd_all.append(pd_); pa_all.append(pa)
        po_all.append(po); pb_all.append(pb)
        pred_res.append(decide(ph, pd_, pa, THRESHOLD))

    ph_all  = np.array(ph_all);  pd_all = np.array(pd_all)
    po_all  = np.array(po_all);  pb_all = np.array(pb_all)
    pred_res = np.array(pred_res)

    correct  = pred_res == true_res
    acc      = correct.mean()
    mae_h    = mean_absolute_error(y_h, lh)
    mae_a    = mean_absolute_error(y_a, la)
    mae_t    = mean_absolute_error(total_true, lh + la)
    ou_acc   = ((total_true > 2.5) == (po_all > 0.5)).mean()
    btts_t   = ((y_h > 0) & (y_a > 0)).astype(int)
    btts_acc = (btts_t == (pb_all > 0.5)).mean()

    n = len(subset)
    print(f"\n{SEP}")
    print(f"  {label}  (n={n:,})")
    print(SEP)
    print(f"  Acurácia resultado : {acc*100:.2f}%  ({correct.sum()}/{n})")
    print(f"  MAE goals_home     : {mae_h:.4f}")
    print(f"  MAE goals_away     : {mae_a:.4f}")
    print(f"  MAE total_goals    : {mae_t:.4f}")
    print(f"  Over/Under 2.5     : {ou_acc*100:.2f}%")
    print(f"  p_draw (DC) range  : min={pd_all.min():.3f}  mean={pd_all.mean():.3f}  max={pd_all.max():.3f}")

    print(f"\n  Distribuição resultado (threshold={THRESHOLD}, rho={RHO:.4f}):")
    hdr = f"  {'Resultado':<12} {'Real':>6} {'Real%':>6}  {'Previsto':>8} {'Prev%':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    recalls = {}
    for res, lbl in [("H", "Home win"), ("D", "Empate"), ("A", "Away win")]:
        n_true = (true_res == res).sum()
        n_pred = (pred_res == res).sum()
        tp   = ((true_res == res) & (pred_res == res)).sum()
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec  = tp / n_true if n_true > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        recalls[res] = rec
        print(f"  {lbl:<12} {n_true:>6} {100*n_true/n:>5.1f}%  "
              f"{n_pred:>8} {100*n_pred/n:>5.1f}%  "
              f"{prec*100:>5.1f}%  {rec*100:>6.1f}%  {f1:.3f}")

    return dict(acc=acc, mae_h=mae_h, mae_a=mae_a, mae_t=mae_t,
                ou_acc=ou_acc, btts_acc=btts_acc, recalls=recalls)


print(SEP)
print("  Avaliação v3 corrigido — Dixon-Coles + threshold=0.33")
print(SEP)
print(f"  rho={RHO:.4f}  threshold={THRESHOLD}")
print(f"  Val total: {len(val):,}  |  EAFC real: {len(sub):,} ({100*len(sub)/len(val):.1f}%)")

r_all = evaluate(val, "GERAL — val completo (2024+)")
r_sub = evaluate(sub, "SUBCONJUNTO — apenas EAFC real (ambos os times)")

# ── Tabela comparativa ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TABELA COMPARATIVA  v3 anterior  vs  v3 corrigido")
print(SEP)
ant = dict(acc_geral=58.16, acc_eafc=44.74, mae=1.394, recall_d=23.0)
print(f"  {'Métrica':<26} {'v3 anterior':>13} {'v3 corrigido':>14}  {'Delta':>8}")
print("  " + "-" * 64)
rows = [
    ("Acurácia geral",       ant["acc_geral"], r_all["acc"] * 100,          "%", ".2f"),
    ("Acurácia EAFC real",   ant["acc_eafc"],  r_sub["acc"] * 100,          "%", ".2f"),
    ("MAE total (geral)",    ant["mae"],        r_all["mae_t"],              "",  ".4f"),
    ("Recall empate (geral)",ant["recall_d"],   r_all["recalls"]["D"] * 100, "%", ".1f"),
]
for name, v_ant, v_new, unit, fmt in rows:
    delta = v_new - v_ant
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<26} {v_ant:>12{fmt}}{unit}  {v_new:>13{fmt}}{unit}  {sign}{delta:{fmt}}{unit}")
print("  " + "-" * 64)
print(f"\n  rho: -0.2000 → {RHO:.4f}  (estimado via MLE no val set)")
print(f"  threshold: 0.38 → {THRESHOLD}")
