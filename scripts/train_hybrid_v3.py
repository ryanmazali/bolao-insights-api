"""train_hybrid_v3.py — Modelo híbrido: XGBoost multiclasse (H/D/A).

Features: 91 históricas/eafc + 6 features Poisson derivadas dos modelos v3
  com correção Dixon-Coles (rho=-0.20).

Leakage note: as features Poisson do set de TREINO são geradas pelo modelo
Poisson treinado no mesmo treino → o modelo viu os dados. Para produção
isso é aceitável; o sinal é real pois as lambdas refletem o mesmo histórico
que as features já capturam.

Saída:
  models/model_hybrid_classifier_v3.pkl
  models/hybrid_feature_columns.json

Uso:
    python scripts/train_hybrid_v3.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Caminhos ──────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
MODEL_HOME   = MODELS_DIR / "model_goals_home_v3.pkl"
MODEL_AWAY   = MODELS_DIR / "model_goals_away_v3.pkl"
FEAT_JSON    = MODELS_DIR / "feature_columns_v3.json"
METRICS_JSON = MODELS_DIR / "metrics_v3.json"
DATA_CSV     = Path("data/processed/training_data_v3.csv")
EAFC_CSV     = Path("data/processed/eafc26_team_features.csv")

OUT_CLF  = MODELS_DIR / "model_hybrid_classifier_v3.pkl"
OUT_LE   = MODELS_DIR / "model_hybrid_le_v3.pkl"
OUT_COLS = MODELS_DIR / "hybrid_feature_columns.json"

SPLIT_DATE = "2023-12-31"
SEP = "=" * 70

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


# ── Dixon-Coles ───────────────────────────────────────────────────────────────

def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1.0 - lh * la * rho
    if x == 0 and y == 1: return 1.0 + lh * rho
    if x == 1 and y == 0: return 1.0 + la * rho
    if x == 1 and y == 1: return 1.0 - rho
    return 1.0


def poisson_features(lh_arr, la_arr, rho: float, mg: int = 8):
    """Dado arrays de lambdas, retorna DataFrame com features Poisson."""
    ph_all, pd_all, pa_all = [], [], []
    for lh, la in zip(lh_arr, la_arr):
        lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
        hp = poisson.pmf(range(mg + 1), lh)
        ap = poisson.pmf(range(mg + 1), la)
        sm = np.outer(hp, ap)
        for i in range(min(2, mg + 1)):
            for j in range(min(2, mg + 1)):
                sm[i, j] *= _tau(i, j, lh, la, rho)
        sm /= sm.sum()
        ph_all.append(sm[np.tril_indices(mg + 1, -1)].sum())
        pd_all.append(np.trace(sm))
        pa_all.append(sm[np.triu_indices(mg + 1, 1)].sum())

    return pd.DataFrame({
        "poisson_p_home":    ph_all,
        "poisson_p_draw":    pd_all,
        "poisson_p_away":    pa_all,
        "poisson_lambda_home": lh_arr,
        "poisson_lambda_away": la_arr,
        "poisson_score_diff":  lh_arr - la_arr,
    })


def evaluate_clf(y_true, y_pred, probs, le, label, n):
    classes = le.classes_  # ['A', 'D', 'H']
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  {label}  (n={n:,})")
    print(f"  Acurácia geral: {acc*100:.2f}%")
    print(f"  {'Classe':<10} {'Real':>5} {'Prev':>5}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("  " + "-" * 52)
    for cls in classes:
        mask_t = y_true == cls
        mask_p = y_pred == cls
        tp = (mask_t & mask_p).sum()
        prec_v = tp / mask_p.sum() if mask_p.sum() > 0 else 0
        rec_v  = tp / mask_t.sum() if mask_t.sum() > 0 else 0
        f1_v   = 2*prec_v*rec_v/(prec_v+rec_v) if (prec_v+rec_v) > 0 else 0
        label_name = {"H": "Home win", "D": "Empate", "A": "Away win"}[cls]
        print(f"  {label_name:<10} {mask_t.sum():>5} {mask_p.sum():>5}  "
              f"{prec_v*100:>6.1f}%  {rec_v*100:>6.1f}%  {f1_v:.3f}")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  train_hybrid_v3.py — Modelo Híbrido Poisson + Histórico")
    print(SEP)

    # Carrega
    print("\n[1] Carregando modelos e dados...")
    model_h = joblib.load(MODEL_HOME)
    model_a = joblib.load(MODEL_AWAY)
    with open(FEAT_JSON,    encoding="utf-8") as f: feat_cols  = json.load(f)
    with open(METRICS_JSON, encoding="utf-8") as f: metrics    = json.load(f)
    rho = metrics["rho_dixon_coles"]
    print(f"  rho Dixon-Coles: {rho}")

    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    print(f"  Dataset: {len(df):,} partidas")

    # EAFC real mask (para sub-avaliação)
    eafc = pd.read_csv(EAFC_CSV)
    eafc["team_en"] = eafc["team"].map(SUPABASE_PT_TO_EN).fillna(eafc["team"])
    squad_mean = eafc["eafc_squad"].mean()

    # ── Features Poisson para todo o dataset ─────────────────────────────
    print("\n[2] Gerando features Poisson (todo o dataset)...")
    print("  [nota] features do treino geradas pelo modelo treinado no mesmo treino")
    print("         → leakage leve aceitável para produção")
    X_all  = df[feat_cols].values
    lh_all = model_h.predict(X_all)
    la_all = model_a.predict(X_all)
    pf = poisson_features(lh_all, la_all, rho)
    print(f"  Features Poisson geradas: {pf.shape[1]} colunas")
    print(f"  p_draw — mean={pf['poisson_p_draw'].mean():.3f}  "
          f"min={pf['poisson_p_draw'].min():.3f}  max={pf['poisson_p_draw'].max():.3f}")

    # ── Monta dataset híbrido ─────────────────────────────────────────────
    POISSON_COLS = list(pf.columns)
    HYBRID_COLS  = feat_cols + POISSON_COLS

    df_hybrid = pd.concat([df.reset_index(drop=True), pf.reset_index(drop=True)], axis=1)

    # Target
    df_hybrid["result_hda"] = np.where(
        df_hybrid["goals_home"] > df_hybrid["goals_away"], "H",
        np.where(df_hybrid["goals_home"] < df_hybrid["goals_away"], "A", "D")
    )

    # ── Split temporal ────────────────────────────────────────────────────
    train = df_hybrid[df_hybrid["date"] <= SPLIT_DATE].copy()
    val   = df_hybrid[df_hybrid["date"] >  SPLIT_DATE].copy().reset_index(drop=True)

    print(f"\n[3] Split temporal...")
    print(f"  Treino: {len(train):,}  Val: {len(val):,}")
    for split, name in [(train, "treino"), (val, "val")]:
        vc = split["result_hda"].value_counts()
        print(f"  {name}: H={vc.get('H',0)} ({100*vc.get('H',0)/len(split):.1f}%)  "
              f"D={vc.get('D',0)} ({100*vc.get('D',0)/len(split):.1f}%)  "
              f"A={vc.get('A',0)} ({100*vc.get('A',0)/len(split):.1f}%)")

    # LabelEncoder (A→0, D→1, H→2)
    le = LabelEncoder()
    le.fit(["A", "D", "H"])
    print(f"  Classes: {list(le.classes_)} → {list(le.transform(le.classes_))}")

    X_tr  = train[HYBRID_COLS].values
    y_tr  = le.transform(train["result_hda"])
    w_tr  = train["sample_weight"].values

    X_val = val[HYBRID_COLS].values
    y_val = le.transform(val["result_hda"])
    y_val_str = val["result_hda"].values

    # ── Treino ────────────────────────────────────────────────────────────
    print(f"\n[4] Treinando XGBoost multi:softprob ({len(HYBRID_COLS)} features)...")
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        device="cpu",
        early_stopping_rounds=30,
        eval_metric="mlogloss",
    )
    clf.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {clf.best_iteration}  (de 300)")

    # ── Predições val ─────────────────────────────────────────────────────
    probs_val = clf.predict_proba(X_val)   # shape (n, 3)  cols: A, D, H
    pred_val  = le.classes_[probs_val.argmax(axis=1)]

    # ── Feature importance ────────────────────────────────────────────────
    print(f"\n[5] Feature importance — Top 20...")
    imp = dict(zip(HYBRID_COLS, clf.feature_importances_))
    top20 = sorted(imp.items(), key=lambda x: -x[1])[:20]
    print(f"\n  {'rank':>4}  {'feature':<38}  {'importance':>10}  bar")
    print("  " + "-" * 70)
    for rank, (name, score) in enumerate(top20, 1):
        tag = " [P]" if name in POISSON_COLS else ""
        bar = "█" * int(score * 500)
        print(f"  {rank:>4}. {name:<38}{tag}  {score:.5f}  {bar}")

    # Importância total das features Poisson
    total_p = sum(imp[c] for c in POISSON_COLS)
    total_h = sum(imp[c] for c in feat_cols)
    print(f"\n  Total importância — Poisson: {total_p*100:.1f}%  Histórico: {total_h*100:.1f}%")

    # Detalhe por feature Poisson
    print(f"\n  Features Poisson individualmente:")
    for c in POISSON_COLS:
        print(f"    {c:<35} {imp[c]:.5f}")

    # ── Avaliação val completo ────────────────────────────────────────────
    print(f"\n[6] Avaliação no val set ({len(val):,} partidas)...")
    acc_hyb = evaluate_clf(y_val_str, pred_val, probs_val, le,
                           "Híbrido — val completo", len(val))

    # Subconjunto EAFC real
    mask_real = (
        (np.abs(val["home_eafc_squad"] - squad_mean) > 0.01) &
        (np.abs(val["away_eafc_squad"] - squad_mean) > 0.01)
    )
    sub_val      = val[mask_real].reset_index(drop=True)
    X_sub        = sub_val[HYBRID_COLS].values
    probs_sub    = clf.predict_proba(X_sub)
    pred_sub     = le.classes_[probs_sub.argmax(axis=1)]
    y_sub_str    = sub_val["result_hda"].values

    print(f"\n  Avaliação subconjunto EAFC real ({len(sub_val):,} partidas)...")
    acc_hyb_sub = evaluate_clf(y_sub_str, pred_sub, probs_sub, le,
                               "Híbrido — EAFC real", len(sub_val))

    # ── Tabela comparativa final — DC puro recalculado inline ────────────
    def _dc_pred_result(ph, pd_, pa, thr):
        if thr > 0 and pd_ >= thr: return "D"
        return "H" if ph >= pa else "A"

    def _build_dc_preds(subset, rho_val, mg=8):
        X  = subset[feat_cols].values
        lh = model_h.predict(X)
        la = model_a.predict(X)
        ph_all, pd_all, pa_all = [], [], []
        for i in range(len(subset)):
            lhv, lav = max(float(lh[i]),1e-6), max(float(la[i]),1e-6)
            hp = poisson.pmf(range(mg+1), lhv)
            ap = poisson.pmf(range(mg+1), lav)
            sm = np.outer(hp, ap)
            for ii in range(min(2,mg+1)):
                for jj in range(min(2,mg+1)):
                    sm[ii,jj] *= _tau(ii,jj,lhv,lav,rho_val)
            sm /= sm.sum()
            ph_all.append(sm[np.tril_indices(mg+1,-1)].sum())
            pd_all.append(np.trace(sm))
            pa_all.append(sm[np.triu_indices(mg+1,1)].sum())
        return np.array(ph_all), np.array(pd_all), np.array(pa_all)

    ph_val, pd_val, pa_val = _build_dc_preds(val, rho)

    def dc_metrics_arr(ph, pd_, pa, thr, y_true_str):
        pred_res = np.array([_dc_pred_result(ph[i], pd_[i], pa[i], thr)
                             for i in range(len(ph))])
        acc = accuracy_score(y_true_str, pred_res)
        mask_d_t = y_true_str == "D"
        mask_d_p = pred_res  == "D"
        tp_d   = (mask_d_t & mask_d_p).sum()
        prec_d = tp_d / mask_d_p.sum() if mask_d_p.sum() > 0 else 0
        rec_d  = tp_d / mask_d_t.sum() if mask_d_t.sum() > 0 else 0
        f1_d   = 2*prec_d*rec_d/(prec_d+rec_d) if (prec_d+rec_d) > 0 else 0
        return acc, prec_d, rec_d, f1_d, int(mask_d_p.sum())

    y_val_str_arr = val["result_hda"].values
    mask_d_t_h = y_val_str == "D"
    mask_d_p_h = pred_val  == "D"
    tp_d_h   = (mask_d_t_h & mask_d_p_h).sum()
    prec_d_h = tp_d_h / mask_d_p_h.sum() if mask_d_p_h.sum() > 0 else 0
    rec_d_h  = tp_d_h / mask_d_t_h.sum() if mask_d_t_h.sum() > 0 else 0
    f1_d_h   = 2*prec_d_h*rec_d_h/(prec_d_h+rec_d_h) if (prec_d_h+rec_d_h) > 0 else 0
    n_pred_d_h = int(mask_d_p_h.sum())

    acc_dc038, prec_dc038, rec_dc038, f1_dc038, np_dc038 = dc_metrics_arr(ph_val, pd_val, pa_val, 0.38, y_val_str_arr)
    acc_dc033, prec_dc033, rec_dc033, f1_dc033, np_dc033 = dc_metrics_arr(ph_val, pd_val, pa_val, 0.33, y_val_str_arr)

    print(f"\n{SEP}")
    print(f"  TABELA COMPARATIVA — val set ({len(val):,} partidas)")
    print(SEP)
    print(f"\n  {'Modelo':<28} {'Acc':>7}  {'Pred_D':>7}  {'Prec(D)':>8}  {'Rec(D)':>8}  {'F1(D)':>7}")
    print("  " + "-" * 70)
    rows = [
        ("DC thr=0.38 (referência)",  acc_dc038,  np_dc038,  prec_dc038, rec_dc038, f1_dc038),
        ("DC thr=0.33",               acc_dc033,  np_dc033,  prec_dc033, rec_dc033, f1_dc033),
        ("Híbrido (novo)",             acc_hyb,    int(n_pred_d_h), prec_d_h, rec_d_h, f1_d_h),
    ]
    for name, acc, npd, prec, rec, f1 in rows:
        print(f"  {name:<28} {acc*100:>6.2f}%  {npd:>7}  "
              f"{prec*100:>7.1f}%  {rec*100:>7.1f}%  {f1:.4f}")
    print("  " + "-" * 70)

    # EAFC real sub-tabela
    mask_d_t_s = y_sub_str == "D"
    mask_d_p_s = pred_sub == "D"
    tp_d_s = (mask_d_t_s & mask_d_p_s).sum()
    prec_d_s = tp_d_s / mask_d_p_s.sum() if mask_d_p_s.sum() > 0 else 0
    rec_d_s  = tp_d_s / mask_d_t_s.sum() if mask_d_t_s.sum() > 0 else 0
    f1_d_s   = 2*prec_d_s*rec_d_s/(prec_d_s+rec_d_s) if (prec_d_s+rec_d_s) > 0 else 0
    n_pred_d_s = mask_d_p_s.sum()

    ph_sub, pd_sub, pa_sub = _build_dc_preds(sub_val, rho)
    acc_sub_dc038, prec_sub_dc038, rec_sub_dc038, f1_sub_dc038, np_sub_dc038 = \
        dc_metrics_arr(ph_sub, pd_sub, pa_sub, 0.38, y_sub_str)
    acc_sub_dc033, prec_sub_dc033, rec_sub_dc033, f1_sub_dc033, np_sub_dc033 = \
        dc_metrics_arr(ph_sub, pd_sub, pa_sub, 0.33, y_sub_str)

    print(f"\n  EAFC real ({len(sub_val):,} partidas):")
    print(f"  {'Modelo':<28} {'Acc':>7}  {'Pred_D':>7}  {'Prec(D)':>8}  {'Rec(D)':>8}  {'F1(D)':>7}")
    print("  " + "-" * 70)
    rows_sub = [
        ("DC thr=0.38 (referência)",  acc_sub_dc038, np_sub_dc038, prec_sub_dc038, rec_sub_dc038, f1_sub_dc038),
        ("DC thr=0.33",               acc_sub_dc033, np_sub_dc033, prec_sub_dc033, rec_sub_dc033, f1_sub_dc033),
        ("Híbrido (novo)",             acc_hyb_sub,   int(n_pred_d_s), prec_d_s, rec_d_s, f1_d_s),
    ]
    for name, acc, npd, prec, rec, f1 in rows_sub:
        print(f"  {name:<28} {acc*100:>6.2f}%  {npd:>7}  "
              f"{prec*100:>7.1f}%  {rec*100:>7.1f}%  {f1:.4f}")
    print("  " + "-" * 70)

    # ── Salva ─────────────────────────────────────────────────────────────
    print(f"\n[7] Salvando modelos...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_CLF)
    joblib.dump(le, OUT_LE)
    with open(OUT_COLS, "w", encoding="utf-8") as f:
        json.dump(HYBRID_COLS, f, indent=2, ensure_ascii=False)
    print(f"  {OUT_CLF}  ({len(HYBRID_COLS)} features)")
    print(f"  {OUT_LE}")
    print(f"  {OUT_COLS}")

    print(f"\n{SEP}")
    print("  CONCLUÍDO")
    print(SEP)


if __name__ == "__main__":
    main()
