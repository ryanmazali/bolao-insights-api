"""train_models_v3.py — Treina os modelos v3 de gols (home e away separados).

Targets:
  model_goals_home_v3: goals_home  (Poisson XGBoost)
  model_goals_away_v3: goals_away  (Poisson XGBoost)

Split temporal:
  Treino : data <= 2023-12-31
  Val    : data >= 2024-01-01

Uso:
    python scripts/train_models_v3.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Caminhos ──────────────────────────────────────────────────────────────────
DATA_CSV      = Path("data/processed/training_data_v3.csv")
FEAT_JSON     = Path("data/processed/feature_columns_v3.json")
V2_DATA_CSV   = Path("data/processed/match_features_v2.csv")
V2_MODEL_PKL  = Path("src/models/saved/model_goals_v2.pkl")
V2_FEAT_JSON  = Path("src/models/saved/feature_columns_v2.json")

OUT_DIR = Path("models")
OUT_HOME = OUT_DIR / "model_goals_home_v3.pkl"
OUT_AWAY = OUT_DIR / "model_goals_away_v3.pkl"
OUT_JSON = OUT_DIR / "feature_columns_v3.json"

SPLIT_DATE = "2023-12-31"
SEP = "=" * 70

XGB_PARAMS = dict(
    objective="count:poisson",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    device="cpu",
)


# ── Utilidades ────────────────────────────────────────────────────────────────

def derive_result(gh, ga):
    """Retorna 'H', 'D', 'A' a partir de arrays de gols."""
    return np.where(gh > ga, "H", np.where(gh < ga, "A", "D"))


def result_accuracy(y_home, y_away, pred_home, pred_away):
    true_res = derive_result(y_home.values, y_away.values)
    pred_res = derive_result(pred_home, pred_away)
    return (true_res == pred_res).mean()


def over_under_accuracy(y_home, y_away, pred_home, pred_away, line=2.5):
    true_over = ((y_home + y_away) > line).astype(int)
    pred_over = ((pred_home + pred_away) > line).astype(int)
    return (true_over == pred_over).mean()


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    with open(FEAT_JSON, encoding="utf-8") as f:
        feat_cols = json.load(f)
    print(f"  Dataset: {df.shape[0]:,} partidas × {df.shape[1]} colunas")
    print(f"  Features: {len(feat_cols)}")
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [AVISO] Features ausentes: {missing}")
    return df, feat_cols


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(df, feat_cols):
    train = df[df["date"] <= SPLIT_DATE].copy()
    val   = df[df["date"] >  SPLIT_DATE].copy()
    print(f"  Treino : {len(train):,} partidas  "
          f"({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Val    : {len(val):,} partidas  "
          f"({val['date'].min().date()} → {val['date'].max().date()})")

    X_tr  = train[feat_cols].values
    X_val = val[feat_cols].values
    w_tr  = train["sample_weight"].values

    return X_tr, X_val, train, val, w_tr


# ── Treino ────────────────────────────────────────────────────────────────────

def train_model(name, X_tr, y_tr, w_tr, X_val, y_val, feat_cols):
    print(f"\n  Treinando {name}...")
    model = XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=50,
        eval_metric="poisson-nloglik",
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best = model.best_iteration
    print(f"  Best iteration: {best}  (de {XGB_PARAMS['n_estimators']})")
    return model


# ── Feature importance ────────────────────────────────────────────────────────

def print_importance(model, feat_cols, label, top=15):
    scores = model.feature_importances_
    pairs = sorted(zip(feat_cols, scores), key=lambda x: -x[1])
    print(f"\n── Top {top} features — {label} ──")
    for rank, (name, score) in enumerate(pairs[:top], 1):
        bar = "█" * int(score * 400)
        print(f"  {rank:>2}. {name:<35} {score:.4f}  {bar}")


# ── Avaliação principal ───────────────────────────────────────────────────────

def evaluate(model_h, model_a, X_val, val_df, feat_cols, label="v3"):
    pred_h = model_h.predict(X_val)
    pred_a = model_a.predict(X_val)

    y_h = val_df["goals_home"]
    y_a = val_df["goals_away"]

    mae_h = mean_absolute_error(y_h, pred_h)
    mae_a = mean_absolute_error(y_a, pred_a)
    mae_total = mean_absolute_error(y_h + y_a, pred_h + pred_a)

    res_acc  = result_accuracy(y_h, y_a, pred_h, pred_a)
    ou_acc   = over_under_accuracy(y_h, y_a, pred_h, pred_a)

    baseline_h = val_df["goals_home"].mean()  # usa só val (justo)
    # Baseline correto: média do treino aplicada ao val
    return {
        "label":     label,
        "mae_home":  mae_h,
        "mae_away":  mae_a,
        "mae_total": mae_total,
        "res_acc":   res_acc,
        "ou_acc":    ou_acc,
        "pred_h":    pred_h,
        "pred_a":    pred_a,
    }


def evaluate_baseline(train_df, val_df):
    """Baseline ingênuo: média do treino para cada target."""
    bl_h = train_df["goals_home"].mean()
    bl_a = train_df["goals_away"].mean()
    y_h = val_df["goals_home"]
    y_a = val_df["goals_away"]
    pred_h = np.full(len(val_df), bl_h)
    pred_a = np.full(len(val_df), bl_a)
    return {
        "label":     "baseline (média treino)",
        "mae_home":  mean_absolute_error(y_h, pred_h),
        "mae_away":  mean_absolute_error(y_a, pred_a),
        "mae_total": mean_absolute_error(y_h + y_a, pred_h + pred_a),
        "res_acc":   result_accuracy(y_h, y_a, pred_h, pred_a),
        "ou_acc":    over_under_accuracy(y_h, y_a, pred_h, pred_a),
        "pred_h":    pred_h,
        "pred_a":    pred_a,
    }


def evaluate_v2(val_df):
    """Tenta carregar model_goals_v2 e avaliar no mesmo split temporal."""
    if not V2_MODEL_PKL.exists() or not V2_DATA_CSV.exists():
        return None

    try:
        model_v2 = joblib.load(V2_MODEL_PKL)

        # Features do v2 (lê do JSON se existir, senão usa as do train_v2.py)
        if V2_FEAT_JSON.exists():
            with open(V2_FEAT_JSON, encoding="utf-8") as f:
                feat_v2 = json.load(f)
        else:
            feat_v2 = [
                'home_fifa_points', 'home_goals_for', 'home_goals_against',
                'home_goal_diff', 'home_win_rate', 'home_draw_rate',
                'home_btts_rate', 'home_clean_sheet',
                'home_form_goals_for', 'home_form_goals_against',
                'home_form_win_rate', 'home_form5_pts',
                'home_win_rate_home', 'home_win_rate_neutral',
                'home_avg_opp_points', 'home_sos_goals_for',
                'home_sos_goals_against', 'home_sos_form',
                'home_fm23_attack_strength', 'home_fm23_best_attacker', 'home_fm23_top3_attack',
                'home_fm23_defense_strength', 'home_fm23_best_defender', 'home_fm23_top5_defense',
                'home_fm23_overall', 'home_fm23_best_overall', 'home_fm23_top11_overall',
                'home_fm23_depth_overall', 'home_fm23_std_overall', 'home_fm23_gk_strength',
                'away_fifa_points', 'away_goals_for', 'away_goals_against',
                'away_goal_diff', 'away_win_rate', 'away_draw_rate',
                'away_btts_rate', 'away_clean_sheet',
                'away_form_goals_for', 'away_form_goals_against',
                'away_form_win_rate', 'away_form5_pts',
                'away_win_rate_away', 'away_win_rate_neutral',
                'away_avg_opp_points', 'away_sos_goals_for',
                'away_sos_goals_against', 'away_sos_form',
                'away_fm23_attack_strength', 'away_fm23_best_attacker', 'away_fm23_top3_attack',
                'away_fm23_defense_strength', 'away_fm23_best_defender', 'away_fm23_top5_defense',
                'away_fm23_overall', 'away_fm23_best_overall', 'away_fm23_top11_overall',
                'away_fm23_depth_overall', 'away_fm23_std_overall', 'away_fm23_gk_strength',
                'diff_fifa_points', 'diff_goals_for', 'diff_goals_against',
                'diff_win_rate', 'diff_form_win_rate', 'diff_form5',
                'diff_sos_goals', 'diff_sos_form', 'diff_avg_opp',
                'fm23_attack_diff', 'fm23_defense_diff', 'fm23_overall_diff',
                'fm23_best_overall_diff', 'fm23_top3_attack_diff', 'fm23_top5_defense_diff',
                'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_goals_avg', 'h2h_n',
                'is_neutral', 'tournament_weight',
            ]

        df_v2 = pd.read_csv(V2_DATA_CSV, parse_dates=["date"])
        df_v2_val = df_v2[df_v2["date"] > SPLIT_DATE].copy()

        missing = [c for c in feat_v2 if c not in df_v2_val.columns]
        if missing:
            print(f"  [v2 skip] Colunas ausentes no v2 val: {len(missing)}")
            return None

        df_v2_val = df_v2_val.dropna(subset=feat_v2 + ["total_goals"])
        X_v2_val = df_v2_val[feat_v2].values
        y_total_v2 = df_v2_val["total_goals"]
        pred_total_v2 = model_v2.predict(X_v2_val)

        mae_total_v2 = mean_absolute_error(y_total_v2, pred_total_v2)

        # Deriva resultado (v2 não separa home/away — usamos metade como proxy)
        pred_h_v2 = pred_total_v2 * 0.591   # proporção média histórica home
        pred_a_v2 = pred_total_v2 * (1 - 0.591)
        y_h = df_v2_val.get("home_score", df_v2_val.get("goals_home", None))
        y_a = df_v2_val.get("away_score", df_v2_val.get("goals_away", None))

        res_acc  = None
        ou_acc   = None
        if y_h is not None and y_a is not None:
            res_acc = result_accuracy(y_h, y_a, pred_h_v2, pred_a_v2)
            ou_acc  = over_under_accuracy(y_h, y_a, pred_h_v2, pred_a_v2)

        return {
            "label":        f"model_goals_v2 (n={len(df_v2_val):,})",
            "mae_home":     None,
            "mae_away":     None,
            "mae_total":    mae_total_v2,
            "res_acc":      res_acc,
            "ou_acc":       ou_acc,
        }
    except Exception as e:
        print(f"  [v2 skip] Erro ao carregar v2: {e}")
        return None


# ── Impressão de comparação ───────────────────────────────────────────────────

def print_comparison(results):
    print(f"\n{'─'*70}")
    print(f"  {'Modelo':<35} {'MAE_home':>8} {'MAE_away':>8} {'MAE_tot':>8} "
          f"{'Res%':>7} {'O/U%':>7}")
    print(f"{'─'*70}")
    for r in results:
        mae_h = f"{r['mae_home']:.4f}" if r["mae_home"] is not None else "     —"
        mae_a = f"{r['mae_away']:.4f}" if r["mae_away"] is not None else "     —"
        mae_t = f"{r['mae_total']:.4f}" if r["mae_total"] is not None else "     —"
        res   = f"{r['res_acc']*100:.1f}%" if r["res_acc"] is not None else "    —"
        ou    = f"{r['ou_acc']*100:.1f}%" if r["ou_acc"] is not None else "    —"
        print(f"  {r['label']:<35} {mae_h:>8} {mae_a:>8} {mae_t:>8} {res:>7} {ou:>7}")
    print(f"{'─'*70}")


# ── Distribuição de predições ─────────────────────────────────────────────────

def print_pred_dist(res, val_df, label):
    print(f"\n── Distribuição de predições — {label} ──")
    pred_h = np.round(res["pred_h"]).astype(int).clip(0, 10)
    pred_a = np.round(res["pred_a"]).astype(int).clip(0, 10)
    y_h    = val_df["goals_home"].values
    y_a    = val_df["goals_away"].values

    for name, pred, true in [("home", pred_h, y_h), ("away", pred_a, y_a)]:
        from collections import Counter
        true_c = Counter(true)
        pred_c = Counter(pred)
        print(f"  {name}:  real=[{', '.join(f'{true_c.get(i,0)}' for i in range(6))}]  "
              f"pred=[{', '.join(f'{pred_c.get(i,0)}' for i in range(6))}]  (idx 0-5 gols)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  train_models_v3.py — XGBoost Poisson goals_home + goals_away")
    print(SEP)

    # 1. Carga
    print("\n[1] Carregando dados...")
    df, feat_cols = load_data()

    # 2. Split
    print("\n[2] Split temporal...")
    X_tr, X_val, train_df, val_df, w_tr = split_data(df, feat_cols)

    y_tr_h = train_df["goals_home"].values
    y_tr_a = train_df["goals_away"].values
    y_val_h = val_df["goals_home"].values
    y_val_a = val_df["goals_away"].values

    print(f"\n  Média treino  — goals_home: {y_tr_h.mean():.3f}  goals_away: {y_tr_a.mean():.3f}")
    print(f"  Média val     — goals_home: {y_val_h.mean():.3f}  goals_away: {y_val_a.mean():.3f}")

    # 3. Treino
    print(f"\n[3] Treinando modelos (XGBoost Poisson)...")
    model_h = train_model("model_goals_home_v3", X_tr, y_tr_h, w_tr, X_val, y_val_h, feat_cols)
    model_a = train_model("model_goals_away_v3", X_tr, y_tr_a, w_tr, X_val, y_val_a, feat_cols)

    # 4. Avaliação
    print(f"\n[4] Avaliando no set de validação ({len(val_df):,} partidas — 2024+)...")
    res_v3       = evaluate(model_h, model_a, X_val, val_df, feat_cols, "model_v3")
    res_baseline = evaluate_baseline(train_df, val_df)
    res_v2       = evaluate_v2(val_df)

    results = [res_v3, res_baseline]
    if res_v2:
        results.append(res_v2)

    print_comparison(results)

    # Detalhes v3
    print(f"\n── Detalhe model_v3 no val ──")
    print(f"  MAE goals_home : {res_v3['mae_home']:.4f}")
    print(f"  MAE goals_away : {res_v3['mae_away']:.4f}")
    print(f"  MAE total_goals: {res_v3['mae_total']:.4f}")
    print(f"  Acurácia resultado (H/D/A): {res_v3['res_acc']*100:.1f}%")
    print(f"  Acurácia over/under 2.5   : {res_v3['ou_acc']*100:.1f}%")

    # Detalhes por resultado
    val_df = val_df.copy()
    val_df["pred_home"] = res_v3["pred_h"]
    val_df["pred_away"] = res_v3["pred_a"]
    val_df["pred_result"] = derive_result(res_v3["pred_h"], res_v3["pred_a"])
    val_df["true_result"] = derive_result(val_df["goals_home"].values, val_df["goals_away"].values)
    val_df["correct"]     = val_df["pred_result"] == val_df["true_result"]

    print(f"\n── Acurácia por resultado (val) ──")
    for res in ["H", "D", "A"]:
        mask = val_df["true_result"] == res
        acc  = val_df.loc[mask, "correct"].mean()
        n    = mask.sum()
        print(f"  {res}: {n:>4} jogos  acc={acc*100:.1f}%")

    # Over/under por torneio (top 5 torneios no val)
    top_tourns = val_df["tournament"].value_counts().head(5).index
    print(f"\n── O/U 2.5 por torneio (top 5 no val) ──")
    for t in top_tourns:
        mask = val_df["tournament"] == t
        sub  = val_df[mask]
        ou_a = over_under_accuracy(
            sub["goals_home"], sub["goals_away"],
            sub["pred_home"].values, sub["pred_away"].values,
        )
        n = mask.sum()
        print(f"  {t[:40]:<40} n={n:>4}  acc={ou_a*100:.1f}%")

    print_pred_dist(res_v3, val_df, "model_v3")

    # 5. Feature importance
    print(f"\n[5] Feature importance...")
    print_importance(model_h, feat_cols, "goals_home")
    print_importance(model_a, feat_cols, "goals_away")

    # 6. Salva
    print(f"\n[6] Salvando modelos...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_h, OUT_HOME)
    joblib.dump(model_a, OUT_AWAY)
    import shutil
    shutil.copy(FEAT_JSON, OUT_JSON)
    print(f"  {OUT_HOME}")
    print(f"  {OUT_AWAY}")
    print(f"  {OUT_JSON}  ({len(feat_cols)} features)")

    print(f"\n{SEP}")
    print("  CONCLUÍDO")
    print(SEP)


if __name__ == "__main__":
    main()
