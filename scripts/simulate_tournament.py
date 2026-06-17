"""simulate_tournament.py — Simulação Monte Carlo completa da Copa 2026,
do estado atual (20 jogos de grupo já encerrados) até a Final.

Pipeline por simulação (simulate_once()):
  1. Fase de grupos: os 20 jogos reais já entram fixos; os 52 restantes
     são sorteados a partir da matriz de placares (modelo v3 + Dixon-Coles).
  2. Classificação final dos 12 grupos com desempate oficial (pontos,
     confronto direto via mini-liga, saldo de gols, gols marcados,
     ranking FIFA como fallback — team conduct score não existe nos
     dados, mesma pendência documentada em bracket_resolver.py).
  3. 8 melhores terceiros + bracket_resolver.build_r32_bracket() monta
     os 16 jogos do R32.
  4. Mata-mata R32 → R16 → QF → SF → 3º lugar → Final, com pênaltis em
     caso de empate (probabilidade via logística no diff de Elo).

Performance: lambda_home/away só depende dos dados históricos reais
(congelados numa data de referência), nunca dos resultados sorteados
em rodadas anteriores da mesma simulação — então a matriz de placares
de cada confronto (home,away,is_neutral) é computada uma única vez e
cacheada, reaproveitada entre as 1000 simulações.

Uso:
    python scripts/simulate_tournament.py --n 10      # smoke test
    python scripts/simulate_tournament.py --n 1000     # rodada completa
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_features_v3 import (  # noqa: E402
    SUPABASE_PT_TO_EN,
    GLOBAL_AVG_ELO,
    load_elo_ratings,
    load_fifa_rankings,
    load_results,
    load_eafc_features,
    compute_team_features,
    compute_h2h,
    get_eafc,
    get_elo_at_date,
)
from src.models.predict_v3 import (  # noqa: E402
    build_score_matrix,
    _build_feature_row as build_feature_row,
)
from bracket_resolver import (  # noqa: E402
    build_r32_bracket,
    R32_FIXED_MATCHES,
    R32_DEPENDENT_LEFT,
)
from supabase import create_client  # noqa: E402

MODELS_DIR   = ROOT / "models"
MODEL_HOME   = MODELS_DIR / "model_goals_home_v3.pkl"
MODEL_AWAY   = MODELS_DIR / "model_goals_away_v3.pkl"
FEAT_JSON    = MODELS_DIR / "feature_columns_v3.json"
METRICS_JSON = MODELS_DIR / "metrics_v3.json"
EAFC_CSV     = ROOT / "data/processed/eafc26_team_features.csv"
OUT_JSON     = ROOT / "data/processed/tournament_simulation.json"
OUT_CSV      = ROOT / "data/processed/tournament_simulation.csv"

MAX_GOALS = 8
SNAPSHOT_DATE = pd.Timestamp("2026-06-18")  # após os 20 jogos reais já disputados
PENALTY_ELO_SCALE = 1000.0  # calibrado para diff_elo=200 -> ~55/45 (ver nota abaixo)

HOST_TEAMS = {"USA", "Mexico", "Canada"}
VENUE_COUNTRY_HINTS = {
    "Azteca": "Mexico", "Akron": "Mexico", "BBVA": "Mexico",
    "BMO Field": "Canada", "BC Place": "Canada",
}  # qualquer venue não listado aqui é tratado como Estados Unidos

# Round of 32 (73-88) -> Round of 16 (89-96) -> QF (97-100) -> SF (101-102)
# -> 3º lugar (103) / Final (104).  Fonte: data/raw/round_of_32_structure.md
KNOCKOUT_MAP = {
    "match89":  ("match74", "match77"),
    "match90":  ("match73", "match75"),
    "match91":  ("match76", "match78"),
    "match92":  ("match79", "match80"),
    "match93":  ("match83", "match84"),
    "match94":  ("match81", "match82"),
    "match95":  ("match86", "match88"),
    "match96":  ("match85", "match87"),
    "match97":  ("match89", "match90"),
    "match98":  ("match93", "match94"),
    "match99":  ("match91", "match92"),
    "match100": ("match95", "match96"),
    "match101": ("match97", "match98"),
    "match102": ("match99", "match100"),
}
R16_MATCHES   = ["match89", "match90", "match91", "match92", "match93", "match94", "match95", "match96"]
QF_MATCHES    = ["match97", "match98", "match99", "match100"]
SF_MATCHES    = ["match101", "match102"]


def venue_country(venue: str) -> str:
    for hint, country in VENUE_COUNTRY_HINTS.items():
        if hint in venue:
            return country
    return "United States"


# ── Carrega estado atual do Supabase ───────────────────────────────────────

def fetch_supabase_state():
    load_dotenv(dotenv_path=str(ROOT / ".env"))
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

    teams = {t["id"]: t["name"] for t in sb.table("teams").select("id, name").execute().data}
    standings = sb.table("group_standings").select("*").execute().data
    team_group = {s["team_id"]: s["world_cup_group"] for s in standings}

    matches = sb.table("matches").select("*").eq("stage", "group").execute().data
    match_results = {r["match_id"]: r for r in sb.table("match_results").select("*").execute().data}

    def to_en(team_id):
        pt = teams[team_id]
        return SUPABASE_PT_TO_EN.get(pt, pt)

    groups_en = {}
    for team_id, g in team_group.items():
        groups_en.setdefault(g, []).append(to_en(team_id))
    for g in groups_en:
        groups_en[g] = sorted(groups_en[g])

    finished, remaining = [], []
    for m in matches:
        home_en, away_en = to_en(m["home_team_id"]), to_en(m["away_team_id"])
        group = team_group[m["home_team_id"]]
        if m["status"] == "finished":
            res = match_results[m["id"]]
            finished.append({
                "group": group, "home": home_en, "away": away_en,
                "home_score": res["home_score"], "away_score": res["away_score"],
            })
        else:
            is_neutral = not (
                (home_en in HOST_TEAMS) and (venue_country(m["venue"]) == (
                    "United States" if home_en == "USA" else home_en
                ))
            )
            remaining.append({
                "group": group, "home": home_en, "away": away_en, "is_neutral": is_neutral,
            })

    # sanity check rápido: pontos recomputados dos 20 jogos reais vs group_standings
    standings_by_team_en = {to_en(s["team_id"]): s for s in standings}
    recomputed = {t: {"pts": 0, "gf": 0, "ga": 0, "played": 0} for g in groups_en.values() for t in g}
    for m in finished:
        h, a, sh, sa = m["home"], m["away"], m["home_score"], m["away_score"]
        recomputed[h]["gf"] += sh; recomputed[h]["ga"] += sa; recomputed[h]["played"] += 1
        recomputed[a]["gf"] += sa; recomputed[a]["ga"] += sh; recomputed[a]["played"] += 1
        if sh > sa: recomputed[h]["pts"] += 3
        elif sa > sh: recomputed[a]["pts"] += 3
        else: recomputed[h]["pts"] += 1; recomputed[a]["pts"] += 1
    mismatches = []
    for t, rec in recomputed.items():
        sup = standings_by_team_en[t]
        if rec["pts"] != sup["points"] or rec["played"] != sup["played"]:
            mismatches.append((t, rec, sup))
    if mismatches:
        print(f"  [AVISO] {len(mismatches)} divergências entre recomputado e group_standings:")
        for t, rec, sup in mismatches[:5]:
            print(f"    {t}: recomputado pts={rec['pts']} played={rec['played']}  "
                  f"supabase pts={sup['points']} played={sup['played']}")
    else:
        print(f"  [OK] group_standings consistente com os {len(finished)} jogos reais.")

    return groups_en, finished, remaining


# ── Pré-computação: features por time, cache de matrizes de placar ─────────

class Engine:
    def __init__(self):
        print("[Setup] Carregando dados históricos (Elo, FIFA, resultados, EAFC26)...")
        self.elo_df, self.current_elo = load_elo_ratings()
        self.rankings_df, self.current_rankings = load_fifa_rankings()
        self.df_results = load_results(self.elo_df, self.current_elo)
        self.eafc_lookup, self.eafc_def, self.former = load_eafc_features()

        with open(FEAT_JSON, encoding="utf-8") as f:
            self.feat_cols = json.load(f)
        with open(METRICS_JSON, encoding="utf-8") as f:
            metrics = json.load(f)
        self.rho = metrics["rho_dixon_coles"]

        self.model_h = joblib.load(MODEL_HOME)
        self.model_a = joblib.load(MODEL_AWAY)

        self._matrix_cache: dict[tuple, np.ndarray] = {}
        self._team_features: dict[str, dict] = {}
        self._team_eafc: dict[str, dict] = {}
        self._team_elo: dict[str, float] = {}
        self.n_cache_miss = 0
        self.n_cache_hit = 0

    def team_features(self, team: str) -> dict:
        if team not in self._team_features:
            self._team_features[team] = compute_team_features(
                self.df_results, team, SNAPSHOT_DATE, self.rankings_df, self.current_rankings
            )
        return self._team_features[team]

    def team_eafc(self, team: str) -> dict:
        if team not in self._team_eafc:
            eafc, _found = get_eafc(team, self.eafc_lookup, self.eafc_def, self.former)
            self._team_eafc[team] = eafc
        return self._team_eafc[team]

    def team_elo(self, team: str) -> float:
        if team not in self._team_elo:
            self._team_elo[team] = get_elo_at_date(self.elo_df, self.current_elo, team, SNAPSHOT_DATE)
        return self._team_elo[team]

    def fifa_rank_position(self) -> dict:
        teams = list(self._team_features.keys())
        pts = {t: self._team_features[t]["fifa_points"] for t in teams}
        ordered = sorted(teams, key=lambda t: -pts[t])
        return {t: i + 1 for i, t in enumerate(ordered)}

    def score_matrix(self, home: str, away: str, is_neutral: bool) -> np.ndarray:
        key = (home, away, is_neutral)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            self.n_cache_hit += 1
            return cached
        self.n_cache_miss += 1
        hf = self.team_features(home)
        af = self.team_features(away)
        h2h = compute_h2h(self.df_results, home, away, SNAPSHOT_DATE)
        row = build_feature_row(hf, af, self.team_eafc(home), self.team_eafc(away), h2h, int(is_neutral))
        X = np.array([[row[c] for c in self.feat_cols]])
        lh = float(self.model_h.predict(X)[0])
        la = float(self.model_a.predict(X)[0])
        sm = build_score_matrix(lh, la, self.rho, MAX_GOALS)
        self._matrix_cache[key] = sm
        return sm


def sample_score(sm: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = sm.flatten()
    flat = flat / flat.sum()
    idx = rng.choice(len(flat), p=flat)
    n_cols = sm.shape[1]
    return int(idx // n_cols), int(idx % n_cols)


def penalty_winner(team_a: str, team_b: str, elo: dict, rng: np.random.Generator) -> str:
    diff = elo[team_a] - elo[team_b]
    p_a = 1.0 / (1.0 + np.exp(-diff / PENALTY_ELO_SCALE))
    return team_a if rng.random() < p_a else team_b


# ── Desempate dentro do grupo (pontos -> confronto direto -> SG -> GP -> FIFA) ─

def resolve_tie_block(block: list[str], team_stats: dict, group_matches: list[tuple],
                      fifa_rank: dict) -> list[str]:
    mini = {t: {"pts": 0, "gf": 0, "ga": 0} for t in block}
    for (a, b, sa, sb) in group_matches:
        if a in mini and b in mini:
            mini[a]["gf"] += sa; mini[a]["ga"] += sb
            mini[b]["gf"] += sb; mini[b]["ga"] += sa
            if sa > sb: mini[a]["pts"] += 3
            elif sb > sa: mini[b]["pts"] += 3
            else: mini[a]["pts"] += 1; mini[b]["pts"] += 1

    def key(t):
        overall_gd = team_stats[t]["gf"] - team_stats[t]["ga"]
        return (
            -mini[t]["pts"], -(mini[t]["gf"] - mini[t]["ga"]), -mini[t]["gf"],
            -overall_gd, -team_stats[t]["gf"], fifa_rank[t],
        )

    return sorted(block, key=key)


def rank_group(teams: list[str], team_stats: dict, group_matches: list[tuple],
              fifa_rank: dict) -> list[str]:
    order = sorted(teams, key=lambda t: -team_stats[t]["pts"])
    blocks: list[list[str]] = []
    for t in order:
        if blocks and team_stats[blocks[-1][-1]]["pts"] == team_stats[t]["pts"]:
            blocks[-1].append(t)
        else:
            blocks.append([t])
    final_order = []
    for block in blocks:
        final_order.extend(block if len(block) == 1 else
                           resolve_tie_block(block, team_stats, group_matches, fifa_rank))
    return final_order


# ── Uma simulação completa ─────────────────────────────────────────────────

def simulate_once(engine: Engine, groups_en: dict, finished: list[dict],
                  remaining: list[dict], fifa_rank: dict, rng: np.random.Generator) -> dict:
    all_teams = [t for g in groups_en.values() for t in g]
    team_stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in all_teams}
    group_matches: dict[str, list[tuple]] = {g: [] for g in groups_en}

    def record(group, home, away, sh, sa):
        team_stats[home]["gf"] += sh; team_stats[home]["ga"] += sa
        team_stats[away]["gf"] += sa; team_stats[away]["ga"] += sh
        if sh > sa: team_stats[home]["pts"] += 3
        elif sa > sh: team_stats[away]["pts"] += 3
        else: team_stats[home]["pts"] += 1; team_stats[away]["pts"] += 1
        group_matches[group].append((home, away, sh, sa))

    for m in finished:
        record(m["group"], m["home"], m["away"], m["home_score"], m["away_score"])

    for m in remaining:
        sm = engine.score_matrix(m["home"], m["away"], m["is_neutral"])
        sh, sa = sample_score(sm, rng)
        record(m["group"], m["home"], m["away"], sh, sa)

    group_standings = {}
    for g, teams in groups_en.items():
        ranked = rank_group(teams, team_stats, group_matches[g], fifa_rank)
        group_standings[g] = [
            {"team": t, "pts": team_stats[t]["pts"],
             "gd": team_stats[t]["gf"] - team_stats[t]["ga"],
             "gf": team_stats[t]["gf"], "fifa_ranking": fifa_rank[t]}
            for t in ranked
        ]

    bracket = build_r32_bracket(group_standings)
    r32 = bracket["r32_matches"]

    winners: dict[str, str] = {}
    losers: dict[str, str] = {}

    def play(match_id, home, away):
        sm = engine.score_matrix(home, away, True)  # mata-mata sempre neutro
        sh, sa = sample_score(sm, rng)
        if sh > sa:
            w, l = home, away
        elif sa > sh:
            w, l = away, home
        else:
            w = penalty_winner(home, away, engine._team_elo, rng)
            l = away if w == home else home
        winners[match_id] = w
        losers[match_id] = l

    for match_id, pair in r32.items():
        play(match_id, pair["home"], pair["away"])

    for match_id, (src_a, src_b) in KNOCKOUT_MAP.items():
        play(match_id, winners[src_a], winners[src_b])

    # 3º lugar e final
    play("match103", losers["match101"], losers["match102"])
    play("match104", winners["match101"], winners["match102"])

    champion = winners["match104"]
    runner_up = losers["match104"]
    third = winners["match103"]
    fourth = losers["match103"]

    reached_r32 = set(t for pair in r32.values() for t in (pair["home"], pair["away"]))
    reached_r16 = set(winners[m] for m in R16_MATCHES) | set(losers[m] for m in R16_MATCHES)
    reached_qf  = set(winners[m] for m in QF_MATCHES)  | set(losers[m] for m in QF_MATCHES)
    reached_sf  = set(winners[m] for m in SF_MATCHES)  | set(losers[m] for m in SF_MATCHES)
    reached_final = {champion, runner_up}

    return {
        "champion": champion, "runner_up": runner_up, "third": third, "fourth": fourth,
        "reached_r32": reached_r32, "reached_r16": reached_r16,
        "reached_qf": reached_qf, "reached_sf": reached_sf, "reached_final": reached_final,
    }


# ── Loop principal + agregação ──────────────────────────────────────────────

def run_simulations(n: int, seed: int = 42) -> tuple[list[dict], Engine, float]:
    print(f"\n[Run] Buscando estado atual no Supabase...")
    groups_en, finished, remaining = fetch_supabase_state()
    print(f"  {sum(len(g) for g in groups_en.values())} times em {len(groups_en)} grupos")
    print(f"  {len(finished)} jogos de grupo já encerrados, {len(remaining)} restantes")

    engine = Engine()
    for t in [x for g in groups_en.values() for x in g]:
        engine.team_features(t)
        engine.team_eafc(t)
        engine.team_elo(t)
    fifa_rank = engine.fifa_rank_position()

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    results = [simulate_once(engine, groups_en, finished, remaining, fifa_rank, rng)
              for _ in range(n)]
    elapsed = time.perf_counter() - t0
    return results, engine, elapsed


def aggregate(results: list[dict], all_teams: list[str]) -> pd.DataFrame:
    n = len(results)
    counts = {t: dict(champion=0, final=0, semis=0, quarters=0, r16=0, r32=0) for t in all_teams}
    for r in results:
        counts[r["champion"]]["champion"] += 1
        for t in r["reached_final"]: counts[t]["final"] += 1
        for t in r["reached_sf"]:    counts[t]["semis"] += 1
        for t in r["reached_qf"]:    counts[t]["quarters"] += 1
        for t in r["reached_r16"]:   counts[t]["r16"] += 1
        for t in r["reached_r32"]:   counts[t]["r32"] += 1

    rows = []
    for t in all_teams:
        c = counts[t]
        rows.append({
            "team": t,
            "pct_champion": 100 * c["champion"] / n,
            "pct_final":    100 * c["final"] / n,
            "pct_semis":    100 * c["semis"] / n,
            "pct_quarters": 100 * c["quarters"] / n,
            "pct_r16":      100 * c["r16"] / n,
            "pct_groups_exit": 100 * (1 - c["r32"] / n),
        })
    df = pd.DataFrame(rows).sort_values("pct_champion", ascending=False).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results, engine, elapsed = run_simulations(args.n, args.seed)
    all_teams = sorted(set(r["champion"] for r in results) |
                       set().union(*[r["reached_r32"] for r in results]))
    # garante os 48 times completos mesmo que algum nunca tenha passado de grupo
    all_teams = sorted(engine._team_features.keys())

    df = aggregate(results, all_teams)

    print(f"\n{'='*78}\n  RESULTADO — {args.n} simulações\n{'='*78}")
    print(f"  Tempo total: {elapsed:.2f}s  |  Tempo médio/simulação: {1000*elapsed/args.n:.2f}ms")
    print(f"  Matrizes de placar: {engine.n_cache_miss} computadas, "
          f"{engine.n_cache_hit} reaproveitadas do cache "
          f"({100*engine.n_cache_hit/max(1, engine.n_cache_hit+engine.n_cache_miss):.1f}% hit rate)")

    print(f"\n  Top 15 por % de título:")
    print(f"  {'Time':<20}{'Título':>10}{'Final':>10}{'Semis':>10}{'Quartas':>10}{'R16':>10}{'Sai grupos':>12}")
    for _, r in df.head(15).iterrows():
        print(f"  {r['team']:<20}{r['pct_champion']:>9.1f}%{r['pct_final']:>9.1f}%"
              f"{r['pct_semis']:>9.1f}%{r['pct_quarters']:>9.1f}%{r['pct_r16']:>9.1f}%"
              f"{r['pct_groups_exit']:>11.1f}%")

    total_champion_pct = df["pct_champion"].sum()
    print(f"\n  Sanity check: soma de pct_champion (48 seleções) = {total_champion_pct:.2f}% "
          f"(esperado ~100%)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_simulations": args.n,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "cache_misses": engine.n_cache_miss,
        "cache_hits": engine.n_cache_hit,
        "sanity_check_champion_pct_sum": total_champion_pct,
        "teams": df.to_dict(orient="records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Salvo: {OUT_JSON}")
    print(f"  Salvo: {OUT_CSV}")


if __name__ == "__main__":
    main()
