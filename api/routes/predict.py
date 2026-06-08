import json
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["predict"])

_BASE        = Path("src/models/saved")
_DATA        = Path("data/processed/team_features.csv")
_COPA_GROUPS = Path("data/copa2026_groups.json")
_FIFA_PATH   = Path("data/fifa_rankings.json")

_result_model  = None
_goals_model   = None
_feature_cols: list[str] = []
_team_features: pd.DataFrame = pd.DataFrame()
_median_stats:  pd.Series | None = None
_fifa_rankings: dict[str, int] = {}

# Colunas ofensivas (mais alto = melhor → escalar para baixo em times fracos)
_OFFENSIVE = {"xg_for", "goals_for", "shots", "sot", "passes", "pressures", "win_rate"}
# Colunas defensivas (mais alto = pior → escalar para cima em times fracos)
_DEFENSIVE = {"xg_against", "goals_against", "opp_shots"}


def load_models() -> None:
    global _result_model, _goals_model, _feature_cols, _team_features, _median_stats, _fifa_rankings

    _result_model = joblib.load(_BASE / "result_model.pkl")
    _goals_model  = joblib.load(_BASE / "goals_model.pkl")

    with open(_BASE / "feature_columns.json") as f:
        _feature_cols = json.load(f)

    _team_features = pd.read_csv(_DATA).set_index("team")
    _median_stats  = _team_features.median()

    if _FIFA_PATH.exists():
        with open(_FIFA_PATH) as f:
            _fifa_rankings = json.load(f)


def _ranking_scale(ranking: int) -> float:
    """Converte ranking FIFA em fator de escala (0-1). Pior ranking = fator menor."""
    if ranking <= 20:  return 0.90
    if ranking <= 50:  return 0.75
    if ranking <= 80:  return 0.55
    return 0.35


def _confidence(ranking: int) -> str:
    if ranking <= 50: return "medium"
    return "low"


def _get_team(name: str) -> pd.Series:
    """Retorna stats do time; aplica fallback escalonado por ranking FIFA para times sem dados."""
    if name in _team_features.index:
        return _team_features.loc[name]

    ranking = _fifa_rankings.get(name, 100)
    scale   = _ranking_scale(ranking)

    fb = _median_stats.copy()
    for col in fb.index:
        if col in _OFFENSIVE:
            fb[col] *= scale          # times fracos atacam menos
        elif col in _DEFENSIVE:
            fb[col] /= scale          # times fracos sofrem mais
        elif col == "n_matches":
            fb[col] = 0
        elif col == "draw_rate":
            pass                      # draw_rate mantém a mediana
        elif col == "win_rate":
            fb[col] *= scale
        elif col == "loss_rate":
            fb[col] = max(0.0, 1 - fb["win_rate"] - fb["draw_rate"])

    # recalcula diferencial
    fb["xg_diff"] = fb["xg_for"] - fb["xg_against"]
    return fb


def _get_team_meta(name: str) -> dict[str, Any]:
    has_data = name in _team_features.index
    ranking  = _fifa_rankings.get(name, None) if not has_data else None
    return {
        "has_data":    has_data,
        "fifa_ranking": ranking,
        "confidence":  "high" if has_data else _confidence(ranking or 100),
    }


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home_team: str
    away_team: str


class GoalsBreakdown(BaseModel):
    total: float
    home:  float
    away:  float


class PredictResponse(BaseModel):
    home_team:          str
    away_team:          str
    probabilities:      dict[str, float]
    expected_goals:     GoalsBreakdown
    most_likely_result: str
    home_team_stats:    dict[str, Any]
    away_team_stats:    dict[str, Any]


class BatchRequest(BaseModel):
    matches: list[PredictRequest]


class TeamStanding(BaseModel):
    team:         str
    pts:          int
    w:            int
    d:            int
    l:            int
    gf:           float
    ga:           float
    gd:           float
    advances:     bool
    has_data:     bool
    confidence:   str
    fifa_ranking: int | None


class GroupResult(BaseModel):
    group:     str
    standings: list[TeamStanding]
    matches:   list[dict[str, Any]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_feature_vector(home: pd.Series, away: pd.Series) -> np.ndarray:
    row: dict[str, float] = {
        "diff_xg_for":        home["xg_for"]      - away["xg_for"],
        "diff_xg_against":    home["xg_against"]  - away["xg_against"],
        "diff_xg_net":        home["xg_diff"]     - away["xg_diff"],
        "diff_win_rate":      home["win_rate"]     - away["win_rate"],
        "diff_goals_for":     home["goals_for"]   - away["goals_for"],
        "diff_shots":         home["shots"]        - away["shots"],
        "diff_pressures":     home["pressures"]   - away["pressures"],
        "home_xg_for":        home["xg_for"],
        "home_xg_against":    home["xg_against"],
        "home_xg_diff":       home["xg_diff"],
        "home_goals_for":     home["goals_for"],
        "home_goals_against": home["goals_against"],
        "home_win_rate":      home["win_rate"],
        "home_shots":         home["shots"],
        "home_sot":           home["sot"],
        "home_pressures":     home["pressures"],
        "away_xg_for":        away["xg_for"],
        "away_xg_against":    away["xg_against"],
        "away_xg_diff":       away["xg_diff"],
        "away_goals_for":     away["goals_for"],
        "away_goals_against": away["goals_against"],
        "away_win_rate":      away["win_rate"],
        "away_shots":         away["shots"],
        "away_sot":           away["sot"],
        "away_pressures":     away["pressures"],
    }
    return np.array([[row[col] for col in _feature_cols]])


def _predict_one(home_name: str, away_name: str) -> dict[str, Any]:
    home = _get_team(home_name)
    away = _get_team(away_name)
    X    = _build_feature_vector(home, away)

    probs      = _result_model.predict_proba(X)[0]
    result_idx = int(np.argmax(probs))
    labels     = ["away_win", "draw", "home_win"]

    total_goals = float(max(0.0, _goals_model.predict(X)[0]))
    xg_sum      = home["xg_for"] + away["xg_for"]
    home_share  = home["xg_for"] / xg_sum if xg_sum > 0 else 0.5

    return {
        "home_team":  home_name,
        "away_team":  away_name,
        "home_meta":  _get_team_meta(home_name),
        "away_meta":  _get_team_meta(away_name),
        "probabilities": {
            "home_win": round(float(probs[2]), 4),
            "draw":     round(float(probs[1]), 4),
            "away_win": round(float(probs[0]), 4),
        },
        "expected_goals": {
            "total": round(total_goals, 2),
            "home":  round(total_goals * home_share, 2),
            "away":  round(total_goals * (1 - home_share), 2),
        },
        "most_likely_result": labels[result_idx],
    }


def _simulate_group(group_name: str, teams: list[str]) -> GroupResult:
    matches = []
    stats: dict[str, list[float]] = {t: [0, 0, 0, 0, 0.0, 0.0] for t in teams}

    for home_name, away_name in combinations(teams, 2):
        pred   = _predict_one(home_name, away_name)
        h_gols = pred["expected_goals"]["home"]
        a_gols = pred["expected_goals"]["away"]
        result = pred["most_likely_result"]

        if result == "home_win":
            stats[home_name][0] += 3; stats[home_name][1] += 1
            stats[away_name][3] += 1
        elif result == "draw":
            stats[home_name][0] += 1; stats[home_name][2] += 1
            stats[away_name][0] += 1; stats[away_name][2] += 1
        else:
            stats[away_name][0] += 3; stats[away_name][1] += 1
            stats[home_name][3] += 1

        stats[home_name][4] += h_gols; stats[home_name][5] += a_gols
        stats[away_name][4] += a_gols; stats[away_name][5] += h_gols

        matches.append({
            "home": home_name,
            "away": away_name,
            "most_likely": result,
            "expected_goals": pred["expected_goals"],
            "probabilities":  pred["probabilities"],
            "home_meta": pred["home_meta"],
            "away_meta": pred["away_meta"],
        })

    ranking = sorted(
        teams,
        key=lambda t: (stats[t][0], stats[t][4] - stats[t][5], stats[t][4]),
        reverse=True,
    )

    standings = []
    for i, team in enumerate(ranking):
        s    = stats[team]
        meta = _get_team_meta(team)
        standings.append(TeamStanding(
            team=team, pts=int(s[0]),
            w=int(s[1]), d=int(s[2]), l=int(s[3]),
            gf=round(s[4], 1), ga=round(s[5], 1),
            gd=round(s[4] - s[5], 1),
            advances=(i < 2),
            has_data=meta["has_data"],
            confidence=meta["confidence"],
            fifa_ranking=meta["fifa_ranking"],
        ))

    return GroupResult(group=group_name, standings=standings, matches=matches)


def _team_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "n_matches":     int(row["n_matches"]),
        "xg_for":        round(float(row["xg_for"]), 3),
        "xg_against":    round(float(row["xg_against"]), 3),
        "xg_diff":       round(float(row["xg_diff"]), 3),
        "goals_for":     round(float(row["goals_for"]), 3),
        "goals_against": round(float(row["goals_against"]), 3),
        "win_rate":      round(float(row["win_rate"]), 3),
        "draw_rate":     round(float(row["draw_rate"]), 3),
        "shots":         round(float(row["shots"]), 1),
        "pressures":     round(float(row["pressures"]), 1),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/teams", summary="Lista times com dados historicos disponíveis")
def list_teams() -> dict[str, Any]:
    return {
        "count": len(_team_features),
        "teams": sorted(_team_features.index.tolist()),
        "note":  "Times sem dados usam fallback escalonado por ranking FIFA.",
    }


@router.post("", response_model=PredictResponse, summary="Prediz uma partida")
def predict(req: PredictRequest) -> PredictResponse:
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="home_team e away_team devem ser diferentes.")

    home = _get_team(req.home_team)
    away = _get_team(req.away_team)
    X    = _build_feature_vector(home, away)

    probs      = _result_model.predict_proba(X)[0]
    result_idx = int(np.argmax(probs))
    labels     = ["away_win", "draw", "home_win"]

    total_goals = float(max(0.0, _goals_model.predict(X)[0]))
    xg_sum      = home["xg_for"] + away["xg_for"]
    home_share  = home["xg_for"] / xg_sum if xg_sum > 0 else 0.5

    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        probabilities={
            "home_win": round(float(probs[2]), 4),
            "draw":     round(float(probs[1]), 4),
            "away_win": round(float(probs[0]), 4),
        },
        expected_goals=GoalsBreakdown(
            total=round(total_goals, 2),
            home=round(total_goals * home_share, 2),
            away=round(total_goals * (1 - home_share), 2),
        ),
        most_likely_result=labels[result_idx],
        home_team_stats=_team_summary(home),
        away_team_stats=_team_summary(away),
    )


@router.post("/batch", summary="Prediz múltiplas partidas de uma vez")
def predict_batch(req: BatchRequest) -> list[dict[str, Any]]:
    if len(req.matches) > 100:
        raise HTTPException(status_code=400, detail="Maximo de 100 partidas por request.")
    return [_predict_one(m.home_team, m.away_team) for m in req.matches]


@router.post("/simulate-group", summary="Simula fase de grupos: round-robin + classificação")
def simulate_group(payload: dict[str, list[str]]) -> list[GroupResult]:
    if not payload:
        raise HTTPException(status_code=400, detail="Envie ao menos um grupo.")
    results = []
    for group_name, teams in payload.items():
        if len(teams) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"Grupo {group_name} deve ter exatamente 4 times.",
            )
        results.append(_simulate_group(group_name, teams))
    return results


@router.get("/copa2026", summary="Simula toda a fase de grupos da Copa 2026")
def copa2026_groups() -> Any:
    if not _COPA_GROUPS.exists():
        raise HTTPException(
            status_code=404,
            detail="Arquivo data/copa2026_groups.json nao encontrado.",
        )
    with open(_COPA_GROUPS) as f:
        groups = json.load(f)

    results = []
    for group_name, teams in groups.items():
        results.append(_simulate_group(group_name, teams))

    # top 8 terceiros colocados avancam (regra Copa 2026)
    third_place = [
        {"group": r.group, **r.standings[2].model_dump()}
        for r in results
    ]
    third_place.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    advancing_thirds = {t["team"] for t in third_place[:8]}

    for r in results:
        if r.standings[2].team in advancing_thirds:
            r.standings[2].advances = True

    return {
        "groups":           [r.model_dump() for r in results],
        "advancing_thirds": list(advancing_thirds),
    }
