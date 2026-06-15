"""Endpoint de projeção de métricas táticas por jogador.

Não modifica nenhum endpoint existente — router próprio, registrado
separadamente em api/main.py junto com o router de /predict.
"""

import json
import logging
from math import exp
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["player-metrics"])

_MODELS_DIR = Path("data/models")
_PLAYER_METRICS_PATH = _MODELS_DIR / "player_metrics_data.json"
_TEAM_METRICS_PATH = _MODELS_DIR / "team_metrics_data.json"
_METADATA_PATH = _MODELS_DIR / "metrics_metadata.json"

_player_metrics_data: dict[str, Any] | None = None
_team_metrics_data: dict[str, Any] | None = None
_metrics_metadata: dict[str, Any] | None = None
_global_avg_shots_p90: float = 0.0

STAGE_FACTORS: dict[str, float] = {
    "group": 1.0,
    "round_of_32": 1.05,
    "quarterfinal": 1.08,
    "semifinal": 1.10,
    "final": 1.12,
}


def load_player_metrics() -> None:
    """Carrega os artefatos do pipeline de métricas táticas (data/models/).

    Chamado uma vez no startup da app (api/main.py). Se algum arquivo não
    existir, loga um warning e deixa o respectivo cache em None — o
    endpoint POST /predict/player-metrics responde 503 nesse caso, sem
    derrubar a aplicação.
    """
    global _player_metrics_data, _team_metrics_data, _metrics_metadata, _global_avg_shots_p90

    if _PLAYER_METRICS_PATH.exists():
        with open(_PLAYER_METRICS_PATH, encoding="utf-8") as f:
            _player_metrics_data = json.load(f)
    else:
        logger.warning("Arquivo não encontrado: %s — /predict/player-metrics retornará 503.", _PLAYER_METRICS_PATH)

    if _TEAM_METRICS_PATH.exists():
        with open(_TEAM_METRICS_PATH, encoding="utf-8") as f:
            _team_metrics_data = json.load(f)
    else:
        logger.warning("Arquivo não encontrado: %s — /predict/player-metrics retornará 503.", _TEAM_METRICS_PATH)

    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, encoding="utf-8") as f:
            _metrics_metadata = json.load(f)
            _global_avg_shots_p90 = _metrics_metadata.get("global_avg_shots_p90", 0.0)
    else:
        logger.warning("Arquivo não encontrado: %s — /predict/player-metrics retornará 503.", _METADATA_PATH)


def _confidence_for(source: str, sample_size: str | None) -> str:
    if source == "fbref":
        return {"large": "high", "medium": "medium", "small": "medium"}.get(sample_size, "medium")
    if source.startswith("fbref_median"):
        return "medium"
    if source == "fm23":
        return "medium"
    if source == "fm23_median":
        return "low"
    return "low"


# ── Schemas ───────────────────────────────────────────────────────────────────

class PlayerMetricsRequest(BaseModel):
    home_team: str
    away_team: str
    home_players: list[str]
    away_players: list[str]
    stage: Literal["group", "round_of_32", "quarterfinal", "semifinal", "final"] = "group"


class PlayerProjection(BaseModel):
    player_id: str
    name: str
    pos: str
    shots_expected: float
    sot_expected: float
    xg_expected: float
    prob_shot: float
    tackles_expected: float
    fouls_expected: float
    recoveries_expected: float
    source: str
    confidence: str


class PlayerMetricsResponse(BaseModel):
    home_team: str
    away_team: str
    stage: str
    metadata: dict[str, Any]
    home_players: list[PlayerProjection]
    away_players: list[PlayerProjection]


# ── Lógica de cálculo ────────────────────────────────────────────────────────

def _not_found_projection(player_id: str) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id,
        name="unknown",
        pos="unknown",
        shots_expected=0.0,
        sot_expected=0.0,
        xg_expected=0.0,
        prob_shot=0.0,
        tackles_expected=0.0,
        fouls_expected=0.0,
        recoveries_expected=0.0,
        source="not_found",
        confidence="none",
    )


def _project_side(
    player_ids: list[str], team_name: str, opponent_name: str, stage_factor: float,
) -> list[PlayerProjection]:
    team_data = _team_metrics_data.get(team_name)
    opp_data = _team_metrics_data.get(opponent_name)

    defensive_factor = opp_data.get("defensive_factor", 1.0) if opp_data else 1.0
    team_shots_p90 = team_data.get("shots_p90", _global_avg_shots_p90) if team_data else _global_avg_shots_p90
    team_shots_expected = team_shots_p90 * defensive_factor

    players_data = [_player_metrics_data.get(pid) for pid in player_ids]
    sum_sh = sum((p.get("Sh_p90") or 0) for p in players_data if p is not None)

    projections = []
    for pid, data in zip(player_ids, players_data):
        if data is None:
            projections.append(_not_found_projection(pid))
            continue

        sh = data.get("Sh_p90") or 0
        sot = data.get("SoT_p90") or 0
        xg = data.get("xG_p90") or 0
        tkl = data.get("Tkl_p90") or 0
        fls = data.get("Fls_p90") or 0
        recov = data.get("Recov_p90") or 0

        share = sh / sum_sh if sum_sh > 0 else 0
        shots_expected = share * team_shots_expected

        if sh == 0:
            sot_expected = 0.0
            xg_expected = 0.0
        else:
            sot_expected = shots_expected * (sot / sh)
            xg_expected = shots_expected * (xg / sh)

        tackles_expected = tkl * stage_factor
        fouls_expected = fls * stage_factor
        recoveries_expected = recov * stage_factor

        prob_shot = 1 - exp(-shots_expected)

        projections.append(PlayerProjection(
            player_id=pid,
            name=data["name"],
            pos=data["position"],
            shots_expected=round(shots_expected, 2),
            sot_expected=round(sot_expected, 2),
            xg_expected=round(xg_expected, 3),
            prob_shot=round(prob_shot, 2),
            tackles_expected=round(tackles_expected, 2),
            fouls_expected=round(fouls_expected, 2),
            recoveries_expected=round(recoveries_expected, 2),
            source=data["source"],
            confidence=_confidence_for(data["source"], data.get("sample_size")),
        ))

    return projections


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/player-metrics",
    response_model=PlayerMetricsResponse,
    summary="Projeção de métricas táticas por jogador (chutes, SoT, xG, desarmes, faltas, recuperações)",
)
def player_metrics(req: PlayerMetricsRequest) -> PlayerMetricsResponse:
    """Projeta métricas táticas esperadas para os titulares de uma partida.

    Fontes de dados (geradas por scripts/build_player_metrics_model.py):
      - StatsBomb (Euro 2024, Copa America 2024, AFCON 2023): stats de
        seleção (shots_p90, defensive_factor, etc.) em team_metrics_data.json.
      - FBref (top 5 ligas europeias, 2024/25): métricas individuais p90
        (Sh_p90, SoT_p90, xG_p90, Tkl_p90, Fls_p90, Recov_p90, ...).
      - FM23: atributos convertidos em estimativas de métricas para
        jogadores sem amostra relevante no FBref.

    Hierarquia de fallback de cada jogador (campo "source"):
      1. "fbref"               — métricas individuais reais do FBref.
      2. "fbref_median_<liga>" — sem amostra no FBref, mas joga em uma das
                                  5 grandes ligas (via clube no FM23);
                                  usa a mediana da liga/posição.
      3. "fm23"                — sem FBref nem liga conhecida; estimativa
                                  a partir dos atributos FM23 do jogador.
      4. "fm23_median"         — sem nenhum dado individual; estimativa a
                                  partir da mediana FM23 por posição.
      5. "not_found"           — player_id não existe em player_metrics_data
                                  (request não falha; retorna tudo zerado).

    Significado de "confidence":
      - "high"   — fbref individual com amostra grande (>= 900 min).
      - "medium" — fbref individual com amostra média/pequena,
                    fbref_median_<liga> ou estimativa fm23 individual.
      - "low"    — fm23_median (mediana por posição, menor confiabilidade).
      - "none"   — player_id não encontrado (source="not_found").

    stage_factor (group=1.0, round_of_32=1.05, quarterfinal=1.08,
    semifinal=1.10, final=1.12) representa a maior intensidade de jogos
    eliminatórios. Afeta APENAS as métricas físicas/defensivas do próprio
    jogador (tackles_expected, fouls_expected, recoveries_expected).
    shots_expected, sot_expected e xg_expected dependem apenas do volume
    ofensivo do time e do defensive_factor do adversário — não são
    afetados pela fase da competição.
    """
    if _player_metrics_data is None or _team_metrics_data is None or _metrics_metadata is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Artefatos de métricas táticas indisponíveis "
                "(data/models/player_metrics_data.json, team_metrics_data.json "
                "e/ou metrics_metadata.json não encontrados). "
                "Rode o pipeline scripts/build_player_metrics_model.py."
            ),
        )

    stage_factor = STAGE_FACTORS[req.stage]

    home_projections = _project_side(req.home_players, req.home_team, req.away_team, stage_factor)
    away_projections = _project_side(req.away_players, req.away_team, req.home_team, stage_factor)

    metadata = {
        "generated_at": _metrics_metadata.get("generated_at"),
        "home_team_statsbomb": req.home_team in _team_metrics_data,
        "away_team_statsbomb": req.away_team in _team_metrics_data,
        "global_avg_shots_p90": _global_avg_shots_p90,
    }

    return PlayerMetricsResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        stage=req.stage,
        metadata=metadata,
        home_players=home_projections,
        away_players=away_projections,
    )
