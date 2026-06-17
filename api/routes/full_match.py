"""full_match.py — Endpoint HTTP com a predição completa de uma partida
(todos os mercados: 1X2, over/under, BTTS, placar exato, handicap asiático).

Reaproveita predict_full_match() de src/models/predict_v3.py — mesma
lógica usada por scripts/predict_match_cli.py, sem duplicação.
"""

from fastapi import APIRouter, HTTPException, Query

from src.models.predict_v3 import load_full_match_context, predict_full_match

router = APIRouter(prefix="/predict", tags=["predict"])


def load_models() -> None:
    """Chamado no startup da API (lifespan) para pré-carregar o contexto
    histórico (Elo, FIFA ranking, resultados, EAFC26) uma única vez."""
    load_full_match_context()


@router.get("/full-match", summary="Predição completa de uma partida (todos os mercados)")
def full_match(
    home: str = Query(..., description="Time da casa — português (Supabase) ou inglês"),
    away: str = Query(..., description="Time de fora — português (Supabase) ou inglês"),
    neutral: bool = Query(True, description="Jogo em campo neutro (default: True, padrão Copa 2026)"),
    stage: str = Query("group", description="Fase do torneio — apenas informativo, não afeta a predição"),
) -> dict:
    if home.strip().lower() == away.strip().lower():
        raise HTTPException(status_code=400, detail="home e away devem ser times diferentes.")

    try:
        result = predict_full_match(home, away, is_neutral=neutral)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result["stage"] = stage
    return result
