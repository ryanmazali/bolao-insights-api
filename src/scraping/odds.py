import json
import os
from pathlib import Path

import requests

BASE_URL = "https://api.the-odds-api.com/v4"

_MAPPING_PATH = Path(__file__).parent.parent.parent / "data" / "team_name_mapping.json"
with open(_MAPPING_PATH, encoding="utf-8") as _f:
    _TEAM_NAME_MAPPING: dict[str, str] = json.load(_f)


def normalize_team_name(name: str) -> str:
    """Normaliza nome do time para o padrão StatsBomb."""
    return _TEAM_NAME_MAPPING.get(name, name)


def get_world_cup_odds() -> list:
    """Busca odds da Copa do Mundo 2026."""
    api_key = os.getenv("ODDS_API_KEY")
    url = f"{BASE_URL}/sports/soccer_fifa_world_cup/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    r = requests.get(url, params=params, timeout=15)

    if r.status_code == 200:
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"[Odds] Requisicoes restantes: {remaining}")
        return r.json()
    else:
        print(f"[Odds] Erro {r.status_code}: {r.text[:200]}")
        return []


def parse_odds(raw_odds: list) -> list:
    """Processa odds brutas e extrai 1X2 e over/under 2.5."""
    parsed = []

    for game in raw_odds:
        home_team = normalize_team_name(game.get("home_team"))
        away_team = normalize_team_name(game.get("away_team"))
        commence_time = game.get("commence_time")

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "h2h": None,
            "totals": None,
            "btts": None,
            "bookmakers_count": len(game.get("bookmakers", [])),
        }

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):

                if market["key"] == "h2h" and result["h2h"] is None:
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    # outcomes usa nomes originais da API, mas nossos campos usam nomes normalizados
                    raw_home = game.get("home_team")
                    raw_away = game.get("away_team")
                    result["h2h"] = {
                        "home_win": outcomes.get(raw_home),
                        "away_win": outcomes.get(raw_away),
                        "draw": outcomes.get("Draw"),
                        "bookmaker": bookmaker["title"],
                    }

                if market["key"] == "totals" and result["totals"] is None:
                    for outcome in market["outcomes"]:
                        if outcome.get("point") == 2.5:
                            if result["totals"] is None:
                                result["totals"] = {}
                            result["totals"][outcome["name"].lower()] = outcome["price"]
                    if result["totals"]:
                        result["totals"]["bookmaker"] = bookmaker["title"]

                if market["key"] == "btts" and result["btts"] is None:
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    if "Yes" in outcomes or "No" in outcomes:
                        result["btts"] = {
                            "yes": outcomes.get("Yes"),
                            "no": outcomes.get("No"),
                            "bookmaker": bookmaker["title"],
                        }

        parsed.append(result)

    return parsed


def calculate_value_bets(parsed_odds: list, predictions: dict) -> list:
    """
    Cruza odds do mercado com probabilidades do modelo.
    Retorna value bets onde model_prob > probabilidade implícita da odd.
    Threshold: 4% de valor mínimo.
    """
    value_bets = []

    for game in parsed_odds:
        home = game["home_team"]  # já normalizado pelo parse_odds
        away = game["away_team"]

        pred_key = f"{home}_vs_{away}"
        if pred_key not in predictions:
            continue

        pred = predictions[pred_key]
        h2h = game.get("h2h")
        totals = game.get("totals")
        btts_odds = game.get("btts")
        markets = pred.get("markets", {})

        # ── 1X2 ──────────────────────────────────────────────────────
        if h2h:
            for outcome, label, model_prob, odd in [
                ("home_win", f"{home} vence", pred["probabilities"]["home_win"], h2h.get("home_win")),
                ("draw",     "Empate",         pred["probabilities"]["draw"],     h2h.get("draw")),
                ("away_win", f"{away} vence",  pred["probabilities"]["away_win"], h2h.get("away_win")),
            ]:
                if not odd:
                    continue

                implied = 1 / odd
                value = model_prob - implied

                if value > 0.04:
                    value_bets.append({
                        "match": f"{home} vs {away}",
                        "market": "1X2",
                        "outcome": label,
                        "model_prob_pct": round(model_prob * 100, 1),
                        "implied_prob_pct": round(implied * 100, 1),
                        "odd": odd,
                        "value_pct": round(value * 100, 1),
                        "value_rating": "alto" if value > 0.10 else "medio",
                        "bookmaker": h2h.get("bookmaker"),
                    })

        # ── Over/Under 2.5 ───────────────────────────────────────────
        if totals and markets.get("over_under_25"):
            ou = markets["over_under_25"]
            for side, label, model_prob, odd in [
                ("over",  "Over 2.5",  ou["over"],  totals.get("over")),
                ("under", "Under 2.5", ou["under"], totals.get("under")),
            ]:
                if not odd:
                    continue

                implied = 1 / odd
                value = model_prob - implied

                if value > 0.04:
                    value_bets.append({
                        "match": f"{home} vs {away}",
                        "market": "Over/Under 2.5",
                        "outcome": label,
                        "model_prob_pct": round(model_prob * 100, 1),
                        "implied_prob_pct": round(implied * 100, 1),
                        "odd": odd,
                        "value_pct": round(value * 100, 1),
                        "value_rating": "alto" if value > 0.10 else "medio",
                        "bookmaker": totals.get("bookmaker"),
                    })

        # ── BTTS (ambas marcam) ──────────────────────────────────────
        if btts_odds and markets.get("btts"):
            btts = markets["btts"]
            for side, label, model_prob, odd in [
                ("yes", "Ambas marcam - Sim", btts["yes"], btts_odds.get("yes")),
                ("no",  "Ambas marcam - Não", btts["no"],  btts_odds.get("no")),
            ]:
                if not odd:
                    continue

                implied = 1 / odd
                value = model_prob - implied

                if value > 0.04:
                    value_bets.append({
                        "match": f"{home} vs {away}",
                        "market": "BTTS",
                        "outcome": label,
                        "model_prob_pct": round(model_prob * 100, 1),
                        "implied_prob_pct": round(implied * 100, 1),
                        "odd": odd,
                        "value_pct": round(value * 100, 1),
                        "value_rating": "alto" if value > 0.10 else "medio",
                        "bookmaker": btts_odds.get("bookmaker"),
                    })

    value_bets.sort(key=lambda x: x["value_pct"], reverse=True)
    return value_bets
