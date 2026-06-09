import os
import requests

BASE_URL = "https://api.the-odds-api.com/v4"


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
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "h2h": None,
            "totals": None,
            "bookmakers_count": len(game.get("bookmakers", [])),
        }

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):

                if market["key"] == "h2h" and result["h2h"] is None:
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    result["h2h"] = {
                        "home_win": outcomes.get(home_team),
                        "away_win": outcomes.get(away_team),
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
        home = game["home_team"]
        away = game["away_team"]

        pred_key = f"{home}_vs_{away}"
        if pred_key not in predictions:
            continue

        pred = predictions[pred_key]
        h2h = game.get("h2h")

        if not h2h:
            continue

        candidates = [
            {
                "outcome": "home_win",
                "label": f"{home} vence",
                "model_prob": pred["probabilities"]["home_win"],
                "odd": h2h.get("home_win"),
            },
            {
                "outcome": "draw",
                "label": "Empate",
                "model_prob": pred["probabilities"]["draw"],
                "odd": h2h.get("draw"),
            },
            {
                "outcome": "away_win",
                "label": f"{away} vence",
                "model_prob": pred["probabilities"]["away_win"],
                "odd": h2h.get("away_win"),
            },
        ]

        for c in candidates:
            if not c["odd"]:
                continue

            implied_prob = 1 / c["odd"]
            value = c["model_prob"] - implied_prob

            if value > 0.04:
                value_bets.append({
                    "match": f"{home} vs {away}",
                    "outcome": c["label"],
                    "model_prob_pct": round(c["model_prob"] * 100, 1),
                    "implied_prob_pct": round(implied_prob * 100, 1),
                    "odd": c["odd"],
                    "value_pct": round(value * 100, 1),
                    "value_rating": "alto" if value > 0.10 else "medio",
                    "bookmaker": h2h.get("bookmaker"),
                })

    value_bets.sort(key=lambda x: x["value_pct"], reverse=True)
    return value_bets
