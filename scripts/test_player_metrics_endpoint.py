"""Teste manual do endpoint POST /predict/player-metrics (servidor local).

Não modifica nenhum endpoint existente — apenas faz requests HTTP contra
um servidor uvicorn já em execução (api.main:app) e imprime as respostas.
"""

import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8123"

MODELS_DIR = Path("data/models")
with open(MODELS_DIR / "player_metrics_data.json", encoding="utf-8") as f:
    player_metrics_data = json.load(f)

PLAYERS_BY_TEAM_NAME = {(p["team"], p["name"]): pid for pid, p in player_metrics_data.items()}

STARTING_XI = {
    "Brasil": [
        "Alisson", "Danilo", "Marquinhos", "Gabriel Magalhães", "Alex Sandro",
        "Casemiro", "Bruno Guimarães", "Lucas Paquetá",
        "Raphinha", "Vini Jr.", "Matheus Cunha",
    ],
    "Marrocos": [
        "Bounou", "Achraf Hakimi", "Aguerd", "Diop", "Mazraoui",
        "Amrabat", "El Khannouss", "Azzedine Ounahi",
        "Ez Abde", "Ayoub El Kaabi", "Brahim Díaz",
    ],
    "Alemanha": [
        "Manuel Neuer", "Joshua Kimmich", "Antonio Rudiger", "Jonathan Tah", "David Raum",
        "Florian Wirtz", "Leon Goretzka", "Jamal Musiala",
        "Leroy Sané", "Kai Havertz", "Maximilian Beier",
    ],
    "Japão": [
        "Zion Suzuki", "Yukinari Sugawara", "Ko Itakura", "Takehiro Tomiyasu", "Hiroki Ito",
        "Wataru Endo", "Daichi Kamada", "Ritsu Doan",
        "Junya Ito", "Takefusa Kubo", "Ayase Ueda",
    ],
}


def xi_ids(team_pt: str) -> list[str]:
    return [PLAYERS_BY_TEAM_NAME[(team_pt, name)] for name in STARTING_XI[team_pt]]


print("=" * 90)
print("TESTE 1: Brasil vs Marrocos (ambos com StatsBomb)")
print("=" * 90)
payload = {
    "home_team": "Brazil",
    "away_team": "Morocco",
    "home_players": xi_ids("Brasil"),
    "away_players": xi_ids("Marrocos"),
    "stage": "group",
}
resp = requests.post(f"{BASE_URL}/predict/player-metrics", json=payload)
print(f"status_code={resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

print("\n" + "=" * 90)
print("TESTE 2: Alemanha vs Japão (Japão sem StatsBomb)")
print("=" * 90)
payload = {
    "home_team": "Germany",
    "away_team": "Japan",
    "home_players": xi_ids("Alemanha"),
    "away_players": xi_ids("Japão"),
    "stage": "quarterfinal",
}
resp = requests.post(f"{BASE_URL}/predict/player-metrics", json=payload)
print(f"status_code={resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

print("\n" + "=" * 90)
print("TESTE 3: player_id inexistente")
print("=" * 90)
payload = {
    "home_team": "Brazil",
    "away_team": "Morocco",
    "home_players": ["00000000-0000-0000-0000-000000000000"] + xi_ids("Brasil")[1:],
    "away_players": xi_ids("Marrocos"),
    "stage": "group",
}
resp = requests.post(f"{BASE_URL}/predict/player-metrics", json=payload)
print(f"status_code={resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
