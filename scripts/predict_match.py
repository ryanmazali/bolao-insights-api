import requests
import json

print("=== Previsão de Partida ===")
home = input("Seleção da casa: ")
away = input("Seleção visitante: ")

url = "https://bolao-insights-api.onrender.com/predict"
response = requests.post(url, json={"home_team": home, "away_team": away})

data = response.json()
print(json.dumps(data, indent=2, ensure_ascii=False))