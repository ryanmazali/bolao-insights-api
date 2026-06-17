import json

with open('data/models/player_metrics_data.json', encoding='utf-8') as f:
    data = json.load(f)

# Todos os jogadores do Egito
egito = {k: v for k, v in data.items() if v.get('country_code') == 'EGY'}
print(f"Jogadores do Egito no JSON: {len(egito)}")
for k, v in egito.items():
    print(f"  {v['name']} | pos={v['position']} | source={v['source']} | Sh_p90={v['Sh_p90']}")

# Busca por nome
for k, v in data.items():
    if 'Mohamed Salah' in str(v.get('name', '')) or 'Marmoush' in str(v.get('name', '')):
        print(f"\nEncontrado: {v['name']} | id={k} | Sh_p90={v['Sh_p90']} | source={v['source']}")