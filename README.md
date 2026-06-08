# Bolão Insights API

API de machine learning para análise e value bets da Copa do Mundo 2026.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Coleta de dados

```bash
python -m src.scraping.fbref
```

## Rodar API localmente

```bash
uvicorn api.main:app --reload
```
