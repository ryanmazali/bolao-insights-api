import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException
from supabase import create_client

from src.models.predict import predict_match
from src.scraping.odds import calculate_value_bets, get_world_cup_odds, parse_odds

load_dotenv()

router = APIRouter(prefix="/admin", tags=["admin"])

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

# Mapeamento nomes em ingles (StatsBomb, usados por parse_odds/predict_match)
# -> nomes em portugues da tabela teams no Supabase.
TEAM_NAME_PT = {
    "Algeria": "Argélia",
    "Argentina": "Argentina",
    "Australia": "Austrália",
    "Austria": "Áustria",
    "Belgium": "Bélgica",
    "Bosnia-Herzegovina": "Bósnia",
    "Brazil": "Brasil",
    "Canada": "Canadá",
    "Cape Verde": "Cabo Verde",
    "Colombia": "Colômbia",
    "Croatia": "Croácia",
    "Curacao": "Curaçao",
    "Czech Republic": "Tchéquia",
    "DR Congo": "Rep. D. Congo",
    "Ecuador": "Equador",
    "Egypt": "Egito",
    "England": "Inglaterra",
    "France": "França",
    "Germany": "Alemanha",
    "Ghana": "Gana",
    "Haiti": "Haiti",
    "Iran": "Irã",
    "Iraq": "Iraque",
    "Ivory Coast": "Costa do Marfim",
    "Japan": "Japão",
    "Jordan": "Jordânia",
    "Mexico": "México",
    "Morocco": "Marrocos",
    "Netherlands": "Holanda",
    "New Zealand": "Nova Zelândia",
    "Norway": "Noruega",
    "Panama": "Panamá",
    "Paraguay": "Paraguai",
    "Portugal": "Portugal",
    "Qatar": "Catar",
    "Saudi Arabia": "Arábia Saudita",
    "Scotland": "Escócia",
    "Senegal": "Senegal",
    "South Africa": "África do Sul",
    "South Korea": "Coreia do Sul",
    "Spain": "Espanha",
    "Sweden": "Suécia",
    "Switzerland": "Suíça",
    "Tunisia": "Tunísia",
    "Turkey": "Turquia",
    "United States": "Estados Unidos",
    "Uruguay": "Uruguai",
    "Uzbekistan": "Uzbequistão",
}


def to_supabase_name(team_en: str) -> str:
    return TEAM_NAME_PT.get(team_en, team_en)


@router.post("/update-odds")
async def update_odds(x_admin_secret: str = Header(None)):
    """
    Busca odds da The Odds API, calcula predições do modelo,
    calcula value bets e salva tudo no Supabase (tabela match_odds).
    """
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    results = {
        "updated": 0,
        "inserted": 0,
        "errors": [],
        "value_bets_found": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Buscar odds
    raw_odds = get_world_cup_odds()
    if not raw_odds:
        raise HTTPException(status_code=503, detail="The Odds API não retornou dados")

    parsed = parse_odds(raw_odds)
    print(f"[update-odds] {len(parsed)} jogos com odds")

    # 2. Buscar todas as partidas do Supabase uma única vez
    match_resp = supabase.table("matches")\
        .select("id, status, home_team:teams!matches_home_team_id_fkey(name), away_team:teams!matches_away_team_id_fkey(name)")\
        .execute()
    matches = match_resp.data

    for game in parsed:
        home = game["home_team"]  # nome em ingles (StatsBomb), usado pelo modelo
        away = game["away_team"]
        home_pt = to_supabase_name(home)
        away_pt = to_supabase_name(away)

        try:
            # Buscar match_id correspondente
            match_id = None
            for m in matches:
                if not m["home_team"] or not m["away_team"]:
                    continue
                ht = m["home_team"]["name"]
                at = m["away_team"]["name"]
                if (home_pt.lower() in ht.lower() or ht.lower() in home_pt.lower()) and \
                   (away_pt.lower() in at.lower() or at.lower() in away_pt.lower()):
                    match_id = m["id"]
                    break

            if not match_id:
                results["errors"].append(f"Match não encontrado: {home} vs {away}")
                continue

            # 3. Buscar predição do modelo (nomes em ingles)
            try:
                pred = predict_match(home, away)
            except Exception as e:
                results["errors"].append(f"Predição falhou: {home} vs {away}: {e}")
                continue

            # 4. Calcular value bets para esse jogo
            pred_key = f"{home}_vs_{away}"
            predictions = {pred_key: pred}
            vbets = calculate_value_bets([game], predictions)

            # 5. Montar registro
            h2h = game.get("h2h") or {}
            totals = game.get("totals") or {}
            p = pred["probabilities"]
            xg = pred["expected_goals"]
            markets = pred.get("markets", {})
            ou = markets.get("over_under_25", {})
            btts = markets.get("btts", {})
            dc = markets.get("double_chance", {})

            record = {
                "match_id": match_id,
                "home_win_odd": h2h.get("home_win"),
                "draw_odd": h2h.get("draw"),
                "away_win_odd": h2h.get("away_win"),
                "over_25_odd": totals.get("over"),
                "under_25_odd": totals.get("under"),
                "bookmaker": h2h.get("bookmaker") or totals.get("bookmaker"),
                "home_win_prob": p.get("home_win"),
                "draw_prob": p.get("draw"),
                "away_win_prob": p.get("away_win"),
                "over_25_prob": ou.get("over"),
                "under_25_prob": ou.get("under"),
                "expected_goals_home": xg.get("home"),
                "expected_goals_away": xg.get("away"),
                "expected_goals_total": xg.get("total"),
                "btts_yes_prob": btts.get("yes"),
                "double_chance_1x": dc.get("1X"),
                "double_chance_x2": dc.get("X2"),
                "double_chance_12": dc.get("12"),
                "most_likely_result": pred.get("most_likely_result"),
                "value_bets": vbets,
                "odds_updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # 6. Upsert no Supabase
            existing = supabase.table("match_odds")\
                .select("id")\
                .eq("match_id", match_id)\
                .execute()

            if existing.data:
                # Atualizar odds e predições do modelo (mantém o cache em
                # dia com a versão atual do modelo v2)
                update_record = {k: v for k, v in record.items() if k != "match_id"}
                supabase.table("match_odds")\
                    .update(update_record)\
                    .eq("match_id", match_id)\
                    .execute()
                results["updated"] += 1
            else:
                # Inserir novo
                supabase.table("match_odds")\
                    .insert(record)\
                    .execute()
                results["inserted"] += 1

            results["value_bets_found"] += len(vbets)
            print(f"[update-odds] OK {home} vs {away} - {len(vbets)} value bets")

        except Exception as e:
            results["errors"].append(f"Erro em {home} vs {away}: {str(e)}")
            print(f"[update-odds] ERRO {home} vs {away}: {e}")

    print(f"[update-odds] Concluido: {results}")
    return results
