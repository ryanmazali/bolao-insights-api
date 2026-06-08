import pandas as pd
from statsbombpy import sb
import os

# Competições relevantes no StatsBomb Open Data
# sb.competitions() para ver todas disponíveis
TARGET_COMPETITIONS = {
    # Copa do Mundo (competition_id=43)
    (43, 106): "wc_2022",
    (43, 3):   "wc_2018",
    # UEFA Euro (competition_id=55)
    (55, 282): "euro_2024",
    (55, 43):  "euro_2020",
    # Copa América (competition_id=223) — open data só tem 2024
    (223, 282): "ca_2024",
}

def collect_match_results() -> pd.DataFrame:
    rows = []
    for (comp_id, season_id), label in TARGET_COMPETITIONS.items():
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                print(f"[StatsBomb] {label}: sem dados")
                continue

            matches["competition"] = label
            cols = [
                "match_id", "match_date", "competition",
                "home_team", "away_team",
                "home_score", "away_score",
                "match_status",
            ]
            available = [c for c in cols if c in matches.columns]
            rows.append(matches[available])
            print(f"[StatsBomb] {label}: {len(matches)} partidas coletadas")
        except Exception as e:
            print(f"[StatsBomb] Erro em {label}: {e}")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_squad_stats() -> pd.DataFrame:
    all_stats = []
    for (comp_id, season_id), label in TARGET_COMPETITIONS.items():
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches.empty:
                continue

            print(f"[StatsBomb] {label}: coletando eventos de {len(matches)} partidas...")
            match_ids = matches["match_id"].tolist()

            for match_id in match_ids:
                try:
                    events = sb.events(match_id=match_id)
                    if events.empty:
                        continue

                    for team in events["team"].dropna().unique():
                        team_events = events[events["team"] == team]
                        shots = team_events[team_events["type"] == "Shot"]

                        stats = {
                            "match_id": match_id,
                            "competition": label,
                            "team": team,
                            "shots": len(shots),
                            "shots_on_target": int(shots["shot_outcome"].isin(["Goal", "Saved"]).sum()) if "shot_outcome" in shots.columns else None,
                            "goals": int(shots["shot_outcome"].eq("Goal").sum()) if "shot_outcome" in shots.columns else None,
                            "xg": round(shots["shot_statsbomb_xg"].sum(), 4) if "shot_statsbomb_xg" in shots.columns else None,
                            "passes": int((team_events["type"] == "Pass").sum()),
                            "pressures": int((team_events["type"] == "Pressure").sum()),
                        }
                        all_stats.append(stats)
                except Exception:
                    continue

            print(f"[StatsBomb] {label}: squad stats coletadas")
        except Exception as e:
            print(f"[StatsBomb] Erro em {label}: {e}")

    return pd.DataFrame(all_stats) if all_stats else pd.DataFrame()


def collect_all_data():
    os.makedirs("data/raw", exist_ok=True)

    print("[StatsBomb] Coletando resultados de partidas...")
    results = collect_match_results()
    if not results.empty:
        results.to_csv("data/raw/match_results.csv", index=False)
        print(f"[StatsBomb] Resultados salvos: data/raw/match_results.csv ({len(results)} linhas)")

    print("\n[StatsBomb] Coletando squad stats por partida (demora alguns minutos)...")
    squad = collect_squad_stats()
    if not squad.empty:
        squad.to_csv("data/raw/squad_stats.csv", index=False)
        print(f"[StatsBomb] Squad stats salvos: data/raw/squad_stats.csv ({len(squad)} linhas)")


if __name__ == "__main__":
    collect_all_data()
