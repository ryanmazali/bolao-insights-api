import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

BASE_URL = "https://fbref.com"

COMPETITION_URLS = {
    # Copas do Mundo
    "wc_2022": "/en/comps/1/2022/2022-FIFA-World-Cup-Stats",
    "wc_2018": "/en/comps/1/2018/2018-FIFA-World-Cup-Stats",
    "wc_2014": "/en/comps/1/2014/2014-FIFA-World-Cup-Stats",

    # Eliminatórias UEFA
    "wcq_uefa_2022": "/en/comps/685/2022/2022-FIFA-World-Cup-Qualification-UEFA-Stats",
    "wcq_uefa_2018": "/en/comps/685/2018/2018-FIFA-World-Cup-Qualification-UEFA-Stats",

    # Eliminatórias CONMEBOL
    "wcq_conmebol_2022": "/en/comps/686/2022/2022-FIFA-World-Cup-Qualification-CONMEBOL-Stats",

    # Eurocopas
    "euro_2024": "/en/comps/676/2024/2024-European-Championship-Stats",
    "euro_2020": "/en/comps/676/2021/2021-European-Championship-Stats",

    # Copa América
    "ca_2024": "/en/comps/685/2024/2024-Copa-America-Stats",
    "ca_2021": "/en/comps/685/2021/2021-Copa-America-Stats",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
}

def get_squad_stats(competition_url: str, competition_name: str) -> pd.DataFrame:
    """Coleta stats de equipes de uma competição do FBref."""
    url = BASE_URL + competition_url

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(response.text)

        if not tables:
            print(f"[FBref] Nenhuma tabela encontrada em {competition_name}")
            return pd.DataFrame()

        df = tables[0]
        df['competition'] = competition_name

        print(f"[FBref] {competition_name}: {len(df)} equipes coletadas")
        return df

    except Exception as e:
        print(f"[FBref] Erro em {competition_name}: {e}")
        return pd.DataFrame()

    finally:
        time.sleep(4)  # respeitar rate limit do FBref

def get_match_results(competition_url: str, competition_name: str) -> pd.DataFrame:
    """Coleta resultados de partidas de uma competição."""
    url = BASE_URL + competition_url.replace("-Stats", "-Schedule")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(response.text)

        if not tables:
            return pd.DataFrame()

        df = tables[0]
        df['competition'] = competition_name

        if 'Score' in df.columns:
            df = df[df['Score'].notna()]
            df = df[df['Score'].str.contains('–|−|-', na=False)]

        print(f"[FBref] {competition_name}: {len(df)} partidas coletadas")
        return df

    except Exception as e:
        print(f"[FBref] Erro em {competition_name}: {e}")
        return pd.DataFrame()

    finally:
        time.sleep(4)

def collect_all_data():
    """Coleta todos os dados e salva em CSV."""
    os.makedirs("data/raw", exist_ok=True)

    all_results = []
    all_squad_stats = []

    for comp_key, comp_url in COMPETITION_URLS.items():
        print(f"\n[FBref] Coletando {comp_key}...")

        results = get_match_results(comp_url, comp_key)
        if not results.empty:
            all_results.append(results)

        squad = get_squad_stats(comp_url, comp_key)
        if not squad.empty:
            all_squad_stats.append(squad)

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv("data/raw/match_results.csv", index=False)
        print(f"\n[FBref] Resultados salvos: data/raw/match_results.csv")

    if all_squad_stats:
        pd.concat(all_squad_stats, ignore_index=True).to_csv("data/raw/squad_stats.csv", index=False)
        print(f"[FBref] Squad stats salvos: data/raw/squad_stats.csv")

if __name__ == "__main__":
    collect_all_data()
