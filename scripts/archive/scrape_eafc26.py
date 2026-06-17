"""Scraping da API oficial do EA FC 26 para as 48 seleções da Copa 2026.

Os dados são retornados como HTML com __NEXT_DATA__ embutido —
não há endpoint XHR separado; os jogadores estão em:
  props.pageProps.ratingsEntries.items  (100 por página)
  props.pageProps.ratingsEntries.totalItems

Paginação: ?page=N  (começa em 1)
Base URL: https://www.ea.com/pt-br/games/ea-sports-fc/ratings/nations-ratings/{slug}/{id}

Uso (raiz do projeto, venv ativado):
    python scripts/scrape_eafc26.py
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Configurações ──────────────────────────────────────────────────────────
BASE_URL = "https://www.ea.com/pt-br/games/ea-sports-fc/ratings/nations-ratings/{slug}/{nat_id}"
OUTPUT_PATH = Path("data/raw/ea_fc26_official.csv")
DELAY_BETWEEN_REQUESTS = 1.0   # segundos entre requests
MAX_RETRIES = 2
BACKOFF_BASE = 3               # segundos base para backoff exponencial

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.ea.com/pt-br/games/ea-sports-fc/ratings",
}

# ── Mapeamento: nome Supabase PT → (slug EA, nationality_id EA) ───────────
# Qatar (Catar) não possui jogadores no EA FC26 e não tem página de nação.
COPA_2026_NATIONS = {
    "México":          ("mexico",                83),
    "África do Sul":   ("south-africa",         140),
    "Coreia do Sul":   ("korea-republic",        167),
    "Tchéquia":        ("czech-republic",         12),
    "Canadá":          ("canada",                70),
    "Bósnia":          ("bosnia-and-herzegovina",  8),
    "Suíça":           ("switzerland",            47),
    "Brasil":          ("brazil",                54),
    "Marrocos":        ("morocco",              129),
    "Haiti":           ("haiti",                 80),
    "Escócia":         ("scotland",              42),
    "Estados Unidos":  ("united-states",         95),
    "Paraguai":        ("paraguay",              58),
    "Austrália":       ("australia",            195),
    "Turquia":         ("turkey",                48),
    "Alemanha":        ("germany",               21),
    "Curaçao":         ("curacao",               85),
    "Costa do Marfim": ("cote-d-ivoire",        108),
    "Equador":         ("ecuador",               57),
    "Holanda":         ("holland",               34),
    "Japão":           ("japan",                163),
    "Suécia":          ("sweden",                46),
    "Tunísia":         ("tunisia",              145),
    "Bélgica":         ("belgium",                7),
    "Egito":           ("egypt",               111),
    "Irã":             ("iran",                 161),
    "Nova Zelândia":   ("new-zealand",          198),
    "Espanha":         ("spain",                 45),
    "Cabo Verde":      ("cape-verde-islands",   104),
    "Arábia Saudita":  ("saudi-arabia",         183),
    "Uruguai":         ("uruguay",               60),
    "França":          ("france",                18),
    "Senegal":         ("senegal",              136),
    "Iraque":          ("iraq",                 162),
    "Noruega":         ("norway",                36),
    "Argentina":       ("argentina",             52),
    "Argélia":         ("algeria",               97),
    "Áustria":         ("austria",                4),
    "Jordânia":        ("jordan",               164),
    "Portugal":        ("portugal",              38),
    "Rep. D. Congo":   ("congo-dr",             110),
    "Uzbequistão":     ("uzbekistan",           191),
    "Colômbia":        ("colombia",              56),
    "Inglaterra":      ("england",               14),
    "Croácia":         ("croatia",               10),
    "Gana":            ("ghana",               117),
    "Panamá":          ("panama",                87),
    # Catar: sem dados no EA FC26 (0 jogadores)
}

# ── Mapeamento de posição ──────────────────────────────────────────────────
POS_CATEGORY = {
    "GOL": "GK",
    "ATA": "FW", "PD": "FW", "PE": "FW",
    "MEI": "MF", "MC": "MF", "VOL": "MF", "MD": "MF", "ME": "MF",
    "ZAG": "DF", "LD": "DF", "LE": "DF",
}

# Stats a extrair (nome na API → nome no CSV)
STATS_COLS = [
    "finishing", "positioning", "shotPower", "longShots", "volleys",
    "standingTackle", "slidingTackle", "interceptions", "defensiveAwareness",
    "aggression", "strength",
    "gkDiving", "gkHandling", "gkKicking", "gkPositioning", "gkReflexes",
]

SEP = "=" * 70


# ── Funções de request ─────────────────────────────────────────────────────

def fetch_page(slug: str, nat_id: int, page: int, session: requests.Session) -> dict | None:
    url = BASE_URL.format(slug=slug, nat_id=nat_id)
    if page > 1:
        url += f"?page={page}"

    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
            r.raise_for_status()
            m = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                r.text, re.DOTALL,
            )
            if not m:
                print(f"    [WARN] __NEXT_DATA__ não encontrado em {url}")
                return None

            data = json.loads(m.group(1))
            pp = data["props"]["pageProps"]
            if "ratingsEntries" not in pp:
                print(f"    [WARN] ratingsEntries ausente em {url}")
                return None

            return pp["ratingsEntries"]

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"    [RETRY {attempt+1}/{MAX_RETRIES}] {e}  aguardando {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [ERRO] Desistindo de {slug}/{nat_id} pág.{page}: {e}")
                return None

    return None


# ── Extração de um jogador ─────────────────────────────────────────────────

def extract_player(player: dict, team_name: str) -> dict:
    common = player.get("commonName") or ""
    first  = player.get("firstName") or ""
    last   = player.get("lastName") or ""
    name   = common if common else f"{first} {last}".strip()

    nat_label  = (player.get("nationality") or {}).get("label", "")
    pos_short  = (player.get("position") or {}).get("shortLabel", "")
    pos_cat    = POS_CATEGORY.get(pos_short, "MF")  # fallback MF

    stats = player.get("stats") or {}
    row = {
        "player_id":   player.get("id"),
        "name":        name,
        "nationality": nat_label,
        "team_name":   team_name,
        "position_short": pos_short,
        "pos_category":   pos_cat,
        "overall":     player.get("overallRating"),
    }
    for stat in STATS_COLS:
        row[stat] = (stats.get(stat) or {}).get("value")

    return row


# ── Scraping de uma nação ─────────────────────────────────────────────────

def scrape_nation(team_name: str, slug: str, nat_id: int, session: requests.Session) -> list[dict]:
    # Página 1
    entries = fetch_page(slug, nat_id, 1, session)
    if entries is None:
        return []

    total = entries["totalItems"]
    players = list(entries["items"])

    pages_needed = (total + 99) // 100  # ceil
    for page in range(2, pages_needed + 1):
        time.sleep(DELAY_BETWEEN_REQUESTS)
        more = fetch_page(slug, nat_id, page, session)
        if more:
            players.extend(more["items"])

    rows = [extract_player(p, team_name) for p in players]
    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  SCRAPING EA FC 26 — 48 seleções Copa 2026")
    print(SEP)
    print(f"\nEndpoint: {BASE_URL}")
    print(f"Estratégia: __NEXT_DATA__ embutido no HTML, paginação via ?page=N")
    print(f"Rate limit: {DELAY_BETWEEN_REQUESTS}s entre requests, {MAX_RETRIES} retries\n")

    session = requests.Session()
    all_rows: list[dict] = []
    summary: list[dict] = []

    nations_sorted = sorted(COPA_2026_NATIONS.items(), key=lambda x: x[0])

    for i, (team_name, (slug, nat_id)) in enumerate(nations_sorted, 1):
        print(f"[{i:02d}/{len(COPA_2026_NATIONS)}] {team_name:<20} ({slug}/{nat_id})")
        rows = scrape_nation(team_name, slug, nat_id, session)
        n = len(rows)
        all_rows.extend(rows)
        summary.append({"team": team_name, "n_players": n})
        print(f"        → {n} jogadores")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Catar: sem dados
    all_rows.append({
        "player_id": None, "name": "—", "nationality": "Qatar",
        "team_name": "Catar", "position_short": None, "pos_category": None,
        "overall": None, **{s: None for s in STATS_COLS},
    })
    summary.append({"team": "Catar", "n_players": 0})

    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── Relatório ──────────────────────────────────────────────────────────
    real_players = df[df["overall"].notna()]
    print(f"\n{SEP}")
    print("  RELATÓRIO FINAL")
    print(SEP)
    print(f"\nTotal de jogadores scrapeados: {len(real_players):,}")
    print(f"Arquivo: {OUTPUT_PATH}")

    df_summary = pd.DataFrame(summary).sort_values("n_players", ascending=False)
    print(f"\nJogadores por seleção (ordenado):")
    print(df_summary.to_string(index=False))

    print(f"\nTop 5 overall por seleção:")
    for team in sorted(COPA_2026_NATIONS.keys()) + ["Catar"]:
        grp = real_players[real_players["team_name"] == team]
        if grp.empty:
            continue
        top5 = grp.nlargest(5, "overall")[["name", "position_short", "overall"]]
        print(f"\n  {team}:")
        for _, row in top5.iterrows():
            print(f"    {row['name']:<30} {row['position_short']:<5} {int(row['overall'])}")

    rasos = df_summary[df_summary["n_players"] < 15]
    print(f"\nSeleções com menos de 15 jogadores ({len(rasos)}):")
    print(rasos.to_string(index=False))


if __name__ == "__main__":
    main()
