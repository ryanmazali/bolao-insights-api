import pandas as pd
import numpy as np
import os

# Competições mais recentes e mais próximas do formato Copa do Mundo pesam mais
COMP_WEIGHTS = {
    'wc_2022':   1.0,
    'ca_2024':   1.0,
    'euro_2024': 1.0,
    'euro_2020': 0.7,
    'wc_2018':   0.5,
}


def _build_match_level(squad: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna uma linha por (time, partida) com stats ofensivas e defensivas.
    Faz self-join em match_id para obter stats do adversário.
    """
    opp = squad.rename(columns={
        'team':            'opp_team',
        'shots':           'opp_shots',
        'shots_on_target': 'opp_sot',
        'goals':           'goals_conceded',
        'xg':              'xg_against',
        'passes':          'opp_passes',
        'pressures':       'opp_pressures',
    })[['match_id', 'opp_team', 'opp_shots', 'opp_sot',
        'goals_conceded', 'xg_against', 'opp_passes', 'opp_pressures']]

    df = squad.merge(opp, on='match_id')
    df = df[df['team'] != df['opp_team']].copy()

    df = df.merge(
        results[['match_id', 'home_team', 'away_team', 'home_score', 'away_score', 'match_date']],
        on='match_id'
    )

    df['comp_weight'] = df['competition'].map(COMP_WEIGHTS).fillna(0.5)

    def get_result(row):
        g_for = row['home_score'] if row['team'] == row['home_team'] else row['away_score']
        g_ag  = row['away_score'] if row['team'] == row['home_team'] else row['home_score']
        if g_for > g_ag: return 'W'
        if g_for == g_ag: return 'D'
        return 'L'

    df['result']  = df.apply(get_result, axis=1)
    df['is_home'] = (df['team'] == df['home_team']).astype(int)
    df['xg_diff_match'] = df['xg'] - df['xg_against']

    return df


def build_team_features(squad: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega stats por time via média ponderada pela relevância da competição.
    Retorna um DataFrame com uma linha por time.
    """
    ml = _build_match_level(squad, results)

    records = []
    for team, grp in ml.groupby('team'):
        w = grp['comp_weight']
        w_sum = w.sum()

        def wa(col):
            return (grp[col] * w).sum() / w_sum

        records.append({
            'team':          team,
            'n_matches':     len(grp),
            # ofensivo
            'xg_for':        round(wa('xg'), 4),
            'goals_for':     round(wa('goals'), 4),
            'shots':         round(wa('shots'), 2),
            'sot':           round(wa('shots_on_target'), 2),
            'passes':        round(wa('passes'), 1),
            'pressures':     round(wa('pressures'), 1),
            # defensivo
            'xg_against':    round(wa('xg_against'), 4),
            'goals_against': round(wa('goals_conceded'), 4),
            'opp_shots':     round(wa('opp_shots'), 2),
            # diferencial
            'xg_diff':       round(wa('xg_diff_match'), 4),
            # resultados ponderados
            'win_rate':      round((grp['result'].eq('W') * w).sum() / w_sum, 4),
            'draw_rate':     round((grp['result'].eq('D') * w).sum() / w_sum, 4),
        })

    tf = pd.DataFrame(records)
    tf['loss_rate'] = (1 - tf['win_rate'] - tf['draw_rate']).round(4)
    return tf.sort_values('xg_diff', ascending=False).reset_index(drop=True)


def build_match_features(results: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Feature matrix de treino: uma linha por partida com features de ambos os times.
    Target: result (0=away win, 1=draw, 2=home win), total_goals.
    """
    tf = team_features.set_index('team')

    FEAT_COLS = [
        'xg_for', 'goals_for', 'shots', 'sot', 'passes', 'pressures',
        'xg_against', 'goals_against', 'xg_diff', 'win_rate', 'draw_rate', 'n_matches',
    ]

    rows = []
    skipped = []
    for _, match in results.iterrows():
        ht, at = match['home_team'], match['away_team']
        if ht not in tf.index:
            skipped.append(ht)
            continue
        if at not in tf.index:
            skipped.append(at)
            continue

        h, a = tf.loc[ht], tf.loc[at]
        hs, as_ = int(match['home_score']), int(match['away_score'])

        if hs > as_:   result = 2  # home win
        elif hs == as_: result = 1  # draw
        else:           result = 0  # away win

        row = {
            'match_id':    match['match_id'],
            'match_date':  match['match_date'],
            'competition': match['competition'],
            'home_team':   ht,
            'away_team':   at,
            'home_score':  hs,
            'away_score':  as_,
        }

        for col in FEAT_COLS:
            row[f'home_{col}'] = h[col]
            row[f'away_{col}'] = a[col]

        # features diferenciais (home - away perspective)
        row['diff_xg_for']      = round(h['xg_for']     - a['xg_for'],      4)
        row['diff_xg_against']  = round(h['xg_against'] - a['xg_against'],   4)
        row['diff_xg_net']      = round(h['xg_diff']    - a['xg_diff'],      4)
        row['diff_win_rate']    = round(h['win_rate']    - a['win_rate'],     4)
        row['diff_goals_for']   = round(h['goals_for']  - a['goals_for'],    4)
        row['diff_shots']       = round(h['shots']       - a['shots'],        2)
        row['diff_pressures']   = round(h['pressures']   - a['pressures'],    1)

        row['result']      = result
        row['total_goals'] = hs + as_

        rows.append(row)

    if skipped:
        print(f"[features] Times sem dados ignorados: {sorted(set(skipped))}")

    return pd.DataFrame(rows)


def run():
    os.makedirs("data/processed", exist_ok=True)

    squad   = pd.read_csv("data/raw/squad_stats.csv")
    results = pd.read_csv("data/raw/match_results.csv")

    print(f"[features] {len(results)} partidas | {squad['team'].nunique()} times")

    print("[features] Construindo team_features...")
    tf = build_team_features(squad, results)
    tf.to_csv("data/processed/team_features.csv", index=False)
    print(f"[features] team_features.csv: {len(tf)} times")
    print(tf[['team', 'n_matches', 'xg_for', 'xg_against', 'xg_diff', 'win_rate']].head(15).to_string(index=False))

    print("\n[features] Construindo match_features...")
    mf = build_match_features(results, tf)
    mf.to_csv("data/processed/match_features.csv", index=False)
    print(f"[features] match_features.csv: {len(mf)} partidas | {len(mf.columns)} colunas")
    print(f"  Distribuição resultado: { mf['result'].value_counts().sort_index().to_dict() }")
    print(f"  (0=away win, 1=draw, 2=home win)")
    print(f"  Total de gols médio: {mf['total_goals'].mean():.2f}")
    print(f"  Colunas: {mf.columns.tolist()}")


if __name__ == "__main__":
    run()
