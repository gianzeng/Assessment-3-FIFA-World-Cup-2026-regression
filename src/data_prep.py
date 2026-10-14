import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
CLEAN = ROOT / 'data/processed'

HOSTS = ['Mexico', 'Canada', 'USA']
REST_CAP = 14  # days; one team rested 75 days before the opening match, which would dominate the rest gap
STAGES = ['GROUP STAGE', 'ROUND OF 32', 'ROUND OF 16', 'QUARTER FINALS', 'SEMI FINALS', 'FINAL']
TSDL_COLUMNS = {'yellow': 9, 'red': 11, 'corners': 17, 'shots': 22, 'sot': 24, 'fouls': 26}
STAT_FIELDS = ['goals', 'shots', 'sot', 'corners', 'yellow', 'red', 'fouls']

# team names are spelled differently in each source
TSDL_TO_FIFA = {
    'South Korea': 'Korea Republic', 'Czech Republic': 'Czechia',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 'Turkey': 'Türkiye',
    'Ivory Coast': "Côte d'Ivoire", 'Cape Verde': 'Cabo Verde', 'Iran': 'IR Iran',
    'D.R. Congo': 'Congo DR', 'Curacao': 'Curaçao',
}
FIFA_TO_DATABASE = {
    'Cabo Verde': 'Cape Verde', 'Congo DR': 'DR Congo', 'Czechia': 'Czech Republic',
    "Côte d'Ivoire": 'Ivory Coast', 'IR Iran': 'Iran', 'Korea Republic': 'South Korea',
    'Türkiye': 'Turkey', 'USA': 'United States',
}


def database_name(team):
    return FIFA_TO_DATABASE.get(team, team)


def read_tsdl():
    rows = []
    stage = None
    with (RAW / 'tsdl_matches.csv').open(encoding='utf-8-sig') as file:
        for line_number, row in enumerate(csv.reader(file), 1):
            if row[0] in STAGES:
                stage = row[0]
                continue
            if not re.fullmatch(r'\d{2}/\d{2}/2026', row[0]):
                continue
            match = {
                'date': pd.to_datetime(row[0], dayfirst=True).strftime('%Y-%m-%d'),
                'stage': stage,
                'team1': TSDL_TO_FIFA.get(row[1], row[1]),
                'team2': TSDL_TO_FIFA.get(row[4], row[4]),
                'goals1': int(row[2].replace('p', '')),
                'goals2': int(row[3].replace('p', '')),
                'source': 'TSDL',
                'source_row': line_number,
                'shootout': 'p' in row[2] + row[3],
            }
            for name, column in TSDL_COLUMNS.items():
                match[f'{name}1'] = int(row[column])
                match[f'{name}2'] = int(row[column + 1])
            rows.append(match)
    assert len(rows) == 103
    return rows


def read_bronze_final():
    match = json.loads((RAW / 'fifa_bronze_match.json').read_text())
    match.update(source='FIFA', source_row=None, shootout=False)
    return {key: value for key, value in match.items() if key not in ['date_basis', 'source_url']}


def align_with_database(matches):
    database = pd.read_csv(RAW / 'intl_results_wc2026_check.csv')
    used = set()
    dates = []
    for match in matches.to_dict('records'):
        team1, team2 = database_name(match['team1']), database_name(match['team2'])
        found = None
        for index, row in database.iterrows():
            if index in used or {row.home_team, row.away_team} != {team1, team2}:
                continue
            score = (row.home_score, row.away_score) if row.home_team == team1 else (row.away_score, row.home_score)
            same_score = score == (match['goals1'], match['goals2'])
            close_date = abs((pd.Timestamp(row.date) - pd.Timestamp(match['date'])).days) <= 1
            if same_score and close_date:
                found = index
                break
        assert found is not None, (match['team1'], match['team2'], match['date'])
        used.add(found)
        dates.append(database.loc[found, 'date'])
    # from here on the fbref date is used, the tsdl date is only kept for reference
    matches['tsdl_date'] = matches['date']
    matches['date'] = dates
    return matches


def team_match_table(matches):
    rows = []
    for match in matches.to_dict('records'):
        for side, other in [(1, 2), (2, 1)]:
            row = {key: match[key] for key in ['match_id', 'date', 'stage', 'source']}
            row.update(team=match[f'team{side}'], opponent=match[f'team{other}'], goals_against=match[f'goals{other}'])
            row.update({field: match[f'{field}{side}'] for field in STAT_FIELDS})
            rows.append(row)
    return pd.DataFrame(rows)


def load_matches():
    matches = pd.DataFrame(read_tsdl() + [read_bronze_final()])
    matches = matches.sort_values(['date', 'team1', 'team2']).reset_index(drop=True)
    assert len(matches) == 104
    assert matches.stage.eq('GROUP STAGE').sum() == 72
    for side in (1, 2):
        assert (matches[f'sot{side}'] <= matches[f'shots{side}']).all()

    matches = align_with_database(matches)
    matches = matches.sort_values(['date', 'team1', 'team2']).reset_index(drop=True)
    matches.insert(0, 'match_id', [f'WC26_{i:03d}' for i in range(1, 105)])

    team_matches = team_match_table(matches)
    ranks = pd.read_csv(RAW / 'fifa_rankings_2026-06-11.csv').set_index('team')['world_rank']
    assert set(team_matches.team) == set(ranks.index) and len(ranks) == 48
    team_matches['world_rank'] = team_matches.team.map(ranks)

    official = pd.read_csv(RAW / 'fifa_goalkeeping.csv').set_index('team')
    conceded = team_matches.groupby('team').goals_against.sum()
    assert (conceded == official.loc[conceded.index, 'goals_conceded']).all()
    group_games = team_matches[team_matches.stage.eq('GROUP STAGE')].groupby('team').size()
    assert group_games.eq(3).all()

    matches.to_csv(CLEAN / 'matches_104.csv', index=False)
    team_matches.to_csv(CLEAN / 'team_matches_208.csv', index=False)
    return matches, team_matches


def pre_tournament_results():
    results = pd.read_csv(RAW / 'intl_results_pre_wc2026.csv')
    assert results.date.max() < '2026-06-11'
    home = results.rename(columns={'home_team': 'team', 'home_score': 'gf', 'away_score': 'ga'})
    away = results.rename(columns={'away_team': 'team', 'away_score': 'gf', 'home_score': 'ga'})
    columns = ['date', 'team', 'gf', 'ga']
    return pd.concat([home[columns], away[columns]]).sort_values('date')


def pre_tournament_form(team_matches, last_n):
    results = pre_tournament_results()
    form = {}
    for team in sorted(team_matches.team.unique()):
        recent = results[results.team.eq(database_name(team))].tail(last_n)
        assert len(recent) == last_n, (team, len(recent))
        form[team] = {
            'gf': recent.gf.mean(),
            'ga': recent.ga.mean(),
            'cs': recent.ga.eq(0).mean(),
            'var': recent.gf.var(ddof=1),
            'last': recent.date.max(),
        }
    return form


def team_features(team_matches, prior):
    # prior = (mode, N, k): raw history, tournament mean, or the team's last N pre-tournament games
    mode, last_n, k = prior
    form = pre_tournament_form(team_matches, last_n) if mode == 'pre' else {}
    mean_fields = [('goals', 'gf'), ('goals_against', 'ga'), ('shots', 'shots'), ('sot', 'sot'),
                   ('corners', 'corners'), ('yellow', 'yellow'), ('fouls', 'fouls')]
    rows = []
    for row in team_matches.to_dict('records'):
        history = team_matches[team_matches.team.eq(row['team']) & team_matches.date.lt(row['date'])]
        earlier = team_matches[team_matches.date.lt(row['date'])]
        games = len(history)
        pre = form.get(row['team'])
        # the prior counts as k extra games next to the games played so far
        shrink = k > 0 and mode != 'raw'

        def blend(prior_value, total):
            if shrink and not np.isnan(prior_value):
                return (k * prior_value + total) / (k + games)
            return total / games if games else np.nan

        def tournament_mean(column, transform=lambda x: x):
            return transform(earlier[column]).mean() if len(earlier) >= 2 else np.nan

        last_game = history.date.max() if games else (pre['last'] if pre else None)
        features = {key: row[key] for key in ['match_id', 'date', 'team', 'opponent', 'goals', 'stage']}
        features['prior_matches'] = games
        rest = np.nan if last_game is None else (pd.Timestamp(row['date']) - pd.Timestamp(last_game)).days
        features['rest_days'] = np.minimum(rest, REST_CAP)
        for column, name in mean_fields:
            prior_value = pre[name] if pre and name in ('gf', 'ga') else tournament_mean(column)
            features[f'{name}_mean'] = blend(prior_value, history[column].sum())

        prior_variance = pre['var'] if pre else (earlier.goals.var(ddof=1) if len(earlier) >= 2 else np.nan)
        squares = (games - 1) * history.goals.var(ddof=1) if games > 1 else 0.0
        if shrink and not np.isnan(prior_variance):
            features['goal_sd'] = float(np.sqrt((k * prior_variance + squares) / (k + max(games - 1, 0))))
        else:
            features['goal_sd'] = history.goals.std(ddof=1) if games > 1 else np.nan
        features['clean_sheet_rate'] = blend(pre['cs'] if pre else tournament_mean('goals_against', lambda x: x.eq(0)),
                                             history.goals_against.eq(0).sum())
        features['host_nation'] = int(row['team'] in HOSTS)
        features['world_rank'] = row['world_rank']
        rows.append(features)
    return pd.DataFrame(rows)


def build_datasets(matches, team_matches, prior, f1, f2):
    team = team_features(team_matches, prior)
    assert team.prior_matches.eq(0).sum() == 48
    lookup = team.set_index(['match_id', 'team'])

    rows = []
    for match in matches.to_dict('records'):
        first = lookup.loc[(match['match_id'], match['team1'])]
        second = lookup.loc[(match['match_id'], match['team2'])]
        row = {key: match[key] for key in ['match_id', 'date', 'team1', 'team2']}
        row['goal_difference'] = match['goals1'] - match['goals2']
        for name in ['gf', 'ga', 'shots', 'corners', 'yellow', 'fouls']:
            row[f'{name}_gap'] = first[f'{name}_mean'] - second[f'{name}_mean']
        row['rank_gap'] = first.world_rank - second.world_rank
        row['rest_gap'] = first.rest_days - second.rest_days
        rows.append(row)
    task1 = pd.DataFrame(rows)[['match_id', 'date', 'team1', 'team2'] + f1 + ['goal_difference']]

    task2 = team[['match_id', 'date', 'team', 'opponent', 'goals']].copy()
    for feature in f2:
        if feature.startswith('opponent_'):
            column = feature[len('opponent_'):]
            task2[feature] = [lookup.loc[(r.match_id, r.opponent), column] for r in task2.itertuples()]
        else:
            task2[feature] = team[feature]
    task2 = task2[['match_id', 'date', 'team', 'opponent'] + f2 + ['goals']]

    assert len(f1) == len(f2) == 8 and len(task1) == 104 and len(task2) == 208
    assert np.array_equal(task1.goal_difference.to_numpy(), (matches.goals1 - matches.goals2).to_numpy())
    return task1, task2, team
