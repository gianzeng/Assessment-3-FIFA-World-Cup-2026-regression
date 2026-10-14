import re

import pandas as pd

from data_prep import RAW, database_name

FBREF = RAW / 'fbref'
FIRST_MATCH_DAY = '2026-06-11'
FIRST_YEAR = '2020-01-01'

# fbref spells some teams differently from fifa
FBREF_TO_FIFA = {
    'United States': 'USA', 'South Korea': 'Korea Republic', 'Iran': 'IR Iran', 'Ivory Coast': "Côte d'Ivoire",
    'Cape Verde': 'Cabo Verde', 'DR Congo': 'Congo DR', 'Czech Republic': 'Czechia', 'Turkey': 'Türkiye',
    "Bosnia & Herz'na": 'Bosnia and Herzegovina', 'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia–Herz': 'Bosnia and Herzegovina', 'Curacao': 'Curaçao',
}


def fifa_name(name):
    name = re.sub(r'^[a-z]{2,3}\s+', '', name.strip())
    return FBREF_TO_FIFA.get(name, name)


def database_team_name(name):
    return database_name(fifa_name(name))


def leading_integer(value):
    match = re.match(r'\s*(\d+)', str(value))
    return int(match.group(1)) if match else None


def read_table(path):
    table = pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]
    return table[table.iloc[:, 0].astype(str).str.lower() != table.columns[0].lower()]


def team_files():
    files = {}
    for path in sorted(FBREF.glob('*.csv')):
        if path.stem != 'world_cup_2026':
            files.setdefault(path.stem.rsplit('_', 1)[0], []).append(path)
    return files


def pre_tournament_matches(files):
    rows = []
    for team, paths in files.items():
        for path in paths:
            table = read_table(path)
            for column in ('Date', 'Comp', 'Opponent', 'GF', 'GA'):
                assert column in table.columns, f'{path.name}: missing column {column}'
            table = table.assign(gf=table.GF.map(leading_integer), ga=table.GA.map(leading_integer))
            table = table[table.gf.notna() & table.ga.notna()]
            table = table[(table.Date >= FIRST_YEAR) & (table.Date < FIRST_MATCH_DAY)]
            for row in table.itertuples():
                rows.append({'date': row.Date, 'team': database_team_name(team),
                             'away_team': database_team_name(row.Opponent), 'home_score': int(row.gf),
                             'away_score': int(row.ga), 'tournament': row.Comp})
    matches = pd.DataFrame(rows).rename(columns={'team': 'home_team'})
    # a match between two finalists is in both files, so put the teams in alphabetical order and keep one copy
    swap = matches.home_team > matches.away_team
    matches.loc[swap, ['home_team', 'away_team']] = matches.loc[swap, ['away_team', 'home_team']].to_numpy()
    matches.loc[swap, ['home_score', 'away_score']] = matches.loc[swap, ['away_score', 'home_score']].to_numpy()
    matches = matches.drop_duplicates(['date', 'home_team', 'away_team'])
    return matches.sort_values(['date', 'home_team']).reset_index(drop=True)


def world_cup_matches():
    table = read_table(FBREF / 'world_cup_2026.csv')
    for column in ('Date', 'Home', 'Score', 'Away'):
        assert column in table.columns, f'world_cup_2026.csv: missing column {column}'
    scores = table.Score.astype(str).str.replace(r'\(.*?\)', '', regex=True).str.extract(r'(\d+)\s*[–-]\s*(\d+)')
    table = table.assign(home_score=scores[0], away_score=scores[1]).dropna(subset=['home_score', 'away_score'])
    return pd.DataFrame({
        'date': table.Date, 'home_team': table.Home.map(database_team_name), 'away_team': table.Away.map(database_team_name),
        'home_score': table.home_score.astype(int), 'away_score': table.away_score.astype(int),
        'tournament': 'FIFA World Cup'}).reset_index(drop=True)


def main():
    files = team_files()
    finalists = pd.read_csv(RAW / 'fifa_rankings_2026-06-11.csv').team
    missing = [team for team in finalists if team not in files]
    assert not missing, f'No FBref file for: {missing}'

    pre = pre_tournament_matches(files)
    check = world_cup_matches()
    assert len(check) == 104, f'world_cup_2026.csv has {len(check)} played matches, expected 104'
    counts = {team: int(((pre.home_team == database_name(team)) | (pre.away_team == database_name(team))).sum())
              for team in finalists}
    print('pre-tournament matches per team: min', min(counts.values()), 'max', max(counts.values()))
    print('teams with fewer than 20:', {t: n for t, n in counts.items() if n < 20})
    pre.to_csv(RAW / 'intl_results_pre_wc2026.csv', index=False)
    check.to_csv(RAW / 'intl_results_wc2026_check.csv', index=False)
    print('wrote', len(pre), 'pre-tournament rows and', len(check), 'World Cup rows')


if __name__ == '__main__':
    main()
