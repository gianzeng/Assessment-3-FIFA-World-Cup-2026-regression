# FBref data

These CSV files come from FBref (https://fbref.com), collected on 14 October 2026.

| File | Content |
|---|---|
| `<Team>_2026.csv`, `<Team>_2025.csv`, `<Team>_2024.csv` | the team's "Scores & Fixtures, All Competitions" page of that season label; the 2026 page covers the World Cup cycle (qualifiers from 2023 and all matches of 2026) |
| `<Team>_<Competition>-<season>.csv` | the team's match log of one competition and season (UEFA Nations League 2024-2025, Africa Cup of Nations 2023, friendlies 2023 and similar) |
| `world_cup_2026.csv` | the 104 World Cup 2026 matches with their local dates and scores |

Columns of the team files: `Date`, `Comp`, `Opponent`, `GF`, `GA` (GF and GA are the team's goals and the opponent's goals, shootout goals in brackets).
Columns of `world_cup_2026.csv`: `Round`, `Date`, `Home`, `Score`, `Away`.

## Why there are competition pages
FBref only creates a calendar-year page when a team has matches under that season label. 26 requested calendar-year pages do not exist
(FBref answers HTTP 500, also on a second request), so their files only contain the header:
Algeria_2024, Australia_2024, Austria_2025, Belgium_2025, Cabo Verde_2024, Congo DR_2024, Croatia_2025, Curaçao_2024, Egypt_2024, France_2025, Germany_2025, Haiti_2024, IR Iran_2024, Iraq_2024, Iraq_2025, Japan_2024, Jordan_2024, Korea Republic_2024, Netherlands_2025, Portugal_2025, Qatar_2024, Saudi Arabia_2024, South Africa_2024, Spain_2025, Tunisia_2024, Uzbekistan_2024.
Matches of competitions whose season label differs from the calendar year, such as the UEFA Nations League 2024-2025 or the Africa Cup of Nations 2023, sit on their own competition pages.
We read each team's competition history page and added all competition pages with the season labels 2023, 2023-2024, 2024-2025, 2025-2026 and 2027 (104 pages).

## What FBref does not list
Some regional tournaments are missing, for example the Arab Cup, the Gulf Cup, the King's Cup and the CONCACAF Nations League.
Teams such as Jordan, Saudi Arabia, Iraq and Mexico therefore have fewer matches than they actually played.

## From files to results
`python src/import_fbref.py` turns these files into `data/raw/intl_results_pre_wc2026.csv` (matches before 11 June 2026, one copy of every match) and
`data/raw/intl_results_wc2026_check.csv` (the 104 World Cup matches), which `src/analyse.py` reads. Both files are already included.
