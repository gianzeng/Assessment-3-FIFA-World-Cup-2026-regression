# HIT140 Assessment 3: FIFA World Cup 2026 regression

A total of two multiple linear regressions with eight predictors each: Task 2.1 regresses the goal difference of a match (104 matches) and Task 2.2 regresses the goals of a team in a match (208 rows) on chronological cross-validation. Ridge regression is compared to Ordinary Least Squares (OLS), Huber Regression, Lasso, Elastic Net, random forests, gradient boosting and a Poisson generalized linear model.

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/analyse.py
```

This takes a minute or two, uses no internet, and creates `data/processed/` (datasets), `results/` (tables) and `figures/` (reports).

## Code

- `src/data_prep.py`: loads the raw data and creates the two datasets
- `src/models.py`: the candidate models and cross-validation
- `src/figures.py`: figures for the reports
- `src/analyse.py`: executes everything
- `src/import_fbref.py`: creates the two `intl_results_*.csv` datasets from `data/raw/fbref/` (done already)

## Data

Collected on 14 October 2026, all in `data/raw/`:

- `tsdl_matches.csv`: World Cup 2026 matches and statistics, The Stats Don't Lie (thestatsdontlie.com/football/world-cup-2026)
- `fifa_bronze_match.json`: the bronze final, which is not included in the previous file, from FIFA
- `fifa_rankings_2026-06-11.csv`: FIFA ranking prior to the first match (inside.fifa.com/fifa-rankings/world-ranking/men)
- `fifa_goalkeeping.csv`: goals against per team, from the FIFA statistics page, for verification purposes
- `fbref/`: FBref match logs of the 48 nations and the World Cup (fbref.com), see `fbref/README.md`
- `intl_results_pre_wc2026.csv`, `intl_results_wc2026_check.csv`: created from `fbref/`

## Method

- Predictors use only data earlier than the match date; resting days capped to 14.
- The 79 matches up to 1 July are for development and the remaining 25 knockout matches are for testing.
- Candidate models and their parameters are selected via 3 expanding cross validation folds on the development data, seed: 1402026.