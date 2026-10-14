import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

import figures
from data_prep import CLEAN, RAW, build_datasets, load_matches
from models import SEED, candidates, cross_validate, date_bootstrap, error_metrics, predict, validation_blocks
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

F1 = ['gf_gap', 'ga_gap', 'shots_gap', 'rest_gap', 'corners_gap', 'yellow_gap', 'rank_gap', 'fouls_gap']
F2 = ['gf_mean', 'opponent_ga_mean', 'shots_mean', 'sot_mean', 'goal_sd', 'opponent_world_rank',
      'host_nation', 'world_rank']
L1 = ['Goals for gap', 'Goals against gap', 'Shots gap', 'Rest days gap', 'Corners gap', 'Yellow cards gap',
      'FIFA rank gap', 'Fouls gap']
L2 = ['Goals for form', 'Opponent goals against form', 'Shots', 'Shots on target', 'Goal variability',
      'Opponent FIFA rank', 'Host nation', 'Own FIFA rank']

TASKS = {
    'model21': {'target': 'goal_difference', 'features': F1, 'labels': L1, 'count': False},
    'model22': {'target': 'goals', 'features': F2, 'labels': L2, 'count': True},
}
# (mode, last N games, k pseudo-games)
PRIORS = ([('raw', 0, 0)]
          + [('grand', 0, k) for k in (1, 2, 4, 8)]
          + [('pre', n, k) for n in (5, 10, 15) for k in (1, 2, 4, 8, 16, 32, 64)])
FORECAST_ORIGINS = ['2026-06-24', '2026-06-28', '2026-07-01', '2026-07-05']


def save_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def holdout_cutoff(matches):
    # test set = last quarter of the matches, cut at a date so no day is split
    cumulative = matches.groupby('date').size().cumsum()
    last_development_date = cumulative.index[np.argmin(abs(cumulative.values - .75 * len(matches)))]
    return matches.loc[matches.date > last_development_date, 'date'].min()


def make_datasets(matches, team_matches, prior):
    task1, task2, _ = build_datasets(matches, team_matches, prior, F1, F2)
    return {'model21': task1, 'model22': task2}


def choose_prior(matches, team_matches, cutoff):
    rows = []
    for prior in PRIORS:
        datasets = make_datasets(matches, team_matches, prior)
        for key, task in TASKS.items():
            train = datasets[key][datasets[key].date < cutoff]
            models = candidates(not task['count'], task['count'])
            _, summary = cross_validate(models, train, task['features'], task['target'], not task['count'],
                                        names=['OLS', 'Ridge 10', 'Ridge 100', 'Ridge 1000'])
            rows.append({'task': key, 'mode': prior[0], 'N': prior[1], 'k': prior[2],
                         'best_linear': summary.index[0], 'cv_mae': float(summary.CV_MAE.iloc[0])})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / 'prior_selection.csv', index=False)

    # priors within 0.01 of the best cv score count as equal, then pre-tournament form wins
    preference = {'pre': 0, 'grand': 1, 'raw': 2}
    best = {}
    for key, group in table.groupby('task'):
        near_best = group[group.cv_mae <= group.cv_mae.min() + 0.01]
        choice = near_best.assign(order=near_best['mode'].map(preference)).sort_values(['order', 'cv_mae']).iloc[0]
        best[key] = (choice['mode'], int(choice['N']), int(choice['k']))
    return best, table


def prior_effect(matches, team_matches, cutoff, best):
    raw = make_datasets(matches, team_matches, ('raw', 0, 0))
    columns = {'model21': ['gf_gap', 'ga_gap'], 'model22': ['gf_mean', 'opponent_ga_mean']}
    rows = []
    for key, task in TASKS.items():
        chosen = make_datasets(matches, team_matches, best[key])[key]
        for label, frame in [('raw', raw[key]), ('chosen', chosen)]:
            development = frame[frame.date < cutoff]
            for column in columns[key]:
                data = development[[column, task['target']]].dropna()
                rows.append({'task': key, 'feature': column, 'prior': label,
                             'pearson_r': float(data[column].corr(data[task['target']])), 'n': len(data)})
    pd.DataFrame(rows).to_csv(RESULTS / 'prior_effect_eda.csv', index=False)


def design_correlations(matches, team_matches, cutoff, best):
    task1, _, _ = build_datasets(matches, team_matches, best['model21'], F1, F2)
    _, task2, team = build_datasets(matches, team_matches, best['model22'], F1, F2)
    lookup = team.set_index(['match_id', 'team'])
    task2 = task2.assign(opponent_clean_sheet_rate=[lookup.loc[(r.match_id, r.opponent), 'clean_sheet_rate']
                                                    for r in task2.itertuples()])
    development1, development2 = task1[task1.date < cutoff], task2[task2.date < cutoff]
    rows = [
        ('model21', 'shots_gap', 'corners_gap', development1.shots_gap.corr(development1.corners_gap)),
        ('model22', 'gf_mean', 'goal_sd', development2.gf_mean.corr(development2.goal_sd)),
        ('model22', 'opponent_clean_sheet_rate', 'opponent_ga_mean',
         development2.opponent_clean_sheet_rate.corr(development2.opponent_ga_mean)),
    ]
    pd.DataFrame(rows, columns=['task', 'variable_1', 'variable_2', 'pearson_r']).to_csv(
        RESULTS / 'design_correlations.csv', index=False)


def evaluate(key, frame, cutoff):
    task = TASKS[key]
    features, target, zero_intercept = task['features'], task['target'], not task['count']
    train, test = frame[frame.date < cutoff], frame[frame.date >= cutoff]
    models = candidates(zero_intercept, task['count'])

    folds, summary = cross_validate(models, train, features, target, zero_intercept)
    folds.to_csv(RESULTS / f'{key}_cv_folds.csv', index=False)
    summary.to_csv(RESULTS / f'{key}_cv_summary.csv')
    # ridge is the required linear model, cv only picks its penalty
    ridge_models = [name for name in summary.index if name.startswith('Ridge ')]
    selected = ridge_models[0]
    winner = summary.index[0]

    fitted, test_metrics = {}, []
    predictions = test[['match_id', 'date']].copy()
    predictions['actual'] = test[target].to_numpy()
    for name, build in models.items():
        fitted[name] = build().fit(train[features], train[target])
        predictions[name] = predict(fitted[name], test[features], name, zero_intercept)
        test_metrics.append({'model': name, **error_metrics(test[target], predictions[name])})
    predictions.to_csv(RESULTS / f'{key}_test_predictions.csv', index=False)

    selected_error = mean_absolute_error(test[target], predictions[selected])
    mean_error = mean_absolute_error(test[target], predictions['Mean baseline'])
    coefficients = pd.DataFrame({'feature': features, 'label': task['labels'],
                                 'coefficient': fitted[selected][-1].coef_})
    coefficients.to_csv(RESULTS / f'{key}_coefficients.csv', index=False)

    result = {
        'key': key, 'features': features, 'target': target,
        'selected_regression': selected, 'cv_winner': winner,
        'train_rows': len(train), 'test_rows': len(test), 'test_dates': int(test.date.nunique()),
        'cv_summary': summary.reset_index().to_dict('records'), 'test_metrics': test_metrics,
        'coefficients': coefficients.to_dict('records'),
        'test_mae_difference_vs_baseline': float(selected_error - mean_error),
        'mae_difference_95pct_date_bootstrap': date_bootstrap(
            test.date, test[target], predictions[selected], predictions['Mean baseline']),
        'negative_predictions': int((predictions[selected] < 0).sum()),
    }
    if task['count']:
        result['train_zero_goals'] = int(train[target].eq(0).sum())
        result['test_zero_goals'] = int(test[target].eq(0).sum())
    return result, train, test


def ols_inference(train, features, target, zero_intercept):
    prepare = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                            StandardScaler(with_mean=not zero_intercept))
    x = prepare.fit(train[features]).transform(train[features])
    y = train[target].to_numpy(float)
    vif = np.diag(np.linalg.inv(np.corrcoef(x, rowvar=False)))

    design = x if zero_intercept else np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    dof = len(y) - design.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * (residual @ residual) / dof)
    t = beta / se
    p = 2 * stats.t.sf(abs(t), dof)
    skip = 0 if zero_intercept else 1
    table = pd.DataFrame({'feature': features, 'vif': vif, 'ols_coef': beta[skip:], 'se': se[skip:],
                          't': t[skip:], 'p': p[skip:]})

    centre = 0 if zero_intercept else y.mean()
    r2 = 1 - (residual @ residual) / ((y - centre) ** 2).sum()
    terms = design.shape[1] - skip
    f_statistic = (r2 / terms) / ((1 - r2) / dof)
    summary = {'dev_rows': len(y), 'dof': dof, 'r2_dev': float(r2), 'f_stat': float(f_statistic),
               'f_p': float(stats.f.sf(f_statistic, terms, dof)), 'max_vif': float(vif.max())}
    return table, summary


def validation_residual_normality(models, selected, train, features, target, zero_intercept):
    residuals = []
    for block in validation_blocks(train.date.unique()):
        fit_part = train[train.date < block[0]]
        valid_part = train[train.date.isin(block)]
        model = models[selected]().fit(fit_part[features], fit_part[target])
        predicted = predict(model, valid_part[features], selected, zero_intercept)
        residuals += list(valid_part[target] - predicted)
    return float(stats.shapiro(residuals)[1])


def diagnostics(key, result, train, test):
    task = TASKS[key]
    features, target, zero_intercept = task['features'], task['target'], not task['count']
    selected = result['selected_regression']
    models = candidates(zero_intercept, task['count'])

    correlations = []
    for feature in features:
        data = train[[feature, target]].dropna()
        correlations.append({
            'feature': feature, 'pearson_r': float(data[feature].corr(data[target])),
            'spearman_rho': float(data[feature].corr(data[target], method='spearman')),
            'pearson_p': float(stats.pearsonr(data[feature], data[target])[1]), 'n': len(data)})
    (pd.DataFrame(correlations)
     .sort_values('pearson_r', key=lambda s: s.abs(), ascending=False)
     .to_csv(RESULTS / f'{key}_eda_correlations.csv', index=False))

    inference, summary = ols_inference(train, features, target, zero_intercept)
    inference.to_csv(RESULTS / f'{key}_ols_inference.csv', index=False)
    summary['shapiro_p_cv_residuals'] = validation_residual_normality(
        models, selected, train, features, target, zero_intercept)
    save_json(f'{key}_ols_summary.json', summary)

    rows = []
    for removed in [None] + features:
        kept = [feature for feature in features if feature != removed]
        model = models[selected]().fit(train[kept], train[target])
        predicted = predict(model, test[kept], selected, zero_intercept)
        rows.append({'removed': removed or 'none', 'test_MAE': float(mean_absolute_error(test[target], predicted))})
    pd.DataFrame(rows).to_csv(RESULTS / f'{key}_ablation.csv', index=False)


def robustness(datasets, results):
    rows = []
    for key, task in TASKS.items():
        frame = datasets[key]
        models = candidates(not task['count'], task['count'])
        names = {'mean': 'Mean baseline', 'ridge': results[key]['selected_regression'],
                 'best_cv': results[key]['cv_winner']}
        for origin in FORECAST_ORIGINS:
            train, test = frame[frame.date < origin], frame[frame.date >= origin]
            row = {'task': key, 'origin': origin, 'test_rows': len(test), 'test_matches': int(test.match_id.nunique()),
                   'winner_model': names['best_cv'], 'ridge_model': names['ridge']}
            for label, name in names.items():
                model = models[name]().fit(train[task['features']], train[task['target']])
                predicted = predict(model, test[task['features']], name, not task['count'])
                row[label] = float(mean_absolute_error(test[task['target']], predicted))
            for label in names:
                row[f'{label}_str'] = f'{row[label]:.3f}'
            rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / 'robustness_origins.csv', index=False)
    save_json('robustness.json', {'rows': rows})


def report_values(results, prior_table):
    def show(value):
        return f'{value:.3f}'.replace('-', '−')

    values = {}
    for key, result in results.items():
        table = prior_table[prior_table.task == key]
        best_by_mode = {'raw': table[table['mode'] == 'raw'].cv_mae.min(),
                        'grand': table[table['mode'] == 'grand'].cv_mae.min()}
        for n in (5, 10, 15):
            best_by_mode[f'pre{n}'] = table[(table['mode'] == 'pre') & (table.N == n)].cv_mae.min()
        values[key] = {
            'test': {row['model']: {metric: show(row[metric]) for metric in ['MAE', 'RMSE', 'R2']}
                     for row in result['test_metrics']},
            'cv': {row['model']: show(row['CV_MAE']) for row in result['cv_summary']},
            'prior_min': {name: show(value) for name, value in best_by_mode.items()},
            'selected': result['selected_regression'],
            'winner': result['cv_winner'],
        }
    save_json('report_values.json', values)


def data_audit(matches, team_matches, cutoff, best):
    save_json('data_audit.json', {
        'matches': len(matches), 'team_rows': len(team_matches), 'teams': int(team_matches.team.nunique()),
        'predictors_per_task': [len(F1), len(F2)], 'cutoff': cutoff, 'seed': SEED,
        'prior_choice': {key: list(value) for key, value in best.items()},
        'source_sha256': {str(path.relative_to(RAW)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(RAW.rglob('*.*'))},
        'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
    })


def main():
    for folder in (CLEAN, RESULTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)

    matches, team_matches = load_matches()
    cutoff = holdout_cutoff(matches)
    best, prior_table = choose_prior(matches, team_matches, cutoff)
    prior_effect(matches, team_matches, cutoff, best)
    design_correlations(matches, team_matches, cutoff, best)

    datasets = {key: make_datasets(matches, team_matches, best[key])[key] for key in TASKS}
    datasets['model21'].to_csv(CLEAN / 'regression_21.csv', index=False)
    datasets['model22'].to_csv(CLEAN / 'regression_22.csv', index=False)

    results, runs = {}, {}
    for key, task in TASKS.items():
        result, train, test = evaluate(key, datasets[key], cutoff)
        result['prior_choice'] = list(best[key])
        save_json(f'{key}.json', result)
        diagnostics(key, result, train, test)
        results[key] = result
        runs[key] = {**task, 'train': train, 'result': result}

    figures.make_all(runs, RESULTS, FIGURES)
    robustness(datasets, results)
    report_values(results, prior_table)
    data_audit(matches, team_matches, cutoff, best)

    for key, result in results.items():
        print(key, 'prior', result['prior_choice'], '| Ridge:', result['selected_regression'],
              '| CV winner:', result['cv_winner'])
        for row in result['test_metrics']:
            if row['model'] in ('Mean baseline', result['selected_regression'], result['cv_winner']):
                print(f"  {row['model']:22s} MAE {row['MAE']:.3f}  RMSE {row['RMSE']:.3f}  R2 {row['R2']:.3f}")


if __name__ == '__main__':
    main()
