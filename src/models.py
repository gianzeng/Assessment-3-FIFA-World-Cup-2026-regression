import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1402026
RIDGE_PENALTIES = [0.01, 0.1, 1, 10, 100, 1000]
BASELINES = ('Mean baseline', 'Median baseline')


def candidates(zero_intercept, count_target):
    # task 2.1 predicts a signed difference, so no intercept and no centring there
    intercept = not zero_intercept

    def linear(estimator):
        scaler = StandardScaler(with_mean=intercept)
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), scaler, estimator)

    def trees(estimator):
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), estimator)

    def forest(leaf):
        return lambda: trees(RandomForestRegressor(
            n_estimators=300, max_features=0.7, min_samples_leaf=leaf, random_state=SEED, n_jobs=1))

    def boosting(trees_count, depth):
        return lambda: trees(GradientBoostingRegressor(
            n_estimators=trees_count, max_depth=depth, learning_rate=0.03, loss='huber', random_state=SEED))

    models = {
        'Mean baseline': lambda: DummyRegressor(strategy='mean'),
        'Median baseline': lambda: DummyRegressor(strategy='median'),
        'OLS': lambda: linear(LinearRegression(fit_intercept=intercept)),
    }
    for penalty in RIDGE_PENALTIES:
        models[f'Ridge {penalty:g}'] = lambda penalty=penalty: linear(Ridge(alpha=penalty, fit_intercept=intercept))
    models['Huber'] = lambda: linear(HuberRegressor(epsilon=1.35, max_iter=3000))
    models['Lasso 0.1'] = lambda: linear(Lasso(alpha=0.1, fit_intercept=intercept, max_iter=20000))
    models['Elastic net 0.1'] = lambda: linear(ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=intercept, max_iter=20000))
    if count_target:
        models['Poisson GLM'] = lambda: linear(PoissonRegressor(alpha=0.3, max_iter=2000))
    models['Random forest'] = forest(3)
    models['Random forest leaf5'] = forest(5)
    models['Gradient boosting'] = boosting(100, 2)
    models['Gradient boosting d1'] = boosting(150, 1)
    models['Gradient boosting d3'] = boosting(100, 3)
    return models


def model_family(name):
    if name in BASELINES:
        return 'baseline'
    if name == 'OLS' or name == 'Huber' or name.startswith(('Ridge', 'Lasso', 'Elastic')):
        return 'linear'
    if name.startswith('Poisson'):
        return 'GLM'
    return 'ensemble'


def predict(model, x, name, zero_intercept):
    prediction = model.predict(x)
    # swapping the two teams has to flip the sign in task 2.1
    if zero_intercept and name not in BASELINES:
        return (prediction - model.predict(-x)) / 2
    return prediction


def error_metrics(actual, predicted):
    return {
        'MAE': float(mean_absolute_error(actual, predicted)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
        'R2': float(r2_score(actual, predicted)),
    }


def validation_blocks(dates):
    # expanding window over the second half of the dates
    dates = np.array(sorted(dates))
    return np.array_split(dates[len(dates) // 2:], 3)


def cross_validate(models, train, features, target, zero_intercept, names=None):
    rows = []
    for name, build in models.items():
        if names is not None and name not in names:
            continue
        for fold, block in enumerate(validation_blocks(train.date.unique()), 1):
            fit_part = train[train.date < block[0]]
            valid_part = train[train.date.isin(block)]
            assert fit_part.date.max() < valid_part.date.min()
            model = build().fit(fit_part[features], fit_part[target])
            predicted = predict(model, valid_part[features], name, zero_intercept)
            rows.append({
                'model': name, 'fold': fold, 'train_rows': len(fit_part), 'valid_rows': len(valid_part),
                'train_end': fit_part.date.max(), 'valid_start': valid_part.date.min(),
                'valid_end': valid_part.date.max(), **error_metrics(valid_part[target], predicted),
            })
    folds = pd.DataFrame(rows)

    def pooled(group):
        return pd.Series({'CV_MAE': np.average(group.MAE, weights=group.valid_rows),
                          'CV_MAE_sd': group.MAE.std(ddof=1)})

    summary = folds.groupby('model').apply(pooled, include_groups=False).sort_values('CV_MAE')
    return folds, summary


def date_bootstrap(dates, actual, predicted, baseline, repeats=2000):
    actual, predicted, baseline = np.asarray(actual), np.asarray(predicted), np.asarray(baseline)
    dates = np.asarray(dates)
    difference = np.abs(actual - predicted) - np.abs(actual - baseline)
    rng = np.random.default_rng(SEED)
    unique_dates = pd.unique(dates)
    means = []
    for _ in range(repeats):
        sample = rng.choice(unique_dates, len(unique_dates), replace=True)
        rows = np.concatenate([np.flatnonzero(dates == date) for date in sample])
        means.append(difference[rows].mean())
    return [float(value) for value in np.quantile(means, [.025, .975])]
