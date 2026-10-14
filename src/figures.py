import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from scipy import stats

from models import SEED, model_family

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'model21': '#244c69', 'model22': '#ae5534'}
FAMILY_COLORS = {'baseline': '#9a9a9a', 'linear': '#244c69', 'GLM': '#5b8c5a', 'ensemble': '#ae5534'}
TITLES = {'model21': 'Task 2.1', 'model22': 'Task 2.2'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7.5, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'savefig.dpi': 420,
})


def save(fig, folder, name):
    fig.savefig(folder / f'{name}.png')
    fig.savefig(folder / f'{name}.pdf')
    plt.close(fig)


def scatter_of_strongest_predictors(tasks, results, folder):
    fig, axes = plt.subplots(2, 3, figsize=(5.8, 2.55))
    for row, (key, task) in enumerate(tasks.items()):
        train = task['train']
        top = pd.read_csv(results / f'{key}_eda_correlations.csv').head(3)
        for ax, feature in zip(axes[row], top.feature):
            data = train[[feature, task['target']]].dropna()
            jitter = np.random.default_rng(SEED).normal(0, .07, len(data)) if task['count'] else 0
            ax.scatter(data[feature], data[task['target']] + jitter, s=14, alpha=.65, color=COLORS[key],
                       edgecolors='white', linewidth=.3)
            slope, intercept, r, _, _ = stats.linregress(data[feature], data[task['target']])
            x = np.array([data[feature].min(), data[feature].max()])
            ax.plot(x, intercept + slope * x, color='black', lw=1)
            ax.set(xlabel=task['labels'][task['features'].index(feature)],
                   ylabel='Team goals' if task['count'] else 'Goal difference',
                   title=f'r = {r:.2f}, n = {len(data)}')
            ax.grid(alpha=.12)
    fig.tight_layout()
    save(fig, folder, 'eda_scatter')


def correlations_and_collinearity(tasks, results, folder):
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True,
                                      gridspec_kw={'width_ratios': [1.5, 1], 'wspace': .06})
    positions, names, offset = [], [], 0
    for key, task in tasks.items():
        correlations = pd.read_csv(results / f'{key}_eda_correlations.csv').sort_values('pearson_r')
        vif = pd.read_csv(results / f'{key}_ols_inference.csv').set_index('feature').vif
        labels = dict(zip(task['features'], task['labels']))
        y = np.arange(len(correlations)) + offset
        left.barh(y - .17, correlations.pearson_r, height=.34, color=COLORS[key], label=f'{TITLES[key]} Pearson')
        left.barh(y + .17, correlations.spearman_rho, height=.34, color=COLORS[key], alpha=.45,
                  label=f'{TITLES[key]} Spearman')
        right.barh(y, vif.loc[correlations.feature].to_numpy(), height=.6, color=COLORS[key])
        positions += list(y)
        names += [labels[feature] for feature in correlations.feature]
        offset += len(correlations) + 1
    left.set_yticks(positions, names, fontsize=7.5)
    left.axvline(0, color='gray', lw=.7)
    left.set_xlabel('Correlation with the target')
    left.grid(axis='x', alpha=.13)
    right.axvline(5, color='gray', lw=.8, ls='--')
    right.set(xlabel='Variance inflation factor', xlim=(0, 6.5))
    right.grid(axis='x', alpha=.13)
    right.tick_params(left=False)
    fig.legend(*left.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7, frameon=False,
               columnspacing=1.2, handlelength=1.4)
    fig.subplots_adjust(left=.29, right=.98, top=.97, bottom=.25)
    save(fig, folder, 'feature_eda')


def model_comparison(tasks, results, folder):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.85))
    for ax, (key, task) in zip(axes, tasks.items()):
        cv = pd.read_csv(results / f'{key}_cv_summary.csv').set_index('model').CV_MAE
        test = {row['model']: row['MAE'] for row in task['result']['test_metrics']}
        order = list(cv.sort_values(ascending=False).index)
        y = np.arange(len(order))
        colors = [FAMILY_COLORS[model_family(name)] for name in order]
        for position, name in zip(y, order):
            ax.plot([cv[name], test[name]], [position, position], color='#cccccc', lw=.8, zorder=1)
        ax.scatter([cv[name] for name in order], y, c=colors, s=22, zorder=2)
        ax.scatter([test[name] for name in order], y, facecolors='white', edgecolors=colors, s=22,
                   linewidth=1.2, zorder=2)
        ax.set_yticks(y, order, fontsize=7)
        ax.set(xlabel='MAE (goals)', title=TITLES[key])
        ax.grid(axis='x', alpha=.13)
    handles = [Line2D([], [], marker='o', ls='', color=color, label=name, markersize=4.5)
               for name, color in FAMILY_COLORS.items()]
    handles += [Line2D([], [], marker='o', ls='', color='#555', label='filled = chronological CV', markersize=4.5),
                Line2D([], [], marker='o', ls='', mfc='white', mec='#555', label='hollow = later matches',
                       markersize=4.5)]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save(fig, folder, 'model_comparison')


def ridge_coefficients(tasks, results, folder):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 1.95))
    for ax, (key, task) in zip(axes, tasks.items()):
        coefficients = pd.DataFrame(task['result']['coefficients'])
        ols = pd.read_csv(results / f'{key}_ols_inference.csv')
        order = np.argsort(coefficients.coefficient.values)
        y = np.arange(len(order))
        ax.barh(y, coefficients.coefficient.iloc[order], color=COLORS[key], height=.6)
        ax.errorbar(ols.ols_coef.iloc[order], y, xerr=1.96 * ols.se.iloc[order], fmt='o', color='black', ms=2.5,
                    lw=.9, capsize=2)
        ax.set_yticks(y, coefficients.label.iloc[order], fontsize=7.5)
        ax.axvline(0, c='gray', lw=.7)
        ax.set(xlabel='Goals per development SD', title=f'{TITLES[key]} coefficients')
        ax.grid(axis='x', alpha=.13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS['model21']),
               Line2D([], [], marker='o', ls='-', color='black', markersize=2.5)]
    fig.legend(handles, ['Ridge (standardised)', 'OLS coefficient ±1.96 SE'], loc='lower center', ncol=2,
               fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save(fig, folder, 'coefficients')


def make_all(tasks, results, folder):
    scatter_of_strongest_predictors(tasks, results, folder)
    correlations_and_collinearity(tasks, results, folder)
    model_comparison(tasks, results, folder)
    ridge_coefficients(tasks, results, folder)
