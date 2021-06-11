import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from functools import wraps

from . import utils
from . import performance as perf

DECIMAL_TO_BPS = 10000


def plot_regression_return_table(regression_return_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["Regression Return Mean"] = regression_return_data.mean()
    ic_summary_table["Regression Return Std."] = regression_return_data.std()
    ic_summary_table["Risk-Adjusted Regression Return"] = \
        regression_return_data.mean() / regression_return_data.std()
    t_stat, p_value = stats.ttest_1samp(regression_return_data, 0)
    ic_summary_table["t-stat(Regression Return)"] = t_stat
    ic_summary_table["p-value(Regression Return)"] = p_value
    ic_summary_table["Regression Return Skew"] = stats.skew(regression_return_data)
    ic_summary_table["Regression Return Kurtosis"] = stats.kurtosis(regression_return_data)

    print("Regression Return Analysis")
    utils.print_table(ic_summary_table.apply(lambda x: x.round(5)).T)


def plot_cumulative_top_minus_bottom(mean_ret_spread_quant,
                            period,
                            freq=None,
                            title=None,
                            ax=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    top_minus_bottom_returns = perf.cumulative_returns(mean_ret_spread_quant)

    top_minus_bottom_returns.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel='Cumulative Returns',
           title=("Cumulative Top Minus Bottom Return ({} Fwd Period)".format(period)
                  if title is None else title),
           xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax