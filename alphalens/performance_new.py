import pandas as pd
import numpy as np
import warnings

import empyrical as ep
from pandas.tseries.offsets import BDay
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from . import utils

def factor_regression_return(factor_data):
    """
    Computes regression_return between factor values and N period 
    forward returns for each period in the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    
    Returns
    -------
    regression_return : pd.DataFrame
        regression return between factor and
        provided forward returns.
    """

    def src_regression_return(group):
        f = group['factor']
        _regression_return = group[utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: OLS(x, add_constant(f)).fit().params[1])
        return _regression_return

    grouper = [factor_data.index.get_level_values('date')]
    regression_return = factor_data.groupby(grouper).apply(src_regression_return)

    return regression_return

# 2. alphaportfolio_return factor_returns