"""Vectorised OLS slope t-statistics for many simulated samples at once.

Every function takes arrays of shape (n_paths, n_obs) and runs one regression
y = a + b x + e per row, so thousands of samples are estimated without a loop.
"""

import numpy as np


def newey_west_lags(n_obs):
    """Common automatic lag rule: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))


def slope_tstats(y, x, nw_lags=None):
    """Return (beta, t_ols, t_nw) for the slope in y = a + b x + e, row by row.

    t_ols uses the classical homoskedastic standard error.
    t_nw  uses the Newey-West (1987) HAC standard error with Bartlett weights.
    """
    n_obs = y.shape[1]
    if nw_lags is None:
        nw_lags = newey_west_lags(n_obs)

    xd = x - x.mean(axis=1, keepdims=True)
    yd = y - y.mean(axis=1, keepdims=True)
    sxx = (xd**2).sum(axis=1)
    beta = (xd * yd).sum(axis=1) / sxx
    resid = yd - beta[:, None] * xd

    # Classical OLS standard error (2 estimated parameters)
    s2 = (resid**2).sum(axis=1) / (n_obs - 2)
    t_ols = beta / np.sqrt(s2 / sxx)

    # Newey-West: long-run variance of the score x_t * e_t
    score = xd * resid
    lrv = (score**2).sum(axis=1)
    for lag in range(1, nw_lags + 1):
        weight = 1.0 - lag / (nw_lags + 1.0)
        lrv += 2.0 * weight * (score[:, lag:] * score[:, :-lag]).sum(axis=1)
    t_nw = beta / np.sqrt(lrv / sxx**2)

    return beta, t_ols, t_nw
