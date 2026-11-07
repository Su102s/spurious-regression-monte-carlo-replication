"""Data-generating process in the spirit of Ferson, Sarkissian & Simin (2003).

Returns contain a persistent, unobserved expected-return component Z*.
The researcher regresses returns on a predictor Z that is *independent* of Z*
but has a similar degree of persistence. Any "significant" slope is spurious.

    Z*_t     = rho_star * Z*_{t-1} + v_t        (true expected return)
    r_{t+1}  = Z*_t + u_{t+1}                   (realised return)
    Z_t      = rho * Z_{t-1} + w_t              (candidate predictor, independent)

Var(Z*) / Var(r) = R2_star, i.e. the share of return variance that the *true*
predictor would explain. Var(r) is normalised to 1.
"""

import numpy as np
from scipy.signal import lfilter


def ar1_paths(rng, n_paths, n_obs, rho, var):
    """Simulate stationary AR(1) paths with unconditional variance `var`.

    Each row is one path. The first observation is drawn from the stationary
    distribution, so no burn-in period is needed.
    """
    if var == 0:
        return np.zeros((n_paths, n_obs))
    innov_sd = np.sqrt(var * (1.0 - rho**2))
    shocks = rng.standard_normal((n_paths, n_obs)) * innov_sd
    shocks[:, 0] = rng.standard_normal(n_paths) * np.sqrt(var)
    # x_t = rho * x_{t-1} + shock_t, applied along each row
    return lfilter([1.0], [1.0, -rho], shocks, axis=1)


def simulate(rng, n_paths, n_obs, rho, rho_star, r2_star):
    """Return (r_next, z): returns r_{t+1} and lagged predictor Z_t, both (n_paths, n_obs)."""
    z_star = ar1_paths(rng, n_paths, n_obs, rho_star, r2_star)
    u = rng.standard_normal((n_paths, n_obs)) * np.sqrt(1.0 - r2_star)
    r_next = z_star + u
    z = ar1_paths(rng, n_paths, n_obs, rho, 1.0)
    return r_next, z
