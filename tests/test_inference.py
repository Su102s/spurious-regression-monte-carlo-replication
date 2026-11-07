"""Sanity checks. Run with:  python -m pytest tests/   (or  python tests/test_inference.py)"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgp import ar1_paths, simulate  # noqa: E402
from inference import slope_tstats  # noqa: E402


def test_slope_matches_lstsq():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 200))
    y = 0.3 * x + rng.standard_normal((5, 200))
    beta, t_ols, _ = slope_tstats(y, x)
    for i in range(5):
        X = np.column_stack([np.ones(200), x[i]])
        coef, res, *_ = np.linalg.lstsq(X, y[i], rcond=None)
        s2 = res[0] / (200 - 2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        assert np.isclose(beta[i], coef[1])
        assert np.isclose(t_ols[i], coef[1] / se)


def test_ar1_moments():
    rng = np.random.default_rng(1)
    x = ar1_paths(rng, 4000, 400, 0.9, 2.0)
    assert abs(x.var() - 2.0) < 0.1
    rho_hat = np.mean(x[:, 1:] * x[:, :-1]) / np.mean(x**2)
    assert abs(rho_hat - 0.9) < 0.01


def test_size_is_correct_without_persistence():
    """With an i.i.d. predictor the 5% test should reject about 5% of the time."""
    rng = np.random.default_rng(2)
    r_next, z = simulate(rng, 20_000, 360, 0.0, 0.0, 0.10)
    _, t_ols, _ = slope_tstats(r_next, z)
    assert abs(np.mean(np.abs(t_ols) > 1.96) - 0.05) < 0.008


def test_data_mining_matches_theory_without_persistence():
    """With K independent i.i.d. predictors, P(best |t| > 1.96) is about 1 - 0.95^K."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_data_mining import max_abs_t

    rng = np.random.default_rng(3)
    out_ols, _ = max_abs_t(rng, 10_000, 240, 0.0, 0.0, 10)
    for k in (1, 5, 10):
        assert abs(np.mean(out_ols[k - 1] > 1.96) - (1 - 0.95**k)) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
