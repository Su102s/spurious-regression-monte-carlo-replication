"""Extension: data mining on top of spurious regression.

For each simulated return series, a researcher tries K independent candidate
predictors (none related to returns) and reports only the one with the largest
|t|. How often is that "best" predictor significant at 5%, and how large must
the critical value be to keep the false-positive rate at 5%?

Writes results/data_mining.csv.

Usage:  python run_data_mining.py           (full run, ~1-2 min)
        python run_data_mining.py --quick
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dgp import ar1_paths  # noqa: E402
from inference import slope_tstats  # noqa: E402

# (label, rho = rho*, R2*) — from "pure data mining" to "mining + spurious regression"
SCENARIOS = [
    ("No persistence (ρ = 0, R²* = 0)", 0.0, 0.0),
    ("ρ = 0.95, R²* = 0.05", 0.95, 0.05),
    ("ρ = 0.99, R²* = 0.10", 0.99, 0.10),
]
KS = [1, 2, 5, 10, 25, 50, 100]
N_OBS = 360
N_PATHS = 10_000
CRITICAL = 1.96
SEED = 20032


def max_abs_t(rng, n_paths, n_obs, rho, r2_star, k_max):
    """Running max |t| over k = 1..k_max candidate predictors for each return series.

    Returns two arrays of shape (k_max, n_paths): the max |t| among the first k
    predictors, for OLS and Newey-West t-statistics. All K predictors are tried
    on the *same* return series, exactly as a researcher mining one data set would.
    """
    z_star = ar1_paths(rng, n_paths, n_obs, rho, r2_star)
    r_next = z_star + rng.standard_normal((n_paths, n_obs)) * np.sqrt(1.0 - r2_star)

    best_ols = np.zeros(n_paths)
    best_nw = np.zeros(n_paths)
    out_ols = np.empty((k_max, n_paths))
    out_nw = np.empty((k_max, n_paths))
    for k in range(k_max):
        z = ar1_paths(rng, n_paths, n_obs, rho, 1.0)  # a fresh, independent candidate
        _, t_ols, t_nw = slope_tstats(r_next, z)
        best_ols = np.maximum(best_ols, np.abs(t_ols))
        best_nw = np.maximum(best_nw, np.abs(t_nw))
        out_ols[k], out_nw[k] = best_ols, best_nw
    return out_ols, out_nw


def run(scenarios, ks, n_obs, n_paths, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for label, rho, r2 in scenarios:
        out_ols, out_nw = max_abs_t(rng, n_paths, n_obs, rho, r2, max(ks))
        for k in ks:
            rows.append({
                "scenario": label, "rho": rho, "R2_star": r2, "K": k, "T": n_obs,
                "reject_ols": np.mean(out_ols[k - 1] > CRITICAL),
                "reject_nw": np.mean(out_nw[k - 1] > CRITICAL),
                # |t| a "best of K" predictor must exceed for a true 5% test
                "crit95_ols": np.quantile(out_ols[k - 1], 0.95),
                "crit95_nw": np.quantile(out_nw[k - 1], 0.95),
                "n_paths": n_paths,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_paths, ks = (2_000, [1, 5, 25]) if args.quick else (N_PATHS, KS)

    start = time.time()
    df = run(SCENARIOS, ks, N_OBS, n_paths, SEED)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    name = "data_mining_quick.csv" if args.quick else "data_mining.csv"
    df.to_csv(out / name, index=False, float_format="%.4f")
    print(f"Saved {out / name}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
