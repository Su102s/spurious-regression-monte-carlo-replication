"""Monte Carlo experiment: how often does an *irrelevant* predictor look significant?

Writes results/rejection_rates.csv with the share of samples in which a
two-sided 5% t-test rejects H0: slope = 0, for OLS and Newey-West standard errors.

Usage:  python run_simulation.py            (full grid, ~a few minutes)
        python run_simulation.py --quick    (small grid for a fast check)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dgp import simulate  # noqa: E402
from inference import newey_west_lags, slope_tstats  # noqa: E402

# Persistence of both the predictor and the true expected return (rho = rho*)
RHOS = [0.0, 0.15, 0.5, 0.9, 0.95, 0.98, 0.99]
# Share of return variance explained by the true (unobserved) expected return
R2_STARS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.15]
# Sample lengths: roughly 5 years, 30 years and 80 years of monthly data
SAMPLE_SIZES = [60, 360, 960]
N_PATHS = 10_000
CRITICAL = 1.96  # two-sided 5% test, normal approximation
SEED = 20031


def run(rhos, r2_stars, sample_sizes, n_paths, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n_obs in sample_sizes:
        for rho in rhos:
            for r2 in r2_stars:
                r_next, z = simulate(rng, n_paths, n_obs, rho, rho, r2)
                _, t_ols, t_nw = slope_tstats(r_next, z)
                rows.append({
                    "T": n_obs,
                    "rho": rho,
                    "R2_star": r2,
                    "reject_ols": np.mean(np.abs(t_ols) > CRITICAL),
                    "reject_nw": np.mean(np.abs(t_nw) > CRITICAL),
                    "nw_lags": newey_west_lags(n_obs),
                    "n_paths": n_paths,
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small grid for a fast check")
    args = parser.parse_args()

    if args.quick:
        grid = ([0.0, 0.9, 0.99], [0.0, 0.10], [60, 360], 2_000)
    else:
        grid = (RHOS, R2_STARS, SAMPLE_SIZES, N_PATHS)

    start = time.time()
    df = run(*grid, seed=SEED)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    name = "rejection_rates_quick.csv" if args.quick else "rejection_rates.csv"
    df.to_csv(out / name, index=False, float_format="%.4f")
    print(f"Saved {out / name}  ({len(df)} cells, {time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
