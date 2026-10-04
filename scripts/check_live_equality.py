"""Vérifie sur le vrai dataset (période dev) que le calcul live == le calcul batch.

Usage : .venv\\Scripts\\python scripts\\check_live_equality.py [n] [seed]
"""

from __future__ import annotations

import sys

import numpy as np

from bot.features import FEATURES, LOOKBACK, compute_features, compute_features_live
from bot.split import load_dataset

N_DEFAULT = 2000
SEED_DEFAULT = 20260104
TOL = 1e-9


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED_DEFAULT
    df = load_dataset()  # dev uniquement
    batch = compute_features(df)
    rng = np.random.default_rng(seed)
    ts = rng.choice(np.arange(LOOKBACK - 1, len(df)), size=n, replace=False)
    n_equal = 0
    n_bitexact = 0
    n_nan_rows = 0
    max_diff = 0.0
    for t in ts:
        live = compute_features_live(df.iloc[t - LOOKBACK + 1: t + 1]).to_numpy()
        ref = batch.iloc[t].to_numpy()
        nan_l, nan_r = np.isnan(live), np.isnan(ref)
        n_nan_rows += bool(nan_r.any())
        same_nan = (nan_l == nan_r).all()
        both = ~nan_r & ~nan_l
        d = float(np.max(np.abs(live[both] - ref[both]))) if both.any() else 0.0
        max_diff = max(max_diff, d)
        if same_nan and d <= TOL:
            n_equal += 1
        if same_nan and np.array_equal(live[both], ref[both]):
            n_bitexact += 1
    print(f"dataset dev : {df.index.min()} -> {df.index.max()}, LOOKBACK={LOOKBACK}, "
          f"{len(FEATURES)} features, seed={seed}")
    print(f"instants comparés        : {n}")
    print(f"instants égaux (tol {TOL:g}) : {n_equal}")
    print(f"instants égaux bit à bit : {n_bitexact}")
    print(f"instants avec >=1 NaN    : {n_nan_rows}")
    print(f"écart absolu max         : {max_diff:.3e}")
    print("RÉSULTAT :", "OK" if n_equal == n else "ÉCHEC")
    return 0 if n_equal == n else 1


if __name__ == "__main__":
    sys.exit(main())
