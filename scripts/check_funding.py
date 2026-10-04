"""Contrôle du funding sur données réelles (période dev uniquement).

- égalité features live == lot sur N instants tirés de la grille 1 h dev (seed fixe) ;
- couverture de la grille 1 h dev par année (part des bougies avec un taux disponible,
  et avec les 5 features définies).

Usage : .venv\\Scripts\\python scripts\\check_funding.py [n] [seed]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bot.funding import (FUNDING_FEATURES, FUNDING_LOOKBACK, H1, assert_no_holdout,
                         funding_features, funding_features_live, load_funding)
from bot.split import load_dataset

N_DEFAULT = 2000
SEED_DEFAULT = 20261005
TOL = 1e-12


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED_DEFAULT
    idx = load_dataset().index  # grille 1 h dev uniquement
    f = load_funding()  # règlements <= DEV_END uniquement
    assert_no_holdout(f)
    batch = funding_features(f, idx)
    cols = FUNDING_FEATURES + ["fr_age_h"]
    ft = pd.DatetimeIndex(f["fundingTime"]).as_unit("ns").asi8

    # instants tirés parmi les bougies dev où au moins un règlement est disponible
    eligible = np.flatnonzero(batch["fr_cur"].notna().to_numpy())
    rng = np.random.default_rng(seed)
    ts = rng.choice(eligible, size=n, replace=False)
    n_equal = n_bitexact = n_nan = 0
    max_diff = 0.0
    for i in ts:
        t = idx[i]
        p = int(np.searchsorted(ft, (t + H1).as_unit("ns").value, side="right"))
        # fenêtre live : les FUNDING_LOOKBACK + 10 derniers règlements connus, plus
        # 3 règlements postérieurs à t + 1 h (que le calcul live doit ignorer)
        recent = f.iloc[max(0, p - FUNDING_LOOKBACK - 10): p + 3]
        live = funding_features_live(recent, t)[cols].to_numpy(dtype=np.float64)
        ref = batch.iloc[i][cols].to_numpy(dtype=np.float64)
        nl, nr = np.isnan(live), np.isnan(ref)
        n_nan += bool(nr.any())
        same_nan = bool((nl == nr).all())
        both = ~nl & ~nr
        d = float(np.max(np.abs(live[both] - ref[both]))) if both.any() else 0.0
        max_diff = max(max_diff, d)
        n_equal += same_nan and d <= TOL
        n_bitexact += same_nan and np.array_equal(live[both], ref[both])

    print(f"grille dev : {idx.min()} -> {idx.max()} ({len(idx)} bougies)")
    print(f"règlements dev : {len(f)} ({f['fundingTime'].iloc[0]} -> {f['fundingTime'].iloc[-1]})")
    print(f"colonnes comparées : {cols}, seed={seed}")
    print(f"instants comparés        : {n} (parmi {len(eligible)} bougies avec un taux disponible)")
    print(f"instants égaux (tol {TOL:g}) : {n_equal}")
    print(f"instants égaux bit à bit : {n_bitexact}")
    print(f"instants avec >=1 NaN    : {n_nan}")
    print(f"écart absolu max         : {max_diff:.3e}")
    ok = n_equal == n

    print("-" * 60)
    print("couverture grille 1 h dev par année")
    print(f"{'année':>6} {'bougies':>8} {'taux dispo':>11} {'%':>8} {'5 features':>11} {'%':>8} {'age_h max':>10}")
    for y, g in batch.groupby(batch.index.year):
        a = int(g["fr_cur"].notna().sum())
        b = int(g[FUNDING_FEATURES].notna().all(axis=1).sum())
        amax = g["fr_age_h"].max()
        print(f"{y:>6} {len(g):>8} {a:>11} {100 * a / len(g):>7.2f}% {b:>11} {100 * b / len(g):>7.2f}% "
              f"{amax:>10.4f}")
    a = int(batch["fr_cur"].notna().sum())
    b = int(batch[FUNDING_FEATURES].notna().all(axis=1).sum())
    print(f"{'total':>6} {len(batch):>8} {a:>11} {100 * a / len(batch):>7.2f}% {b:>11} "
          f"{100 * b / len(batch):>7.2f}%")
    print("RÉSULTAT :", "OK" if ok else "ÉCHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
