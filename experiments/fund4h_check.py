r"""Funding sur la grille 4 h (brief P3-B7, A2) : règle, anti-fuite et direct = lot (synthétique, b7.funding_selftest),
direct = lot sur les données dev réelles, couverture par fold. N'est pas un essai.

Usage : .venv\Scripts\python experiments\fund4h_check.py
Sorties : experiments/results/fund4h_check.txt / .json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg4h  # noqa: E402
import b7  # noqa: E402
import harness  # noqa: E402
from bot import funding as bf  # noqa: E402
from bot.split import H1, make_folds, split_fold  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    log = harness.Tee()
    log("=== 1. Tests sur données SYNTHÉTIQUES (b7.funding_selftest) ===")
    b7.funding_selftest(log)
    log("\n=== 2. Données dev réelles ===")
    df1 = harness.load_dev(log)
    fund = b7.load_fund(log)
    df4 = agg4h.aggregate_4h(df1)
    G = b7.funding_4h(fund, df1.index, df4.index)
    assert G.index.max() <= harness.MAX_INDEX
    ft = fund["fundingTime"]
    late = int(((ft - ft.dt.floor("h")) > pd.Timedelta(0)).sum())
    log(f"règlements {len(fund)} ({ft.min()} -> {ft.max()}) ; horodatés après l'heure pile (gigue, utilisés à la "
        f"clôture 4 h suivante) : {late}")
    age = G["fr_age_h"].dropna()
    log(f"fr_age_h aux bougies 4 h (heures entre le règlement utilisé et la clôture) : valeurs arrondies "
        f"{dict(age.round(0).value_counts().sort_index().astype(int))}")
    fu = G[b7.FU].notna().all(axis=1)
    first = G.index[fu.to_numpy()][0]
    log(f"bougies 4 h avec fu défini : {int(fu.sum())}/{len(fu)} ; première {first}")
    # direct = lot sur 2000 bougies réelles
    rng = np.random.default_rng(20261005)
    i0 = int(df4.index.searchsorted(first))
    pts = np.sort(rng.choice(np.arange(i0, len(df4)), 2000, replace=False))
    neq = 0
    for i in pts:
        T = df4.index[i]
        known = fund[ft <= T + 4 * H1].tail(bf.FUNDING_LOOKBACK)
        neq += int(np.array_equal(b7.funding_4h_live(known, T).to_numpy(), G.iloc[i].to_numpy(), equal_nan=True))
    log(f"direct = lot (données dev, 2000 bougies 4 h tirées seed 20261005, 90 derniers règlements connus à T+4h) : "
        f"{neq}/2000 bit à bit")
    assert neq == 2000
    # couverture par fold
    log("\n=== 3. Couverture par fold (part des bougies 4 h avec les 4 features fu définies) ===")
    log("fold | train (split_fold horizon 123 h) : bougies 4 h, avec fu, part | test : bougies 4 h, avec fu, part")
    cov = []
    for f in make_folds(horizon=agg4h.horizon_hours(30)):
        tr, te = split_fold(df4, f, horizon=agg4h.horizon_hours(30))
        a, b = fu.loc[tr.index], fu.loc[te.index]
        cov.append({"fold": f.k, "train_bars": len(a), "train_fu": int(a.sum()), "test_bars": len(b),
                    "test_fu": int(b.sum())})
        log(f"{f.k:>4} | {len(a):>6} {int(a.sum()):>6} {a.mean():.4f} | {len(b):>5} {int(b.sum()):>5} {b.mean():.4f}")
    out = {"n_settlements": len(fund), "late_settlements": late, "first_fu_bar": first, "live_eq": neq,
           "coverage": cov}
    with open(harness.RESULTS / "fund4h_check.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(harness.RESULTS / "fund4h_check.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
