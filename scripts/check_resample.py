"""Vérifie l'agrégation 4 h / journalière sur le vrai dataset (période dev uniquement).

- nombre de bougies 4 h et 1 j, nombre de bougies manquantes ;
- égalité bit à bit resample_ohlcv(df, "4h") == experiments/agg4h.aggregate_4h(df) ;
- égalité live = lot sur n instants (4 h et 1 j).

Usage : .venv\\Scripts\\python scripts\\check_resample.py [n] [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bot.resample import OHLCV, close_time, resample_ohlcv, resample_ohlcv_live
from bot.split import DEV_END, load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import agg4h  # noqa: E402  (lecture seule)

N_DEFAULT = 2000
SEED_DEFAULT = 20261005
N_BARS = {"4h": 200, "1d": 60}
K = {"4h": 4, "1d": 24}
H1 = pd.Timedelta(hours=1)


def bitwise_diffs(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Nombre de cellules différentes (comparaison des octets, NaN == NaN si même motif binaire)."""
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        return -1
    n = 0
    for col in OHLCV:
        x = a[col].to_numpy(dtype=np.float64).view(np.uint64)
        y = b[col].to_numpy(dtype=np.float64).view(np.uint64)
        n += int((x != y).sum())
    n += int((a["missing"].to_numpy() != b["missing"].to_numpy()).sum())
    return n


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED_DEFAULT
    df = load_dataset()  # dev uniquement
    assert df.index.max() <= DEV_END, "garde : index au-delà de la période dev"
    print(f"dataset dev : {df.index.min()} -> {df.index.max()} ({len(df)} heures, "
          f"{int(df['missing'].sum())} heures missing) ; garde index max <= {DEV_END} : "
          f"{df.index.max() <= DEV_END} ; seed={seed}")
    ok = True
    batch = {}
    for rule in ("4h", "1d"):
        a = resample_ohlcv(df, rule)
        batch[rule] = a
        print(f"[{rule}] bougies : {len(a)} ; manquantes : {int(a['missing'].sum())} ; "
              f"première {a.index[0]} ; dernière {a.index[-1]} (clôture {close_time(a.index[-1], rule)})")
    ref = agg4h.aggregate_4h(df)
    d = bitwise_diffs(batch["4h"], ref)
    same = batch["4h"].equals(ref)
    print(f"[égalité agg4h] bougies src : {len(batch['4h'])} ; bougies experiments/agg4h : {len(ref)} ; "
          f"cellules différentes (bit à bit) : {d} ; DataFrame.equals : {same}")
    ok &= d == 0 and same
    rng = np.random.default_rng(seed)
    for rule in ("4h", "1d"):
        k, nb = K[rule], N_BARS[rule]
        need = k * nb + (k - 1)
        a = batch[rule]
        ct = close_time(a.index, rule)
        js = rng.choice(np.arange(need - 1, len(df)), size=n, replace=False)
        n_eq = 0
        n_last = 0
        for j in js:
            hours = df.iloc[j - need + 1: j + 1]
            live = resample_ohlcv_live(hours, rule, nb)
            end = df.index[j] + H1
            r = a.loc[ct <= end].iloc[-nb:]
            n_eq += int(live.equals(r) and bitwise_diffs(live, r) == 0)
            n_last += int(close_time(live.index[-1], rule) <= end)
        print(f"[live = lot {rule}] instants : {n} ; fenêtre {need} heures ; {nb} bougies comparées par instant ; "
              f"égaux bit à bit : {n_eq}/{n} ; dernière bougie live clôturée : {n_last}/{n}")
        ok &= n_eq == n and n_last == n
    print("RÉSULTAT :", "OK" if ok else "ÉCHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
