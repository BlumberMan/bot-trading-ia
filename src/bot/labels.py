"""Labels.

Convention : signal calculé à la clôture de la bougie t, exécution à
l'ouverture de t+1, sortie à l'ouverture de t+1+H.
    y_t       = 1 si open[t+1+H] > open[t+1], sinon 0
    fwd_ret_t = log(open[t+1+H] / open[t+1])
NaN si une bougie de [t+1, t+1+H] est manquante ou hors des données.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 4  # H, en bougies 1h


def compute_labels(ohlcv: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Renvoie un DataFrame (index de ohlcv) avec colonnes `label` et `fwd_ret`.

    `label` est float64 (0.0 / 1.0 / NaN).
    """
    o = ohlcv["open"].to_numpy(dtype=np.float64)
    n = len(o)
    missing = np.isnan(ohlcv[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in ohlcv.columns:
        missing |= ohlcv["missing"].to_numpy(dtype=bool)
    o = np.where(missing, np.nan, o)

    entry = np.full(n, np.nan)
    exit_ = np.full(n, np.nan)
    if n > 1:
        entry[: n - 1] = o[1:]
    if n > 1 + horizon:
        exit_[: n - 1 - horizon] = o[1 + horizon:]

    # bougie manquante dans [t+1, t+1+H] ou hors données -> invalide
    bad = np.ones(n, dtype=bool)
    m = missing.astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(m)])
    t = np.arange(n)
    ok_range = t + 1 + horizon <= n - 1
    tt = t[ok_range]
    bad[ok_range] = (csum[tt + 2 + horizon] - csum[tt + 1]) > 0

    fwd = np.log(exit_ / entry)
    label = (exit_ > entry).astype(np.float64)
    fwd[bad] = np.nan
    label[bad] = np.nan
    return pd.DataFrame({"label": label, "fwd_ret": fwd}, index=ohlcv.index)
