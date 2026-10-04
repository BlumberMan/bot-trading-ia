"""Features supplémentaires (exploration palier 3), même discipline que bot.features.

Chaque feature à l'index t n'utilise que les bougies d'index <= t (bougie t
clôturée), sur une fenêtre FINIE ; toute fenêtre contenant une bougie manquante
donne NaN. Calcul par sommes / max / min dans un ordre fixe sur la fenêtre
(réutilise les cœurs _shift, _rmean, _rstd, _any_missing de bot.features, sans
les modifier), donc intégrable plus tard dans src/ avec un test d'égalité
live/batch identique à celui du palier 1.

Formules (c = close, h = high, l = low, v = volume, r1 = log c_t - log c_{t-1}) :
    ret_72        = log c_t - log c_{t-72}                         span 73
    sma_dev_72    = c_t / mean(c_{t-71..t}) - 1                    span 72
    vol_ratio     = std(r1, 24) / std(r1, 168)                     span 169
    dist_max_168  = c_t / max(h_{t-167..t}) - 1                    span 168
    dist_min_168  = c_t / min(l_{t-167..t}) - 1                    span 168
    logvol_z_24   = (log1p v_t - mean(log1p v, 24)) / std(log1p v, 24)   span 24
    ret_24_z      = (log c_t - log c_{t-24}) / (std(r1, 168) * sqrt(24))  span 169
Décalage : aucun (valeur disponible à la clôture de t, décision à t, exécution t+1).
Spans <= 169 : LOOKBACK = 200 du palier 1 suffit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.features import OHLCV_COLS, _any_missing, _rmean, _rstd, _shift

EXTRA_SPANS: dict[str, int] = {
    "ret_72": 73,
    "sma_dev_72": 72,
    "vol_ratio": 169,
    "dist_max_168": 168,
    "dist_min_168": 168,
    "logvol_z_24": 24,
    "ret_24_z": 169,
}
EXTRA_FEATURES = list(EXTRA_SPANS)


def _rmax(x: np.ndarray, n: int) -> np.ndarray:
    acc = x.astype(np.float64, copy=True)
    for k in range(1, n):
        acc = np.maximum(acc, _shift(x, k))  # NaN propagé
    acc[: n - 1] = np.nan
    return acc


def _rmin(x: np.ndarray, n: int) -> np.ndarray:
    acc = x.astype(np.float64, copy=True)
    for k in range(1, n):
        acc = np.minimum(acc, _shift(x, k))
    acc[: n - 1] = np.nan
    return acc


def compute_extra(ohlcv: pd.DataFrame) -> pd.DataFrame:
    idx = ohlcv.index
    vals = {c: ohlcv[c].to_numpy(dtype=np.float64) for c in OHLCV_COLS}
    missing = np.isnan(np.column_stack([vals[c] for c in OHLCV_COLS])).any(axis=1)
    if "missing" in ohlcv.columns:
        missing |= ohlcv["missing"].to_numpy(dtype=bool)
    o, h, l, c, v = (np.where(missing, np.nan, vals[k]) for k in OHLCV_COLS)
    logc = np.log(c)
    r1 = logc - _shift(logc, 1)
    out: dict[str, np.ndarray] = {}
    out["ret_72"] = logc - _shift(logc, 72)
    out["sma_dev_72"] = c / _rmean(c, 72) - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vol_ratio"] = _rstd(r1, 24) / _rstd(r1, 168)
        out["dist_max_168"] = c / _rmax(h, 168) - 1.0
        out["dist_min_168"] = c / _rmin(l, 168) - 1.0
        lv = np.log1p(v)
        sd = _rstd(lv, 24)
        out["logvol_z_24"] = np.where(sd > 0, (lv - _rmean(lv, 24)) / sd, np.nan)
        out["ret_24_z"] = (logc - _shift(logc, 24)) / (_rstd(r1, 168) * np.sqrt(24.0))
    for name, span in EXTRA_SPANS.items():
        bad = _any_missing(missing, span) | ~np.isfinite(out[name])
        out[name] = np.where(bad, np.nan, out[name]).astype(np.float64)
    return pd.DataFrame(out, index=idx, columns=EXTRA_FEATURES)
