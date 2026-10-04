"""Features : un seul cœur de calcul, partagé par le backtest et le live.

Règles :
- la feature à l'index t n'utilise que les bougies d'index <= t (bougie t clôturée) ;
- toutes les fenêtres sont finies (aucune EMA / ewm / mémoire infinie) ;
- toute fenêtre contenant une bougie manquante donne NaN (pas d'imputation) ;
- chaque valeur est calculée par une somme dans un ordre fixe sur sa propre
  fenêtre (pas d'algorithme glissant cumulatif), donc le résultat à t ne
  dépend que des valeurs de la fenêtre : live == batch bit à bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Nombre de bougies (t inclus) nécessaires à chaque feature.
SPANS: dict[str, int] = {
    "ret_1": 2,
    "ret_4": 5,
    "ret_24": 25,
    "ret_168": 169,
    "vol_24": 25,
    "vol_168": 169,
    "rsi_14": 15,
    "sma_dev_24": 24,
    "sma_dev_168": 168,
    "logvol_z_168": 168,
    "range_1": 1,
    "range_24": 24,
    "hour_sin": 0,
    "hour_cos": 0,
    "dow_sin": 0,
    "dow_cos": 0,
}
FEATURES: list[str] = list(SPANS)
MAX_SPAN = max(SPANS.values())  # 169
LOOKBACK = MAX_SPAN + 31  # 200 bougies : max des fenêtres + marge


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """x décalé de k vers le passé : out[i] = x[i-k] (NaN si i-k < 0)."""
    if k == 0:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[k:] = x[:-k]
    return out


def _rsum(x: np.ndarray, n: int) -> np.ndarray:
    """Somme sur x[i-n+1..i], accumulée dans un ordre fixe (k = 0..n-1)."""
    acc = x.astype(np.float64, copy=True)
    for k in range(1, n):
        acc = acc + _shift(x, k)
    if n > 1:
        acc[: n - 1] = np.nan
    return acc


def _rmean(x: np.ndarray, n: int) -> np.ndarray:
    return _rsum(x, n) / n


def _rstd(x: np.ndarray, n: int) -> np.ndarray:
    """Écart-type (ddof=1) sur x[i-n+1..i], en deux passes."""
    m = _rmean(x, n)
    acc = (x - m) ** 2
    for k in range(1, n):
        acc = acc + (_shift(x, k) - m) ** 2
    return np.sqrt(acc / (n - 1))


def _any_missing(missing: np.ndarray, span: int) -> np.ndarray:
    """True si une bougie de [i-span+1, i] manque ou est hors données."""
    if span == 0:
        return np.zeros(len(missing), dtype=bool)
    cnt = _rsum(missing.astype(np.float64), span)
    return ~(cnt == 0)  # NaN (début de série) -> True


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Cœur unique de calcul des features (backtest et live).

    Entrée : DataFrame indexé par open_time (UTC, pas horaire strict), colonnes
    open, high, low, close, volume et éventuellement `missing`. Une bougie est
    considérée manquante si `missing` est vrai ou si une valeur OHLCV est NaN.
    Sortie : DataFrame de mêmes index, colonnes FEATURES, float64.
    """
    idx = ohlcv.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("index attendu : DatetimeIndex tz-aware (UTC)")
    if len(idx) > 1 and not (np.diff(idx.asi8) == np.diff(idx.asi8)[0]).all():
        raise ValueError("index non régulier")
    if len(idx) > 1 and idx[1] - idx[0] != pd.Timedelta(hours=1):
        raise ValueError("pas horaire attendu")

    vals = {c: ohlcv[c].to_numpy(dtype=np.float64) for c in OHLCV_COLS}
    missing = np.isnan(np.column_stack([vals[c] for c in OHLCV_COLS])).any(axis=1)
    if "missing" in ohlcv.columns:
        missing |= ohlcv["missing"].to_numpy(dtype=bool)
    o, h, l, c, v = (np.where(missing, np.nan, vals[k]) for k in OHLCV_COLS)

    logc = np.log(c)
    r1 = logc - _shift(logc, 1)
    out: dict[str, np.ndarray] = {}
    for n in (1, 4, 24, 168):
        out[f"ret_{n}"] = logc - _shift(logc, n)
    for n in (24, 168):
        out[f"vol_{n}"] = _rstd(r1, n)

    gain = np.where(r1 > 0, _diff(c), 0.0)
    loss = np.where(r1 < 0, -_diff(c), 0.0)
    gain = np.where(np.isnan(r1), np.nan, gain)
    loss = np.where(np.isnan(r1), np.nan, loss)
    ag, al = _rmean(gain, 14), _rmean(loss, 14)
    tot = ag + al
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(tot > 0, 100.0 * ag / tot, 50.0)
    out["rsi_14"] = np.where(np.isnan(tot), np.nan, rsi)

    for n in (24, 168):
        out[f"sma_dev_{n}"] = c / _rmean(c, n) - 1.0

    lv = np.log1p(v)
    sd = _rstd(lv, 168)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["logvol_z_168"] = np.where(sd > 0, (lv - _rmean(lv, 168)) / sd, np.nan)

    rng = (h - l) / c
    out["range_1"] = rng
    out["range_24"] = _rmean(rng, 24)

    hour = idx.hour.to_numpy(dtype=np.float64)
    dow = idx.dayofweek.to_numpy(dtype=np.float64)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    for name, span in SPANS.items():
        bad = _any_missing(missing, span)
        out[name] = np.where(bad, np.nan, out[name]).astype(np.float64)

    return pd.DataFrame(out, index=idx, columns=FEATURES)


def _diff(x: np.ndarray) -> np.ndarray:
    return x - _shift(x, 1)


def compute_features_live(window: pd.DataFrame) -> pd.Series:
    """Features de la dernière bougie clôturée, à partir des LOOKBACK dernières.

    `window` doit contenir au moins LOOKBACK bougies consécutives ; seules les
    LOOKBACK dernières sont utilisées. Appelle le même cœur que le backtest.
    """
    if len(window) < LOOKBACK:
        raise ValueError(f"fenêtre de {len(window)} bougies < LOOKBACK={LOOKBACK}")
    w = window.iloc[-LOOKBACK:]
    return compute_features(w).iloc[-1]
