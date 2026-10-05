"""Agrégation des bougies 1 h en bougies 4 h et journalières (UTC).

Définition (reprise de experiments/agg4h.py, validée P3-V5) :
- Entrée : bougies 1 h sur une grille horaire complète et régulière, index tz-aware UTC
  (open_time). Une heure est manquante si `missing` est vrai ou si une valeur OHLCV est NaN.
- Bougie de k heures (k = 4 pour "4h", k = 24 pour "1d") d'horodatage T = open_time de sa
  1re heure, T aligné sur la grille UTC : 00/04/08/12/16/20 h en 4 h, 00:00 en journalier
  (jour UTC 00:00 -> 23:59). Heures lues : T, T+1h, ..., T+(k-1)h.
      open   = open(T)
      high   = max des k high (np.maximum appliqué de gauche à droite)
      low    = min des k low
      close  = close(T+(k-1)h)
      volume = (...((v(T) + v(T+1h)) + v(T+2h)) + ...) + v(T+(k-1)h)   (ordre fixe)
- Une bougie n'est émise que si ses k heures sont présentes dans l'entrée : une fenêtre qui
  commence ou finit au milieu d'une bougie n'émet pas cette bougie partielle.
- Si une seule des k heures est manquante, la bougie est missing=True et ses OHLCV valent NaN.
- La bougie T est clôturée à T + k h : elle n'est utilisable qu'à partir de là (décalage 0
  en bougies agrégées : la valeur de T ne lit que des heures < T + k h).

Cœur unique `_aggregate`, appelé en lot par `resample_ohlcv` et en direct par
`resample_ohlcv_live`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
RULES: dict[str, int] = {"4h": 4, "1d": 24}


def _hours_of(rule: str) -> int:
    if rule not in RULES:
        raise ValueError(f"rule inconnue : {rule!r} (attendu : {sorted(RULES)})")
    return RULES[rule]


def hourly_missing(df1: pd.DataFrame) -> np.ndarray:
    """Masque des heures manquantes : missing=True ou au moins une valeur OHLCV NaN."""
    m = np.isnan(df1[OHLCV].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in df1.columns:
        m |= df1["missing"].to_numpy(dtype=bool)
    return m


def _aggregate(df1: pd.DataFrame, k: int) -> pd.DataFrame:
    idx = df1.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("index tz-aware (UTC) requis")
    if len(idx) > 1 and not (np.diff(idx.to_numpy()) == np.timedelta64(1, "h")).all():
        raise ValueError("grille horaire complète et régulière requise")
    n = len(idx)
    utc = idx.tz_convert("UTC")
    on_hour = (utc.minute == 0) & (utc.second == 0) & (utc.microsecond == 0) & (utc.nanosecond == 0)
    if n and not np.asarray(on_hour).all():
        raise ValueError("open_time doit être sur l'heure pleine")
    miss = hourly_missing(df1)
    hrs = utc.hour.to_numpy()
    s = np.flatnonzero((hrs % k == 0) & (np.arange(n) + (k - 1) < n))
    o = df1["open"].to_numpy(dtype=np.float64)
    h = df1["high"].to_numpy(dtype=np.float64)
    l = df1["low"].to_numpy(dtype=np.float64)
    c = df1["close"].to_numpy(dtype=np.float64)
    v = df1["volume"].to_numpy(dtype=np.float64)
    hi, lo, vol, mk = h[s], l[s], v[s], miss[s]
    for j in range(1, k):
        hi = np.maximum(hi, h[s + j])
        lo = np.minimum(lo, l[s + j])
        vol = vol + v[s + j]
        mk = mk | miss[s + j]
    out = pd.DataFrame({"open": o[s], "high": hi, "low": lo, "close": c[s + (k - 1)], "volume": vol},
                       index=idx[s])
    out.loc[mk, OHLCV] = np.nan
    out["missing"] = mk
    out.index.name = "open_time"
    return out


def resample_ohlcv(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrégation en lot de bougies 1 h en bougies `rule` ("4h" ou "1d").

    Colonnes : open, high, low, close, volume, missing. Index : open_time UTC de la bougie agrégée.
    """
    return _aggregate(df_1h, _hours_of(rule))


def resample_ohlcv_live(hours: pd.DataFrame, rule: str, n_bars: int | None = None) -> pd.DataFrame:
    """Agrégation « en direct » : `hours` = dernières bougies 1 h CLÔTURÉES disponibles.

    Renvoie les bougies agrégées complètes (clôturées) contenues dans la fenêtre ; la bougie en
    cours (partielle) et une éventuelle bougie tronquée au début de la fenêtre sont ignorées.
    Si n_bars est donné, renvoie les n_bars dernières (ValueError s'il y en a moins).
    Même cœur que `resample_ohlcv`.
    """
    a = _aggregate(hours, _hours_of(rule))
    if n_bars is None:
        return a
    if len(a) < n_bars:
        raise ValueError(f"{len(a)} bougies {rule} clôturées < {n_bars}")
    return a.iloc[-n_bars:]


def close_time(open_time: pd.Timestamp | pd.DatetimeIndex, rule: str):
    """Instant à partir duquel la bougie est utilisable : open_time + k heures."""
    return open_time + pd.Timedelta(hours=_hours_of(rule))
