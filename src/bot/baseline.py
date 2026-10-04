"""Baseline de référence (palier 2) : tendance SMA168, long / flat.

Règle (figée a priori, aucune optimisation, voir docs/baseline.md) :
    signal[t] = 1 si close[t] > SMA168(close)[t], sinon 0
    SMA168[t] = moyenne arithmétique des 168 clôtures t-167..t (bougie t incluse,
    clôturée). NaN si l'une de ces 168 bougies manque ou si l'historique est
    insuffisant ; le signal vaut alors NaN (position précédente conservée par
    le moteur).
Décision à la clôture de t, exécution à l'ouverture de t+1 (exec_delay=1),
coûts 0,15 % par côté appliqués à |Δposition|.

La moyenne est calculée par le même cœur à fenêtre finie que les features
(bot.features._rmean : somme dans un ordre fixe), donc identique en backtest et
en live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest import DEFAULT_COST_PER_SIDE, DEFAULT_EXEC_DELAY, BacktestResult, run_backtest
from bot.features import _rmean

SMA_WINDOW = 168  # bougies horaires = 7 jours, fixé a priori
COST_PER_SIDE = DEFAULT_COST_PER_SIDE  # 0,0015
EXEC_DELAY = DEFAULT_EXEC_DELAY  # 1


def _clean_close(df: pd.DataFrame) -> np.ndarray:
    c = df["close"].to_numpy(dtype=np.float64).copy()
    if "missing" in df.columns:
        c[df["missing"].to_numpy(dtype=bool)] = np.nan
    return c


def sma(df: pd.DataFrame, window: int = SMA_WINDOW) -> pd.Series:
    """SMA des clôtures sur [t-window+1, t] ; NaN si une bougie de la fenêtre manque."""
    return pd.Series(_rmean(_clean_close(df), window), index=df.index, name=f"sma{window}")


def baseline_signal(df: pd.DataFrame, window: int = SMA_WINDOW) -> pd.Series:
    """Position cible décidée à la clôture de t : 1.0 / 0.0, NaN si indisponible."""
    c = _clean_close(df)
    m = _rmean(c, window)
    out = np.where(c > m, 1.0, 0.0)
    out[np.isnan(c) | np.isnan(m)] = np.nan
    return pd.Series(out, index=df.index, name="baseline_signal")


def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, name="buy_and_hold_signal")


def run_baseline(df: pd.DataFrame, start=None, end=None, *, exec_delay: int = EXEC_DELAY,
                 cost_per_side: float = COST_PER_SIDE, cost_multiplier: float = 1.0) -> BacktestResult:
    """Baseline évaluée sur [start, end] ; signal calculé sur tout l'historique de df,
    position tenue avant start héritée (la stratégie tourne en continu)."""
    return run_backtest(baseline_signal(df), df["open"], start=start, end=end,
                        exec_delay=exec_delay, cost_per_side=cost_per_side,
                        cost_multiplier=cost_multiplier)


def run_buy_and_hold(df: pd.DataFrame, start=None, end=None, *,
                     cost_per_side: float = COST_PER_SIDE,
                     cost_multiplier: float = 1.0) -> BacktestResult:
    """Achat à l'ouverture de la première bougie évaluée, vente au marquage final
    (open suivant la dernière bougie évaluée s'il existe, sinon dernier open connu),
    coûts d'entrée et de sortie inclus."""
    return run_backtest(buy_and_hold_signal(df), df["open"], start=start, end=end,
                        exec_delay=1, cost_per_side=cost_per_side,
                        cost_multiplier=cost_multiplier, initial_position=0.0,
                        close_at_end=True)
