"""Métriques de performance sur l'equity nette d'un backtest.

- Sharpe (définition GONOGO) : rendements journaliers (jours UTC) de l'equity
  nette, taux sans risque 0, sur toute la période évaluée, annualisé par √365.
  Equity de fin de jour = equity à la fin de la dernière bougie du jour (bougies
  groupées par jour UTC de leur open_time). Le premier jour est rapporté à
  l'equity initiale (1). Écart-type ddof=1. NaN si moins de 2 jours ou écart-type nul.
- Drawdown max : max_t (1 - E_t / max_{s<=t} E_s) sur l'equity par bougie,
  equity initiale 1 incluse dans le maximum courant. Valeur positive (0,25 = -25 %).
- Trades : nombre d'aller-retours (segments de position > 0) de la fenêtre.
- Win rate : part des trades à rendement net > 0.
- Profit factor : somme des rendements nets gagnants / |somme des perdants|
  (inf si aucune perte et au moins un gain, NaN si aucun trade).
- Exposition : part des bougies évaluées avec une position tenue > 0.
- Rendement total : E_final - 1. Rendement annualisé : E_final^(8760 / n_bougies) - 1
  (année de 365 jours de 24 bougies horaires).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bot.backtest import BacktestResult

BARS_PER_YEAR = 24 * 365


def daily_returns(equity: pd.Series, initial: float = 1.0) -> pd.Series:
    """Rendements journaliers (UTC) d'une equity par bougie indexée par open_time."""
    if len(equity) == 0:
        return pd.Series(dtype=np.float64)
    idx = equity.index
    if idx.tz is None:
        raise ValueError("index tz-aware (UTC) requis")
    days = idx.tz_convert("UTC").floor("D")
    eod = equity.groupby(days).last()
    prev = eod.shift(1)
    prev.iloc[0] = initial
    return eod / prev - 1.0


def sharpe_daily(equity: pd.Series, initial: float = 1.0) -> float:
    r = daily_returns(equity, initial)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not sd > 0:
        return float("nan")
    return float(r.mean()) / sd * math.sqrt(365.0)


def max_drawdown(equity: pd.Series, initial: float = 1.0) -> float:
    e = np.concatenate([[initial], np.asarray(equity, dtype=np.float64)])
    peak = np.maximum.accumulate(e)
    return float(np.max(1.0 - e / peak))


def win_rate(trade_returns) -> float:
    r = np.asarray(trade_returns, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    return float(np.mean(r > 0))


def profit_factor(trade_returns) -> float:
    r = np.asarray(trade_returns, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def exposure(position: pd.Series) -> float:
    p = np.asarray(position, dtype=np.float64)
    return float(np.mean(p > 0)) if len(p) else float("nan")


def total_return(equity: pd.Series, initial: float = 1.0) -> float:
    return float(equity.iloc[-1]) / initial - 1.0


def annualized_return(equity: pd.Series, initial: float = 1.0) -> float:
    n = len(equity)
    return (float(equity.iloc[-1]) / initial) ** (BARS_PER_YEAR / n) - 1.0


def summarize(res: BacktestResult) -> dict:
    """Toutes les métriques d'un résultat de backtest."""
    tr = res.trades["net_return"].to_numpy() if len(res.trades) else np.array([])
    return {
        "start": res.equity.index[0].isoformat(),
        "end": res.equity.index[-1].isoformat(),
        "end_mark_time": res.end_mark_time.isoformat(),
        "n_bars": int(len(res.equity)),
        "sharpe": sharpe_daily(res.equity),
        "max_drawdown": max_drawdown(res.equity),
        "trades": int(len(res.trades)),
        "win_rate": win_rate(tr),
        "profit_factor": profit_factor(tr),
        "exposure": exposure(res.position),
        "total_return": total_return(res.equity),
        "annualized_return": annualized_return(res.equity),
        "total_cost": float(res.cost.sum()),
        "final_equity": res.final_equity,
    }
