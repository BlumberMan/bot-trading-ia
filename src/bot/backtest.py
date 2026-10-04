"""Moteur de backtest minimal (long / flat, positions fractionnaires dans [0, 1]).

Conventions (réutilisées aux paliers 3 et 4) :
- `signal[t]` = position cible décidée à la CLÔTURE de la bougie t (p_t dans [0, 1]).
- Exécution : la cible décidée à t est tenue de l'ouverture de t+d à l'ouverture
  de t+d+1, d = `exec_delay` (défaut 1). Position tenue pendant la bougie t :
  h_t = cible la plus récente décidée à t-d ou avant.
- Rendement de la bougie t : open[t+1] / open[t] - 1, appliqué à h_t.
- Coûts : cost_per_side * cost_multiplier * |h_t - h_{t-1}|, prélevés (en
  fraction de l'equity) à l'ouverture de t, au moment de l'exécution.
- Equity : E_t = E_{t-1} * (1 - coût_t) * (1 + h_t * r_t), equity nette à la fin
  de la bougie t (= à l'ouverture de t+1). E = 1 juste avant la fenêtre évaluée.

Données manquantes (grille horaire complète, bougies manquantes = lignes NaN) :
- signal NaN à t (bougie manquante ou feature indisponible) : la cible
  précédente est conservée, aucun trade forcé. Avant le premier signal valide,
  la cible vaut 0.
- open NaN à t : impossible d'exécuter, la position précédente est tenue.
  La nouvelle cible est exécutée à la première ouverture disponible.
- Pendant un trou de prix, la position est tenue ; le P&L du trou
  (open[j] / open[i] - 1, i = dernière ouverture connue, j = première ouverture
  disponible après le trou) est réalisé sur la bougie j-1, juste avant la
  première ouverture disponible. L'equity reste plate pendant le trou.
- Dernière bougie de l'entrée : pas d'ouverture suivante, rendement 0 (marquage
  au dernier open connu). Aucune donnée postérieure n'est jamais lue.

Fenêtres d'évaluation : `start` / `end` délimitent les bougies évaluées. Les
positions sont calculées sur tout l'historique fourni (la stratégie tourne en
continu) ; par défaut la position tenue juste avant `start` est héritée sans coût
(`initial_position=None`). Ainsi des fenêtres contiguës chaînées donnent
exactement le backtest continu. `initial_position=0.0` force un départ à plat
(cas du buy & hold) ; `close_at_end=True` liquide la position au marquage final
(coût de sortie prélevé sur la dernière bougie).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_COST_PER_SIDE = 0.0015  # 0,10 % frais + 0,05 % spread/slippage
DEFAULT_EXEC_DELAY = 1

TRADE_COLUMNS = ["entry_time", "exit_time", "entry_price", "exit_price", "net_return",
                 "n_bars", "open_at_start", "open_at_end"]


@dataclass
class BacktestResult:
    equity: pd.Series  # equity nette en fin de bougie, base 1 avant la fenêtre
    position: pd.Series  # position tenue pendant chaque bougie
    bar_return: pd.Series  # rendement net de chaque bougie (E_t / E_{t-1} - 1)
    cost: pd.Series  # coût prélevé à chaque bougie (fraction de l'equity)
    trades: pd.DataFrame  # un trade = un segment de position > 0 (aller-retour)
    initial_position: float  # position tenue juste avant la fenêtre
    end_mark_time: pd.Timestamp  # horodatage de l'open utilisé comme marquage final

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else 1.0


def target_positions(signal: pd.Series, exec_delay: int = DEFAULT_EXEC_DELAY) -> np.ndarray:
    """Cible applicable à chaque bougie : dernier signal valide décidé à t-d ou avant."""
    if exec_delay < 1:
        raise ValueError("exec_delay >= 1 requis (une décision à la clôture de t ne peut "
                         "pas être exécutée avant l'ouverture de t+1)")
    s = signal.astype(np.float64)
    v = s.to_numpy()
    ok = ~np.isnan(v)
    if ((v[ok] < 0) | (v[ok] > 1)).any():
        raise ValueError("signal hors de [0, 1] (long / flat uniquement)")
    return s.ffill().shift(exec_delay).fillna(0.0).to_numpy()


def held_positions(target: np.ndarray, valid_open: np.ndarray, start_pos: float = 0.0) -> np.ndarray:
    """Position réellement tenue : la cible n'est exécutable que si l'open existe."""
    h = np.empty(len(target))
    prev = start_pos
    for t in range(len(target)):
        if valid_open[t]:
            prev = target[t]
        h[t] = prev
    return h


def run_backtest(signal: pd.Series, open_: pd.Series, *, start=None, end=None,
                 exec_delay: int = DEFAULT_EXEC_DELAY,
                 cost_per_side: float = DEFAULT_COST_PER_SIDE,
                 cost_multiplier: float = 1.0,
                 initial_position: float | None = None,
                 close_at_end: bool = False) -> BacktestResult:
    """Backtest d'une série de positions cibles sur les prix d'ouverture.

    `signal` et `open_` partagent le même index (grille complète, strictement
    croissante) ; `signal` est réindexé sur `open_`.
    """
    idx = open_.index
    if len(idx) == 0:
        raise ValueError("série de prix vide")
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("index strictement croissant requis")
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        d = np.diff(idx.asi8)
        if not (d == d[0]).all():
            raise ValueError("grille irrégulière : les bougies manquantes doivent être des lignes NaN")
    if cost_per_side < 0 or cost_multiplier < 0:
        raise ValueError("coûts négatifs interdits")
    sig = signal.reindex(idx)
    o = open_.to_numpy(dtype=np.float64)
    valid = ~np.isnan(o)
    n = len(idx)

    target = target_positions(sig, exec_delay)
    h_full = held_positions(target, valid)

    a = 0 if start is None else int(idx.searchsorted(start, side="left"))
    b = n - 1 if end is None else int(idx.searchsorted(end, side="right")) - 1
    if a > b:
        raise ValueError("fenêtre vide")

    prev = (h_full[a - 1] if a > 0 else 0.0) if initial_position is None else float(initial_position)
    if not 0.0 <= prev <= 1.0:
        raise ValueError("initial_position hors de [0, 1]")
    h = held_positions(target[a:b + 1], valid[a:b + 1], prev)

    # dernier open connu <= t (pour réaliser le P&L des trous)
    last_valid = np.where(valid, np.arange(n), -1)
    last_valid = np.maximum.accumulate(last_valid)

    k = cost_per_side * cost_multiplier
    m = b - a + 1
    cost = np.empty(m)
    raw = np.zeros(m)
    eq = np.empty(m)
    e = 1.0
    p_prev = prev
    for i in range(m):
        t = a + i
        cost[i] = k * abs(h[i] - p_prev)
        lv = last_valid[t]
        if t + 1 < n and valid[t + 1] and lv >= 0:
            raw[i] = h[i] * (o[t + 1] / o[lv] - 1.0)
        e = e * (1.0 - cost[i]) * (1.0 + raw[i])
        eq[i] = e
        p_prev = h[i]
    if close_at_end and h[-1] != 0.0:
        exit_cost = k * abs(h[-1])
        cost[-1] = 1.0 - (1.0 - cost[-1]) * (1.0 - exit_cost)  # coût combiné de la bougie
        eq[-1] = eq[-1] * (1.0 - exit_cost)
    growth = np.empty(m)
    growth[0] = eq[0]
    growth[1:] = eq[1:] / eq[:-1]

    # marquage final : open[b+1] s'il existe, sinon dernier open connu
    if b + 1 < n and valid[b + 1]:
        end_mark = b + 1
    else:
        end_mark = int(last_valid[b])
    win_idx = idx[a:b + 1]
    trades = _extract_trades(win_idx, h, prev, growth, o, last_valid, a, end_mark, close_at_end, idx)
    return BacktestResult(
        equity=pd.Series(eq, index=win_idx, name="equity"),
        position=pd.Series(h, index=win_idx, name="position"),
        bar_return=pd.Series(growth - 1.0, index=win_idx, name="bar_return"),
        cost=pd.Series(cost, index=win_idx, name="cost"),
        trades=trades,
        initial_position=prev,
        end_mark_time=idx[end_mark] if end_mark >= 0 else idx[b],
    )


def _extract_trades(win_idx, h, prev, growth, o, last_valid, a, end_mark, close_at_end, idx) -> pd.DataFrame:
    """Segments de position > 0. Rendement net = produit des facteurs d'equity
    du segment, coût d'entrée et coût de sortie (bougie de sortie) inclus."""
    rows = []
    m = len(h)
    i = 0
    in_pos_before = prev > 0
    while i < m:
        if h[i] > 0:
            s = i
            open_at_start = (s == 0 and in_pos_before)
            j = s
            while j < m and h[j] > 0:
                j += 1
            if j < m:  # sortie exécutée à l'ouverture de la bougie j (coût inclus)
                g = float(np.prod(growth[s:j + 1]))
                exit_time, exit_price, open_at_end = win_idx[j], o[a + j], False
                n_bars = j - s
            else:
                g = float(np.prod(growth[s:m]))
                exit_time = idx[end_mark] if end_mark >= 0 else win_idx[-1]
                exit_price = o[end_mark] if end_mark >= 0 else np.nan
                open_at_end = not close_at_end
                n_bars = m - s
            lv = last_valid[a + s]
            rows.append((win_idx[s], exit_time, o[lv] if lv >= 0 else np.nan, exit_price,
                         g - 1.0, n_bars, open_at_start, open_at_end))
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)
