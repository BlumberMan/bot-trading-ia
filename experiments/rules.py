r"""Cycle « règles » (brief P3-B9) : moteur commun, familles A / B20 / B55 / C, évaluation walk-forward.

Usage :
    .venv\Scripts\python experiments\rules.py --selftest                       auto-tests sur données synthétiques
    .venv\Scripts\python experiments\rules.py experiments\configs\E48.json     un essai (n'écrit PAS au registre)
    .venv\Scripts\python experiments\rules.py experiments\configs\E48.json --out CHEMIN   (relance de reproductibilité)
    .venv\Scripts\python experiments\rules.py --register E48                   ajoute la ligne de E48 au registre

AUCUN TUNING : tous les paramètres sont fixés a priori par le brief P3-B9 (conventions publiées), aucun n'est
choisi sur nos données. Pas de seed (règles déterministes) ; les seeds n'interviennent que dans les tests du veto.

DONNÉES : bot.split.load_dataset() par défaut (période dev seule), garde « index max <= 2025-09-30 23:00 UTC »
affichée (harness.load_dev). Données réservées 2025-10-01 -> 2026-08-31 jamais lues.

MOTEUR COMMUN (définitions du brief, point 1)
- Grille 1 h complète (bougies manquantes = lignes NaN). Décision à la clôture de la bougie 1 h t, exécution à
  l'open de t+1 (bot.backtest, exec_delay=1 ; si l'open est manquant, première ouverture disponible, comme
  bot.backtest). Positions long / flat (0 ou 1).
- Signal journalier : la bougie 1 j D (bot.resample.resample_ohlcv(df, "1d")) clôture à D + 24 h ; sa valeur est
  posée sur la bougie 1 h D + 23 h (décision à sa clôture). Signal 4 h : bougie 4 h T posée sur la bougie 1 h T + 3 h.
  Aux autres heures, aucun signal (NaN).
- Bougie 1 j ou 4 h manquante (missing) -> signal NaN -> aucune nouvelle entrée.
- Entrée : signal == 1, position à plat, entrée autorisée (voir ré-entrée) et barrière b finie. E = open d'exécution
  (bougie j = première ouverture disponible >= t + 1).
- Sortie triple barrière, contrôlée à chaque clôture 1 h k >= j (bougie d'entrée comprise) :
    haute  : close_k >= E * exp(+b) ;  basse : close_k <= E * exp(-b) ;
    temps  : k - j + 1 >= 120 (décision à la clôture de la 120e bougie détenue, j + 119 ; bougies calendaires).
  Ordre d'examen à une même clôture : haute, basse, temps (haute et basse sont exclusives). Close manquant : pas de
  contrôle de prix à cette bougie (le compteur de temps avance). Sortie exécutée à la première ouverture disponible
  x >= k + 1.
- Barrière fixe : b = log(1,027). Barrière vol : b = max(sigma30 * sqrt(5), log(1,009)), sigma30 = écart-type
  (ddof=1) des log-rendements journaliers close à close log(C_D / C_{D-1}) des 30 derniers jours clôturés au moment
  du signal (D-29..D, D = dernier jour clôturé). b est figée à la décision d'entrée pour toute la durée du trade.
- Ré-entrée : un signal n'est pris qu'à une clôture d'indice >= x (clôture postérieure à l'exécution de la sortie,
  qui a lieu à l'ouverture de x). Aucun pyramidage : un signal reçu en position est ignoré.
- Décision prise à la dernière clôture dev (bougie 2025-09-30 23:00, clôture = HOLDOUT_START) : jamais exécutée
  (pas d'open suivant dans les données dev) ; vérifié à chaque essai (backtest invariant à cette décision).
- Anti-fuite : la cible à t ne dépend que des données <= clôture de t (test --selftest : données modifiées après
  t0 -> cibles <= t0 et sorties déjà décidées inchangées ; signaux des familles inchangés).

FENÊTRES ET VALEURS MANQUANTES (précision d'implémentation, fixée avant tout résultat)
- Fenêtres glissantes (max des high, SMA, sigma30) : calculées sur les valeurs disponibles de la fenêtre calendaire
  (bougies manquantes ignorées) ; NaN tant que la fenêtre calendaire complète n'est pas dans l'historique, ou si
  aucune valeur (sigma30 : moins de 2) n'est disponible.
- Valeurs ponctuelles (close de la bougie du signal, close_{d-L} de la famille A) : manquantes -> signal NaN.

FAMILLES (point 2 du brief, paramètres a priori)
- A  (journalier) : signal = 1 si au moins 2 des 3 rendements close_d / close_{d-L} - 1 sont > 0, L = 20, 60, 120 j.
- B20 (journalier) : signal = 1 si close_d > max(high_{d-20..d-1}).  B55 : idem sur 55 jours.
- C  (4 h, filtre journalier) : signal = 1 si close du dernier jour clôturé D > SMA50 des closes journaliers
  (D-49..D) ET close_4h(T) > max(high des 20 bougies 4 h T-20..T-1). D = jour UTC de (T + 4 h) moins 1 jour.
- Chaque famille avec la barrière fixe puis la barrière vol : E48 A fixe, E49 A vol, E50 B20 fixe, E51 B20 vol,
  E52 B55 fixe, E53 B55 vol, E54 C fixe, E55 C vol.

ÉVALUATION (point 4) : une série de cibles continue 2019-01-01 -> 2025-09-30 (2019-2020 = préchauffage), un
backtest bot.backtest (exec_delay=1, 0,0015/côté, x1) par fold de test (fenêtres [test_start, test_end] de
bot.split.make_folds, position héritée sans coût, comme scripts/run_baseline.py) et sur la concaténation
2021-01-01 -> 2025-09-30 23:00 (comme experiments/harness.py) ; baseline / B&H recalculés par harness.baseline_block
et comparés à results/baseline_metrics.json (<= 1e-9, sinon arrêt). Informatif : Sharpe 2019-2020 (préchauffage).
Sorties : results/<ID>.json / .txt, signaux results/rules_<ID>/signal_1h.parquet, trades results/rules_<ID>/trades.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
sys.path.insert(0, str(EXP))

import harness  # noqa: E402
from bot.backtest import run_backtest  # noqa: E402
from bot.data import sha256_file  # noqa: E402
from bot.metrics import daily_returns, summarize  # noqa: E402
from bot.resample import resample_ohlcv  # noqa: E402
from bot.split import DEFAULT_DATASET, make_folds  # noqa: E402

RES = EXP / "results"
REGISTRE = EXP / "REGISTRE.md"
H1 = pd.Timedelta(hours=1)
COST_PER_SIDE = 0.0015
EXEC_DELAY = 1
H_TIME = 120
B_FIXED = math.log(1.027)
B_FLOOR = math.log(1.009)
SIGMA_WIN = 30
SIGMA_SCALE = math.sqrt(5.0)
FAMILIES = ("A", "B20", "B55", "C")
A_LAGS = (20, 60, 120)
DONCHIAN = {"B20": 20, "B55": 55}
C_SMA = 50
C_BREAK = 20
CONCAT_START = harness.CONCAT_START
CONCAT_END = harness.CONCAT_END
IS_START = pd.Timestamp("2019-01-01 00:00", tz="UTC")
IS_END = pd.Timestamp("2020-12-31 23:00", tz="UTC")
REGISTRY_FIRST, REGISTRY_LAST = 48, 55
CODE_PATHS = ["src", "experiments/rules.py", "experiments/rules_report.py", "experiments/harness.py",
              "experiments/configs"]


# ------------------------------------------------------------------ fenêtres

def win_stat(x: np.ndarray, w: int, how: str, lag: int = 0, min_valid: int = 1) -> np.ndarray:
    """Statistique sur x[i-lag-w+1 .. i-lag], NaN ignorés ; NaN si la fenêtre déborde du début ou trop peu de valeurs."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    r = s.rolling(w, min_periods=min_valid)
    if how == "max":
        v = r.max()
    elif how == "mean":
        v = r.mean()
    elif how == "std":
        v = r.std(ddof=1)
    else:
        raise ValueError(how)
    v = v.shift(lag).to_numpy().copy()
    v[: min(len(v), w + lag - 1)] = np.nan
    return v


def sigma30(close_d: np.ndarray) -> np.ndarray:
    """sigma des log-rendements journaliers r_D = log(C_D / C_{D-1}), D-29..D, ddof=1 ; NaN si D < 30."""
    c = np.asarray(close_d, dtype=np.float64)
    lr = np.full(len(c), np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    v = win_stat(lr, SIGMA_WIN, "std", lag=0, min_valid=2)
    v[: min(len(v), SIGMA_WIN)] = np.nan
    return v


def decision_positions(df1: pd.DataFrame, agg_index: pd.DatetimeIndex, k_hours: int) -> np.ndarray:
    pos = df1.index.get_indexer(agg_index + (k_hours - 1) * H1)
    if (pos < 0).any():
        raise ValueError("bougie agrégée sans bougie 1 h de décision")
    return pos


def daily_bars(df1: pd.DataFrame) -> pd.DataFrame:
    d = resample_ohlcv(df1, "1d")
    if len(d) > 1 and not (np.diff(d.index.to_numpy()) == np.timedelta64(1, "D")).all():
        raise ValueError("grille journalière incomplète")
    return d


# ------------------------------------------------------------------ familles

def family_signal(df1: pd.DataFrame, family: str) -> tuple[np.ndarray, np.ndarray]:
    """(signal, sigma30) sur la grille 1 h : valeur aux bougies de décision, NaN ailleurs."""
    if family not in FAMILIES:
        raise ValueError(family)
    n = len(df1)
    sig = np.full(n, np.nan)
    s30_1h = np.full(n, np.nan)
    d = daily_bars(df1)
    C = d["close"].to_numpy(dtype=np.float64)
    Hd = d["high"].to_numpy(dtype=np.float64)
    nd = len(C)
    s30 = sigma30(C)
    if family == "A":
        votes = np.zeros(nd)
        bad = np.isnan(C).copy()
        for L in A_LAGS:
            ref = np.full(nd, np.nan)
            if nd > L:
                ref[L:] = C[:-L]
            r = C / ref - 1.0
            bad |= np.isnan(r)
            votes += np.where(np.isnan(r), 0.0, (r > 0).astype(float))
        s = np.where(bad, np.nan, (votes >= 2).astype(float))
        pos = decision_positions(df1, d.index, 24)
        sig[pos], s30_1h[pos] = s, s30
    elif family in DONCHIAN:
        N = DONCHIAN[family]
        m = win_stat(Hd, N, "max", lag=1)
        s = np.where(np.isnan(C) | np.isnan(m), np.nan, (C > m).astype(float))
        pos = decision_positions(df1, d.index, 24)
        sig[pos], s30_1h[pos] = s, s30
    else:  # C
        sma = win_stat(C, C_SMA, "mean", lag=0)
        filt = np.where(np.isnan(C) | np.isnan(sma), np.nan, (C > sma).astype(float))
        h4 = resample_ohlcv(df1, "4h")
        if len(h4) > 1 and not (np.diff(h4.index.to_numpy()) == np.timedelta64(4, "h")).all():
            raise ValueError("grille 4 h incomplète")
        C4 = h4["close"].to_numpy(dtype=np.float64)
        H4 = h4["high"].to_numpy(dtype=np.float64)
        m4 = win_stat(H4, C_BREAK, "max", lag=1)
        brk = np.where(np.isnan(C4) | np.isnan(m4), np.nan, (C4 > m4).astype(float))
        d_last = (h4.index + 4 * H1).floor("D") - pd.Timedelta(days=1)
        di = d.index.get_indexer(d_last)
        f_at = np.where(di >= 0, filt[np.maximum(di, 0)], np.nan)
        s_at = np.where(di >= 0, s30[np.maximum(di, 0)], np.nan)
        s = np.where(np.isnan(brk) | np.isnan(f_at), np.nan, brk * f_at)
        pos = decision_positions(df1, h4.index, 4)
        sig[pos], s30_1h[pos] = s, s_at
    return sig, s30_1h


def barrier(s30: np.ndarray, kind: str, scale: float = 1.0) -> np.ndarray:
    """Demi-largeur log b. scale (batterie ±20 %) multiplie log(1,027) ou sigma30*sqrt(5) ; plancher inchangé."""
    if kind == "fixed":
        return np.full(len(s30), B_FIXED * scale)
    if kind == "vol":
        return np.maximum(scale * s30 * SIGMA_SCALE, B_FLOOR)  # NaN si sigma30 NaN
    raise ValueError(kind)


# ------------------------------------------------------------------ moteur

def next_valid(valid: np.ndarray) -> np.ndarray:
    """nv[t] = plus petit j >= t avec open valide (n si aucun) ; taille n+1."""
    n = len(valid)
    nv = np.full(n + 1, n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        nv[t] = t if valid[t] else nv[t + 1]
    return nv


def run_engine(open_: np.ndarray, close: np.ndarray, sig: np.ndarray, b: np.ndarray,
               horizon: int = H_TIME, delay: int = EXEC_DELAY) -> tuple[np.ndarray, list[dict]]:
    """Cibles décidées à chaque clôture (0/1) et journal des trades (indices de la grille 1 h)."""
    o = np.asarray(open_, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = len(o)
    nv = next_valid(~np.isnan(o))

    def first_open(t: int) -> int:
        return int(nv[min(t, n)])

    target = np.zeros(n)
    trades: list[dict] = []
    allowed = 0
    i = 0
    while i < n:
        if i >= allowed and sig[i] == 1.0 and np.isfinite(b[i]):
            j = first_open(i + delay)
            bb = float(b[i])
            if j >= n:  # décision à la dernière clôture : jamais exécutée
                target[i:] = 1.0
                trades.append({"dec": i, "entry": None, "E": None, "b": bb, "exit_dec": None,
                               "reason": "non exécutée", "exit": None})
                break
            E = float(o[j])
            up, dn = E * math.exp(bb), E * math.exp(-bb)
            k, reason = j, None
            while k < n:
                ck = c[k]
                if ck == ck:
                    if ck >= up:
                        reason = "haute"
                        break
                    if ck <= dn:
                        reason = "basse"
                        break
                if k - j + 1 >= horizon:
                    reason = "temps"
                    break
                k += 1
            if reason is None:  # fin des données, position ouverte
                target[i:] = 1.0
                trades.append({"dec": i, "entry": j, "E": E, "b": bb, "exit_dec": None, "reason": "ouverte",
                               "exit": None})
                break
            target[i:k] = 1.0
            target[k] = 0.0
            x = first_open(k + delay)
            trades.append({"dec": i, "entry": j, "E": E, "b": bb, "exit_dec": k, "reason": reason,
                           "exit": x if x < n else None})
            if x >= n:
                break
            allowed = x
            i = x
            continue
        i += 1
    return target, trades


def expected_held(n: int, trades: list[dict]) -> np.ndarray:
    h = np.zeros(n)
    for t in trades:
        if t["entry"] is None:
            continue
        h[t["entry"]: (t["exit"] if t["exit"] is not None else n)] = 1.0
    return h


def build(df1: pd.DataFrame, cfg: dict, *, b_scale: float = 1.0, horizon: int = H_TIME,
          delay: int = EXEC_DELAY) -> dict:
    sig, s30 = family_signal(df1, cfg["family"])
    b = barrier(s30, cfg["barrier"], b_scale)
    target, trades = run_engine(df1["open"].to_numpy(), df1["close"].to_numpy(), sig, b, horizon, delay)
    return {"raw": sig, "sigma30": s30, "b": b, "target": target, "trades": trades}


def bt(target: np.ndarray, df1: pd.DataFrame, start=None, end=None, delay: int = EXEC_DELAY, cost_mult: float = 1.0):
    return run_backtest(pd.Series(target, index=df1.index), df1["open"], start=start, end=end, exec_delay=delay,
                        cost_per_side=COST_PER_SIDE, cost_multiplier=cost_mult)


def concat_sharpe(df1: pd.DataFrame, cfg: dict, **kw) -> float:
    """Sharpe concaténé 2021-01 -> 2025-09 (utilisé par T1-règles et la batterie)."""
    delay = kw.get("delay", EXEC_DELAY)
    r = build(df1, cfg, **kw)
    return summarize(bt(r["target"], df1, CONCAT_START, CONCAT_END, delay=delay))["sharpe"]


# ------------------------------------------------------------------ essai complet

def exits_table(trades: list[dict], idx: pd.DatetimeIndex, a=None, b=None) -> dict:
    out = {"haute": 0, "basse": 0, "temps": 0, "ouverte": 0, "non exécutée": 0}
    for t in trades:
        ref = t["entry"] if t["entry"] is not None else t["dec"]
        tt = idx[ref]
        if (a is None or tt >= a) and (b is None or tt <= b):
            out[t["reason"]] += 1
    return out


def run_trial(cfg: dict, log) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = harness.load_dev(log)
    folds = make_folds()
    log(f"essai {cfg['id']} : famille {cfg['family']}, barrière {cfg['barrier']}, temps {H_TIME} h, "
        f"exec_delay {EXEC_DELAY}, coût/côté {COST_PER_SIDE}, aucun tuning")
    base_rows, base_concat, base_diffs = harness.baseline_block(df, folds, log)
    r = build(df, cfg)
    target, trades = r["target"], r["trades"]
    n = len(df)
    idx = df.index

    # cohérence moteur / bot.backtest : positions tenues
    full = bt(target, df)
    exp_h = expected_held(n, trades)
    same_pos = bool(np.array_equal(full.position.to_numpy(), exp_h))
    log(f"[cohérence] positions bot.backtest == positions attendues du moteur ({n} bougies) : {same_pos}")
    assert same_pos

    # décision à la dernière clôture dev (= HOLDOUT_START) : jamais exécutée
    res_c = bt(target, df, CONCAT_START, CONCAT_END)
    t2 = target.copy()
    t2[-1] = 1.0 - t2[-1]
    res_alt = bt(t2, df, CONCAT_START, CONCAT_END)
    inv = bool(np.array_equal(res_c.equity.to_numpy(), res_alt.equity.to_numpy())
               and np.array_equal(res_c.position.to_numpy(), res_alt.position.to_numpy()))
    log(f"[garde] décision prise à la clôture de {idx[-1]} (= HOLDOUT_START) jamais exécutée : backtest "
        f"invariant à cette décision : {inv} ; cible à cette clôture {target[-1]:.0f} ; raw signal "
        f"{r['raw'][-1]}")
    assert inv
    for t in trades:
        if t["entry"] is not None:
            assert idx[t["entry"]] <= harness.MAX_INDEX
        if t["exit"] is not None:
            assert idx[t["exit"]] <= harness.MAX_INDEX

    fold_out = []
    for f in folds:
        rf = bt(target, df, f.test_start, f.test_end)
        fold_out.append({"fold": f.k, "test_start": f.test_start, "test_end": f.test_end, "model": summarize(rf),
                         "exits": exits_table(trades, idx, f.test_start, f.test_end)})
    m_c = summarize(res_c)
    m_is = summarize(bt(target, df, IS_START, IS_END))
    dr = daily_returns(res_c.equity).to_numpy()
    from scipy.stats import kurtosis, skew
    daily = {"n_days": int(len(dr)), "mean": float(dr.mean()), "std": float(dr.std(ddof=1)),
             "sr_daily": float(dr.mean() / dr.std(ddof=1)), "skew": float(skew(dr, bias=False)),
             "kurtosis": float(kurtosis(dr, fisher=False, bias=False))}
    ex_oos = exits_table(trades, idx, CONCAT_START, CONCAT_END)
    ex_all = exits_table(trades, idx)
    # trades : moteur (entrées exécutées dans l'OOS) vs segments bot.backtest
    eng_oos = sum(1 for t in trades if t["entry"] is not None and CONCAT_START <= idx[t["entry"]] <= CONCAT_END)
    inherited = bool(res_c.initial_position > 0)
    log(f"[trades] segments bot.backtest OOS {m_c['trades']} ; entrées moteur exécutées dans l'OOS {eng_oos} ; "
        f"position héritée au 2021-01-01 : {inherited} -> cohérent : {m_c['trades'] == eng_oos + int(inherited)}")
    assert m_c["trades"] == eng_oos + int(inherited)

    raw = r["raw"]
    oos_m = np.asarray((idx >= CONCAT_START) & (idx <= CONCAT_END))
    k_hours = 4 if cfg["family"] == "C" else 24
    hours = idx.hour.to_numpy()
    is_dec = (hours % k_hours == k_hours - 1)
    nan_dec_oos = int((is_dec & oos_m & np.isnan(raw)).sum())
    on_dec_oos = int((is_dec & oos_m & (raw == 1.0)).sum())
    bvals = np.array([t["b"] for t in trades if t["entry"] is not None and idx[t["entry"]] >= CONCAT_START])
    floor_hits = int(np.sum(np.isclose(bvals, B_FLOOR))) if cfg["barrier"] == "vol" else 0

    # ------------------------------------------------ affichage
    log()
    log("=== Par fold (test) : règle / baseline SMA168 / buy & hold, mêmes folds, mêmes coûts ===")
    log(f"{'fold':<16} | {'stratégie':<9} | " + harness.metrics_header())
    for f, fo, b in zip(folds, fold_out, base_rows):
        lab = f"{f.k} {f.test_start.date()}"
        log(f"{lab:<16} | {'règle':<9} | " + harness.fmt_metrics(fo["model"]))
        log(f"{'':<16} | {'baseline':<9} | " + harness.fmt_metrics(b["baseline"]))
        log(f"{'':<16} | {'B&H':<9} | " + harness.fmt_metrics(b["buy_and_hold"]))
    lab = "concat 2021-25"
    log(f"{lab:<16} | {'règle':<9} | " + harness.fmt_metrics(m_c))
    log(f"{'':<16} | {'baseline':<9} | " + harness.fmt_metrics(base_concat["baseline"]))
    log(f"{'':<16} | {'B&H':<9} | " + harness.fmt_metrics(base_concat["buy_and_hold"]))
    log(f"{'IS 2019-2020':<16} | {'règle':<9} | " + harness.fmt_metrics(m_is) + "  (préchauffage, informatif)")
    log()
    log("=== Sorties (trades par bougie d'entrée exécutée) : haute / basse / temps / ouverte en fin / non exécutée ===")
    for fo in fold_out:
        e = fo["exits"]
        log(f"fold {fo['fold']} : " + " ; ".join(f"{k} {v}" for k, v in e.items()))
    log("OOS 2021-25 : " + " ; ".join(f"{k} {v}" for k, v in ex_oos.items()))
    log("tout 2019-25 : " + " ; ".join(f"{k} {v}" for k, v in ex_all.items()))
    if len(bvals):
        log(f"barrière b des trades OOS : min {bvals.min():.5f} médiane {np.median(bvals):.5f} max {bvals.max():.5f} ; "
            f"au plancher log(1,009) : {floor_hits}")
    log(f"bougies de décision OOS : signal 1 sur {on_dec_oos}, NaN sur {nan_dec_oos}")
    d_base = m_c["sharpe"] - base_concat["baseline"]["sharpe"]
    d_bh = m_c["sharpe"] - base_concat["buy_and_hold"]["sharpe"]
    npos = sum(1 for fo in fold_out if isinstance(fo["model"]["sharpe"], float) and fo["model"]["sharpe"] > 0)
    log()
    log(f"Sharpe OOS concaténé règle {m_c['sharpe']:.4f} ; baseline {base_concat['baseline']['sharpe']:.4f} "
        f"(écart {d_base:+.4f}) ; B&H {base_concat['buy_and_hold']['sharpe']:.4f} (écart {d_bh:+.4f}) ; "
        f"trades {m_c['trades']} ; folds à Sharpe > 0 : {npos}/9 ; Sharpe IS 2019-2020 {m_is['sharpe']:.4f}")
    log(f"rendements journaliers OOS : {daily}")
    last_open = max(res_c.end_mark_time, *[pd.Timestamp(fo['model']['end_mark_time']) for fo in fold_out])
    log(f"[garde] dernier open lu par les backtests = {last_open} <= {harness.MAX_INDEX} : "
        f"{last_open <= harness.MAX_INDEX}")
    assert last_open <= harness.MAX_INDEX

    out = {
        "id": cfg["id"], "config": cfg, "seed": None, "tuning": "aucun (paramètres a priori)",
        "dataset": DEFAULT_DATASET.name, "dataset_sha256": sha256_file(DEFAULT_DATASET),
        "index_max_read": df.index.max(), "cost_per_side": COST_PER_SIDE, "exec_delay": EXEC_DELAY,
        "cost_multiplier": 1.0, "h_time": H_TIME, "b_fixed": B_FIXED, "b_floor": B_FLOOR,
        "baseline_equality": base_diffs,
        "folds": [{**fo, "baseline": b["baseline"], "buy_and_hold": b["buy_and_hold"]}
                  for fo, b in zip(fold_out, base_rows)],
        "concatenated": {"model": m_c, "baseline": base_concat["baseline"], "buy_and_hold": base_concat["buy_and_hold"]},
        "in_sample_2019_2020": m_is,
        "delta_sharpe_vs_baseline": d_base, "delta_sharpe_vs_bh": d_bh, "folds_sharpe_positive": npos,
        "daily_oos": daily, "exits_oos": ex_oos, "exits_all": ex_all, "engine_trades_oos": eng_oos,
        "inherited_position_2021": inherited, "decision_bars_oos_signal1": on_dec_oos,
        "decision_bars_oos_nan": nan_dec_oos, "barrier_floor_hits_oos": floor_hits,
        "positions_consistent": same_pos, "last_decision_not_executed": inv,
    }
    sig_df = pd.DataFrame({"signal": target, "raw_signal": r["raw"], "b": r["b"], "sigma30": r["sigma30"]},
                          index=idx)
    sig_df.index.name = "open_time"
    rows = []
    for t in trades:
        rows.append({"decision_time": idx[t["dec"]],
                     "entry_time": idx[t["entry"]] if t["entry"] is not None else pd.NaT,
                     "E": np.nan if t["E"] is None else t["E"], "b": t["b"],
                     "exit_decision_time": idx[t["exit_dec"]] if t["exit_dec"] is not None else pd.NaT,
                     "reason": t["reason"], "exit_time": idx[t["exit"]] if t["exit"] is not None else pd.NaT})
    tr_df = pd.DataFrame(rows, columns=["decision_time", "entry_time", "E", "b", "exit_decision_time", "reason",
                                        "exit_time"])
    return out, sig_df, tr_df


def registry_line(out: dict, date: str) -> str:
    cfg = out["config"]
    fs = ";".join(f"{fo['model']['sharpe']:.3f}" for fo in out["folds"])
    ft = ";".join(str(fo["model"]["trades"]) for fo in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = "éligible (> baseline ; candidat désigné dans le rapport P3-R9)" if eligible else "abandonné (<= baseline)"
    bar = ("barrière fixe b=log(1,027)" if cfg["barrier"] == "fixed"
           else "barrière vol b=max(sigma30 journalier x sqrt(5) ; log(1,009))")
    eo = out["exits_oos"]
    cmd = (f"`.venv\\Scripts\\python experiments\\rules.py experiments\\configs\\{cfg['id']}.json` puis "
           f"`.venv\\Scripts\\python experiments\\rules.py --register {cfg['id']}`")
    cells = [cfg["id"], date, out["commit"][:7], "seed=aucune (règle déterministe)",
             f"règle {cfg['family']} ({cfg['name']})", f"signal={cfg['signal_text']}",
             f"sortie triple barrière {bar} ; temps 120 bougies 1 h ; contrôle aux clôtures 1 h ; exec t+1 ; "
             f"sorties OOS haute {eo['haute']} basse {eo['basse']} temps {eo['temps']}",
             "paramètres a priori (brief P3-B9) ; aucun tuning", "grille interne 0 (aucun tuning)",
             "folds 1-9 (make_folds ; série continue 2019-01 -> 2025-09)",
             f"Sharpe/fold {fs}", f"trades/fold {ft}", f"Sharpe concat {m['sharpe']:.4f}",
             f"trades concat {m['trades']}", f"écart baseline {out['delta_sharpe_vs_baseline']:+.4f}",
             f"écart B&H {out['delta_sharpe_vs_bh']:+.4f}", statut, cmd]
    for c_ in cells:
        if "|" in c_:
            raise ValueError(f"« | » interdit dans une cellule : {c_}")
    line = "| " + " | ".join(cells) + " |"
    assert line.count("|") == 19
    return line


def register(tid: str) -> int:
    num = int(tid[1:])
    if not REGISTRY_FIRST <= num <= REGISTRY_LAST:
        raise RuntimeError(f"{tid} hors du budget P3-B9 (E48-E55)")
    out = json.loads((RES / f"{tid}.json").read_text(encoding="utf-8"))
    if out["code_dirty"]:
        raise RuntimeError("résultat produit avec du code non commité : refus")
    head = harness.git("rev-parse", "HEAD")
    changed = harness.git("diff", "--name-only", out["commit"], head, "--", *CODE_PATHS)
    if changed:
        raise RuntimeError(f"code modifié depuis le commit du résultat : {changed}")
    lines = REGISTRE.read_text(encoding="utf-8").splitlines()
    if len(lines) != num - 1:
        raise RuntimeError(f"registre : {len(lines)} lignes, {num - 1} attendues avant {tid}")
    if any(ln.split("|")[1].strip() == tid for ln in lines):
        raise RuntimeError(f"{tid} déjà au registre")
    date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    line = registry_line(out, date)
    with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    n_lines = len(REGISTRE.read_text(encoding="utf-8").splitlines())
    print(f"registre : ligne {tid} ajoutée ({n_lines} lignes)")
    return 0


# ------------------------------------------------------------------ auto-tests (synthétique)

def synth_1h(n_days: int, seed: int = 0, start: str = "2018-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    lr = rng.normal(0.0, 0.006, n)
    c = 100.0 * np.exp(np.cumsum(lr))
    o = np.concatenate([[100.0], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": rng.uniform(1, 2, n)}, index=idx)
    df["missing"] = False
    df.index.name = "open_time"
    return df


def flat_1h(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0}, index=idx)
    df["missing"] = False
    return df


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ÉCHEC'}] {name}")

    print("=== auto-tests du moteur (synthétique) ===")
    up_lvl = 100.0 * math.exp(B_FIXED)
    dn_lvl = 100.0 * math.exp(-B_FIXED)
    print(f"  niveaux à la main pour E = 100 : haute {up_lvl:.6f} (= 102,7), basse {dn_lvl:.6f} (= 97,371...)")
    check("log(1,027) -> haute = 102,7", abs(up_lvl - 102.7) < 1e-9)

    # 1. barrière haute : signal à 10, E = open[11] = 100, close[14] = 102,69 (< 102,7), close[15] = niveau exact
    df = flat_1h(400)
    df.loc[df.index[14], "close"] = 102.69
    df.loc[df.index[15], "close"] = up_lvl
    df.loc[df.index[16], "open"] = 102.70
    sig = np.full(len(df), np.nan)
    sig[10] = 1.0
    b = np.full(len(df), B_FIXED)
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b)
    check("haute : 1 trade, entrée 11, sortie décidée 15 (close = niveau haut, égalité incluse ; 102,69 non), exécutée 16",
          len(tr) == 1 and tr[0]["entry"] == 11 and tr[0]["exit_dec"] == 15 and tr[0]["exit"] == 16
          and tr[0]["reason"] == "haute")
    res = bt(tg, df)
    check("haute : positions bot.backtest = moteur (tenue 11..15)",
          np.array_equal(res.position.to_numpy(), expected_held(len(df), tr))
          and res.position.iloc[11:16].eq(1).all() and res.position.iloc[16] == 0)
    exp_ret = (1 - COST_PER_SIDE) * (102.70 / 100.0) * (1 - COST_PER_SIDE) - 1
    check(f"haute : rendement net du trade = (1-c)*1,027*(1-c)-1 = {exp_ret:.8f}",
          abs(res.trades["net_return"].iloc[0] - exp_ret) < 1e-12)

    # 2. barrière basse
    df = flat_1h(400)
    df.loc[df.index[20], "close"] = 97.38
    df.loc[df.index[21], "close"] = dn_lvl
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b)
    check("basse : sortie décidée 21 (close = niveau bas 100/1,027, égalité incluse ; 97,38 non), exécutée 22",
          tr[0]["exit_dec"] == 21 and tr[0]["exit"] == 22 and tr[0]["reason"] == "basse")

    # 3. barrière temporelle
    df = flat_1h(400)
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b)
    check("temps : entrée 11, sortie décidée 11+119 = 130, exécutée 131, 120 bougies tenues",
          tr[0]["exit_dec"] == 130 and tr[0]["exit"] == 131 and tr[0]["reason"] == "temps"
          and expected_held(len(df), tr).sum() == 120)
    tg2, tr2 = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b, delay=2)
    check("delay 2 : entrée 12, sortie décidée 12+119 = 131, exécutée 133",
          tr2[0]["entry"] == 12 and tr2[0]["exit_dec"] == 131 and tr2[0]["exit"] == 133)
    res2 = bt(tg2, df, delay=2)
    check("delay 2 : positions bot.backtest (exec_delay=2) = moteur",
          np.array_equal(res2.position.to_numpy(), expected_held(len(df), tr2)))

    # 4. pas de pyramidage, ré-entrée
    sig = np.full(len(df), np.nan)
    sig[[10, 50, 100, 130]] = 1.0  # 50, 100, 130 : en position (130 = clôture de la décision de sortie)
    sig[131] = 1.0  # clôture de la bougie d'exécution de la sortie : autorisée
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b)
    check("aucun pyramidage (signaux 50, 100 ignorés) ; signal à la clôture de décision de sortie (130) ignoré",
          len(tr) == 2 and tr[0]["dec"] == 10)
    check("ré-entrée : signal pris à la clôture 131 (bougie d'exécution de la sortie), entrée 132",
          tr[1]["dec"] == 131 and tr[1]["entry"] == 132)
    res = bt(tg, df)
    check("ré-entrée : 2 segments distincts dans bot.backtest, bougie 131 à plat",
          len(res.trades) == 2 and res.position.iloc[131] == 0 and res.position.iloc[132] == 1)

    # 5. open manquant à l'exécution, close manquant pendant le trade
    df = flat_1h(400)
    df.loc[df.index[11], ["open", "high", "low", "close"]] = np.nan
    df.loc[df.index[11], "missing"] = True
    df.loc[df.index[12], "open"] = 101.0
    sig = np.full(len(df), np.nan)
    sig[10] = 1.0
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b)
    check("open manquant : entrée à la 1re ouverture disponible (12), E = 101",
          tr[0]["entry"] == 12 and tr[0]["E"] == 101.0)
    check("open manquant : positions bot.backtest = moteur",
          np.array_equal(bt(tg, df).position.to_numpy(), expected_held(len(df), tr)))

    # 6. plancher vol et sigma30 à la main
    rng = np.random.default_rng(3)
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 40)))
    s = sigma30(C)
    lr = np.diff(np.log(C))
    hand = float(np.std(lr[39 - 30:39], ddof=1))  # r_10..r_39 = lr[9..38]
    check(f"sigma30 au jour 39 = std ddof=1 de r_10..r_39 = {hand:.8f}", abs(s[39] - hand) < 1e-15)
    check("sigma30 NaN avant le jour 30", np.isnan(s[:30]).all() and np.isfinite(s[30]))
    bv = barrier(s, "vol")
    check(f"plancher : sigma30*sqrt(5) = {s[39] * SIGMA_SCALE:.6f} < log(1,009) = {B_FLOOR:.6f} -> b = plancher",
          s[39] * SIGMA_SCALE < B_FLOOR and bv[39] == B_FLOOR)
    C2 = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 40)))
    s2 = sigma30(C2)
    check("au-dessus du plancher : b = sigma30*sqrt(5)", barrier(s2, "vol")[39] == s2[39] * SIGMA_SCALE > B_FLOOR)
    check("barrière vol NaN si sigma30 NaN -> aucune entrée", np.isnan(barrier(s, "vol")[:30]).all())
    sig = np.full(10, np.nan)
    sig[2] = 1.0
    _, trn = run_engine(np.full(10, 100.0), np.full(10, 100.0), sig, np.full(10, np.nan))
    check("signal 1 avec b NaN : aucune entrée", len(trn) == 0)

    # 7. signaux des familles à la main, bougie manquante
    df = synth_1h(200, seed=5)
    d = daily_bars(df)
    Cd, Hd = d["close"].to_numpy(), d["high"].to_numpy()
    for fam in FAMILIES:
        sg, _ = family_signal(df, fam)
        k = 4 if fam == "C" else 24
        nonnan = np.flatnonzero(~np.isnan(sg))
        check(f"{fam} : signal uniquement aux heures de décision (heure % {k} == {k - 1})",
              (df.index[nonnan].hour % k == k - 1).all())
    sgA, _ = family_signal(df, "A")
    dd = 150
    hand = int(sum(Cd[dd] / Cd[dd - L] - 1 > 0 for L in A_LAGS) >= 2)
    check(f"A jour {dd} : vote à la main {hand}", sgA[dd * 24 + 23] == hand)
    check("A : NaN avant 120 jours d'historique", np.isnan(sgA[119 * 24 + 23]) and not np.isnan(sgA[120 * 24 + 23]))
    sgB, _ = family_signal(df, "B20")
    hand = float(Cd[dd] > Hd[dd - 20:dd].max())
    check(f"B20 jour {dd} : close > max(high d-20..d-1) à la main = {hand}", sgB[dd * 24 + 23] == hand)
    df_t = synth_1h(200, seed=6)
    d_t = daily_bars(df_t)
    jj = 120
    df_t.loc[d_t.index[jj] + 23 * H1, "close"] = d_t["high"].iloc[jj - 20:jj].max() * 1.0  # égalité stricte
    df_t.loc[d_t.index[jj] + 23 * H1, "high"] = max(df_t.loc[d_t.index[jj] + 23 * H1, "high"],
                                                     d_t["high"].iloc[jj - 20:jj].max())
    sg_t, _ = family_signal(df_t, "B20")
    check("B20 : égalité close = max -> 0 (inégalité stricte)", sg_t[jj * 24 + 23] == 0.0)
    allA = all((np.isnan(sgA[q * 24 + 23]) if q < 120 else
                sgA[q * 24 + 23] == float(sum(Cd[q] / Cd[q - L] - 1 > 0 for L in A_LAGS) >= 2)) for q in range(len(Cd)))
    check(f"A : vote à la main sur les {len(Cd)} jours (signal 1 sur {int(np.nansum(sgA))})", allA)
    for fam, N in DONCHIAN.items():
        sgN, _ = family_signal(df, fam)
        allN = all((np.isnan(sgN[q * 24 + 23]) if q < N else
                    sgN[q * 24 + 23] == float(Cd[q] > Hd[q - N:q].max())) for q in range(len(Cd)))
        check(f"{fam} : Donchian à la main sur les {len(Cd)} jours (signal 1 sur {int(np.nansum(sgN))})", allN)
    sgC, s30C = family_signal(df, "C")
    h4 = resample_ohlcv(df, "4h")
    C4h, H4h = h4["close"].to_numpy(), h4["high"].to_numpy()
    allC = True
    for T_ in range(len(h4)):
        D_ = (T_ * 4 + 4) // 24 - 1
        v_ = sgC[T_ * 4 + 3]
        if T_ < C_BREAK or D_ < C_SMA - 1:
            allC &= bool(np.isnan(v_))
        else:
            allC &= bool(v_ == float((Cd[D_] > Cd[D_ - 49:D_ + 1].mean()) and (C4h[T_] > H4h[T_ - 20:T_].max())))
    check(f"C : filtre SMA50 du dernier jour clôturé ET cassure 4 h, à la main sur les {len(h4)} bougies 4 h "
          f"(signal 1 sur {int(np.nansum(sgC))})", allC)
    T = 150 * 6 + 2  # bougie 4 h 08:00 du jour 150 -> jour clôturé 149
    sma = Cd[149 - 49:150].mean()
    hand = float((Cd[149] > sma) and (h4["close"].iloc[T] > h4["high"].iloc[T - 20:T].max()))
    check(f"C bougie 4 h {h4.index[T]} : filtre jour 149 ET cassure 20 bougies 4 h, à la main = {hand}",
          sgC[T * 4 + 3] == hand)
    T2 = 150 * 6 + 5  # 20:00 -> clôture à minuit : jour 150 clôturé
    sma2 = Cd[150 - 49:151].mean()
    hand2 = float((Cd[150] > sma2) and (h4["close"].iloc[T2] > h4["high"].iloc[T2 - 20:T2].max()))
    check("C bougie 4 h 20:00 : filtre sur le jour qui clôture en même temps", sgC[T2 * 4 + 3] == hand2)
    check("C : sigma30 du dernier jour clôturé", s30C[T * 4 + 3] == sigma30(Cd)[149])
    dm = df.copy()
    dm.loc[dm.index[160 * 24 + 5], ["open", "high", "low", "close", "volume"]] = np.nan
    dm.loc[dm.index[160 * 24 + 5], "missing"] = True
    for fam in ("A", "B20", "B55"):
        sm, _ = family_signal(dm, fam)
        check(f"{fam} : jour 160 manquant -> signal NaN", np.isnan(sm[160 * 24 + 23]))
    sm, _ = family_signal(dm, "C")
    check("C : bougie 4 h manquante (04:00 du jour 160) -> signal NaN", np.isnan(sm[(160 * 6 + 1) * 4 + 3]))
    check("C : jour 160 manquant -> NaN à 04:00 du jour 161 (filtre)", np.isnan(sm[(161 * 6 + 1) * 4 + 3]))
    sB, _ = family_signal(dm, "B20")
    hand = float(Cd[170] > np.nanmax(np.delete(Hd[150:170], 10)))
    check("B20 : fenêtre contenant le jour manquant -> max des valeurs disponibles", sB[170 * 24 + 23] == hand)

    # 8. anti-fuite : données modifiées après t0
    df = synth_1h(260, seed=11)
    rng = np.random.default_rng(12)
    for fam in FAMILIES:
        for kind in ("fixed", "vol"):
            cfg = {"family": fam, "barrier": kind}
            r0 = build(df, cfg)
            leak_ok, any_changed = True, False
            for t0 in rng.integers(130 * 24, 255 * 24, 4):
                dx = df.copy()
                cols = ["open", "high", "low", "close"]
                noise = np.exp(np.cumsum(rng.normal(0, 0.02, len(df) - t0 - 1)))[:, None]  # marche aléatoire
                dx.iloc[t0 + 1:, [dx.columns.get_loc(c_) for c_ in cols]] = (
                    dx.iloc[t0 + 1:][cols].to_numpy() * noise)
                r1 = build(dx, cfg)
                same_sig = np.array_equal(r0["raw"][:t0 + 1], r1["raw"][:t0 + 1], equal_nan=True)
                same_b = np.array_equal(r0["b"][:t0 + 1], r1["b"][:t0 + 1], equal_nan=True)
                same_tg = np.array_equal(r0["target"][:t0 + 1], r1["target"][:t0 + 1])
                dec0 = [(t["dec"], t["exit_dec"], t["reason"]) for t in r0["trades"]
                        if t["exit_dec"] is not None and t["exit_dec"] <= t0]
                dec1 = [(t["dec"], t["exit_dec"], t["reason"]) for t in r1["trades"]
                        if t["exit_dec"] is not None and t["exit_dec"] <= t0]
                ok_t = same_sig and same_b and same_tg and dec0 == dec1 and len(dec0) > 0
                changed_after = not np.array_equal(r0["raw"][t0 + 1:], r1["raw"][t0 + 1:], equal_nan=True)
                leak_ok &= ok_t
                any_changed |= changed_after
            check(f"anti-fuite {fam}/{kind} : 4 coupures, signaux, b, cibles <= t0 et sorties décidées <= t0 "
                  f"inchangés ({len(r0['trades'])} trades) ; test non vide : signaux > t0 modifiés", leak_ok and any_changed)

    # 9. décision à la dernière clôture : jamais exécutée
    df = flat_1h(300)
    sig = np.full(len(df), np.nan)
    sig[-1] = 1.0
    tg, tr = run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, np.full(len(df), B_FIXED))
    res = bt(tg, df)
    check("décision à la dernière clôture : trade 'non exécutée', aucune position tenue",
          len(tr) == 1 and tr[0]["reason"] == "non exécutée" and res.position.sum() == 0)
    print(f"auto-tests : {'OK' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--register")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.register:
        return register(args.register)
    log = harness.Tee()
    commit = harness.git("rev-parse", "HEAD")
    dirty = harness.git("status", "--porcelain", "--", *CODE_PATHS) != ""
    log(f"commit {commit} ; code modifié non commité (src, rules, rules_report, harnais, configs) : {dirty}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out, sig_df, tr_df = run_trial(cfg, log)
    out = {"commit": commit, "code_dirty": dirty, **out}
    path = args.out or (RES / f"{cfg['id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    sd = path.parent / f"rules_{cfg['id']}{'' if args.out is None else '_' + path.stem}"
    sd.mkdir(parents=True, exist_ok=True)
    sig_df.to_parquet(sd / "signal_1h.parquet", engine="pyarrow")
    tr_df.to_parquet(sd / "trades.parquet", engine="pyarrow")
    log(f"écrit : {path} ; {sd / 'signal_1h.parquet'} ; {sd / 'trades.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
