"""Brief P3-B5 : bougies 4 h agrégées depuis les bougies 1 h, features 4 h, labels 4 h, projection 1 h.

AGRÉGATION (aggregate_4h)
- Entrée : bougies 1 h de bot.split.load_dataset() (grille horaire complète, bougies manquantes =
  lignes NaN et/ou missing=True). Une heure est manquante si `missing` est vrai ou si une valeur
  OHLCV est NaN.
- Bougie 4 h d'horodatage T (open_time de sa 1re heure), T aligné sur 00:00 / 04:00 / ... / 20:00 UTC,
  heures T, T+1h, T+2h, T+3h :
      open = open(T) ; high = max des 4 high ; low = min des 4 low ; close = close(T+3h) ;
      volume = ((v(T) + v(T+1h)) + v(T+2h)) + v(T+3h)  (ordre fixe).
- Une bougie 4 h n'est émise que si ses 4 heures sont présentes dans l'entrée (une fenêtre qui
  commence ou finit au milieu d'une bougie 4 h n'émet pas cette bougie partielle : elle n'est pas
  clôturée). Si l'une des 4 heures est manquante, la bougie 4 h est marquée missing=True et ses
  OHLCV valent NaN.
- La bougie 4 h T est clôturée à T+4h (fin de l'heure T+3h) : elle n'est utilisable qu'à partir de là.

DÉCISION / EXÉCUTION
- Décision à la clôture de la bougie 4 h t (horodatage T), exécution à l'ouverture de la bougie 4 h
  suivante (open de l'heure T+4h), position tenue 4 heures (jusqu'à la décision suivante).
- Projection sur la grille 1 h (project_to_1h) : signal_1h[T+3h] = signal_4h[T], NaN ailleurs ; le
  moteur bot.backtest (non modifié, exec_delay=1 h) conserve la cible sur les NaN, donc la cible
  décidée sur la bougie 4 h T est exécutée à l'open de T+4h et tenue sur T+4h..T+7h.
  exec_delay = 1 bougie 4 h. Variante « +1 bougie 4 h » : exec_delay = 5 h (exécution à T+8h).

FEATURES 4 h (compute_features_4h) — n en bougies 4 h, valeur à t calculée sur les bougies 4 h <= t
(t clôturée), décalage 0, fenêtres finies, NaN si une bougie de la fenêtre manque (même cœur
numérique que bot.features : _rsum / _rmean / _rstd, sommes dans un ordre fixe) :
    r1_t        = log c_t - log c_{t-1}
    ret_n       = log c_t - log c_{t-n}                     n = 1, 6 (24 h), 42 (7 j), 180 (30 j)
    vol_n       = écart-type ddof=1 de r1 sur t-n+1..t      n = 6, 42, 180
    rsi_14      = 100 * moy(gain, 14) / (moy(gain,14) + moy(perte,14)), 50 si dénominateur nul
    sma_dev_n   = c_t / moyenne(c_{t-n+1..t}) - 1           n = 6, 42, 180
    logvol_z_42 = (log1p(v_t) - moy_42) / et_42 (log1p(v)), NaN si et nul
    range_1     = (h_t - l_t) / c_t ; range_6 = moyenne de range_1 sur 6 bougies
  Jeu « f4 » (14 features, déclaré a priori, SANS calendrier) : ret_1, ret_6, ret_42, ret_180,
  vol_6, vol_42, vol_180, rsi_14, sma_dev_6, sma_dev_42, sma_dev_180, logvol_z_42, range_1, range_6.
  Calendrier (hors f4, uniquement pour les contrôles) : hour_sin/cos (heure de début T, période 24 h),
  dow_sin/cos (jour de semaine de T, période 7).
  Fenêtres (bougies 4 h, t inclus) : SPANS4 ; LOOKBACK4 = 200 bougies 4 h.

LABELS 4 h (make_label_4h) — convention de P1 appliquée aux bougies 4 h : décision à la clôture de t,
entrée à l'open de la bougie 4 h t+1, sortie à l'open de t+1+H (H en bougies 4 h).
    fwd_t = log(open4[t+1+H] / open4[t+1])  = bot.labels.compute_labels(df4, horizon=H)["fwd_ret"]
            (NaN si une bougie 4 h de [t+1, t+1+H] est manquante ou hors données)
    C = log(1 + 2 x 0,0015)
  - "net"   : y = 1 si fwd_t > C, sinon 0.
  - "tb"    : trajectoire aux opens 4 h r_j = log(open4[t+1+j] / open4[t+1]), j = 1..H ;
              haute u = max(U, C), basse d ; U, d :
                "vol"   : k * vol_42_t * sqrt(H) (vol_42 4 h, données <= t) ;
                "fixed_log" : constante v (log) déclarée dans la config. Règle a priori de ce brief :
                          v = 0,027683 * sqrt(4H / 24), où 0,027683 = médiane de vol_168 * sqrt(24)
                          (bougies 1 h < 2021-01-01) publiée en P3-A3 (test C1d) ; mise à l'échelle
                          en racine du temps (24 h -> 4H heures). Aucune donnée OOS utilisée ;
              y = 1 si la haute est touchée strictement avant la basse (ou seule), 0 si basse
              d'abord ou expiration à H ; option expire_nan (« direction seule ») : expiration -> NaN.
  - "dir"   : y = 1 si fwd_t > 0 (direction seule, sans seuil de coût ; contrôle pour un label net).
  Données lues pour le label de t : open4[t+1 .. t+1+H] (et données <= t pour vol_42).
  Purge = H + 1 bougies 4 h : horizon passé à bot.split en heures = 4H + 3 (purge_mask retire t si
  t + (4H+4) h >= début de la fenêtre protégée, soit les H+1 bougies 4 h qui la précèdent).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bot.features import _any_missing, _rmean, _rstd, _shift
from bot.labels import compute_labels

OHLCV = ["open", "high", "low", "close", "volume"]
H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)
COST_RT = 2 * 0.0015
C_LOG = math.log(1.0 + COST_RT)
MED_VOL168_24H_PRE2021 = 0.027683  # P3-A3, C1d (bougies 1 h < 2021-01-01)


def fixed_log_rule(H: int) -> float:
    """Seuil fixe a priori (log) pour H bougies 4 h : 0,027683 * sqrt(4H / 24)."""
    return round(MED_VOL168_24H_PRE2021 * math.sqrt(4 * int(H) / 24), 6)

SPANS4: dict[str, int] = {
    "ret_1": 2, "ret_6": 7, "ret_42": 43, "ret_180": 181,
    "vol_6": 7, "vol_42": 43, "vol_180": 181,
    "rsi_14": 15,
    "sma_dev_6": 6, "sma_dev_42": 42, "sma_dev_180": 180,
    "logvol_z_42": 42,
    "range_1": 1, "range_6": 6,
    "hour_sin": 0, "hour_cos": 0, "dow_sin": 0, "dow_cos": 0,
}
F4: list[str] = [k for k in SPANS4 if not k.startswith(("hour_", "dow_"))]
CAL4: list[str] = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ALL4: list[str] = F4 + CAL4
LOOKBACK4 = max(SPANS4.values()) + 19  # 200 bougies 4 h


def horizon_hours(H: int) -> int:
    """Horizon à passer à bot.split (make_folds / split_fold / purge_mask) pour un label de H bougies 4 h."""
    return 4 * int(H) + 3


# ------------------------------------------------------------------ agrégation

def hourly_missing(df1: pd.DataFrame) -> np.ndarray:
    m = np.isnan(df1[OHLCV].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in df1.columns:
        m |= df1["missing"].to_numpy(dtype=bool)
    return m


def aggregate_4h(df1: pd.DataFrame) -> pd.DataFrame:
    idx = df1.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("index tz-aware (UTC) requis")
    if len(idx) > 1 and not (np.diff(idx.to_numpy()) == np.timedelta64(1, "h")).all():
        raise ValueError("grille horaire complète et régulière requise")
    n = len(idx)
    miss = hourly_missing(df1)
    hrs = idx.tz_convert("UTC").hour.to_numpy()
    starts = np.flatnonzero((hrs % 4 == 0) & (np.arange(n) + 3 < n))
    o = df1["open"].to_numpy(dtype=np.float64)
    h = df1["high"].to_numpy(dtype=np.float64)
    l = df1["low"].to_numpy(dtype=np.float64)
    c = df1["close"].to_numpy(dtype=np.float64)
    v = df1["volume"].to_numpy(dtype=np.float64)
    s = starts
    out = pd.DataFrame({
        "open": o[s],
        "high": np.maximum(np.maximum(np.maximum(h[s], h[s + 1]), h[s + 2]), h[s + 3]),
        "low": np.minimum(np.minimum(np.minimum(l[s], l[s + 1]), l[s + 2]), l[s + 3]),
        "close": c[s + 3],
        "volume": ((v[s] + v[s + 1]) + v[s + 2]) + v[s + 3],
    }, index=idx[s])
    m4 = miss[s] | miss[s + 1] | miss[s + 2] | miss[s + 3]
    out.loc[m4, OHLCV] = np.nan
    out["missing"] = m4
    out.index.name = "open_time"
    return out


def aggregate_4h_live(hours: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Agrégation « en direct » : à partir des seules dernières heures clôturées disponibles,
    renvoie les n_bars dernières bougies 4 h CLÔTURÉES (bougie partielle en cours ignorée)."""
    a = aggregate_4h(hours)
    if len(a) < n_bars:
        raise ValueError(f"{len(a)} bougies 4 h clôturées < {n_bars}")
    return a.iloc[-n_bars:]


# ------------------------------------------------------------------ features

def compute_features_4h(df4: pd.DataFrame) -> pd.DataFrame:
    idx = df4.index
    if len(idx) > 1 and not (np.diff(idx.to_numpy()) == np.timedelta64(4, "h")).all():
        raise ValueError("grille 4 h régulière requise")
    vals = {k: df4[k].to_numpy(dtype=np.float64) for k in OHLCV}
    missing = np.isnan(np.column_stack([vals[k] for k in OHLCV])).any(axis=1)
    if "missing" in df4.columns:
        missing |= df4["missing"].to_numpy(dtype=bool)
    o, h, l, c, v = (np.where(missing, np.nan, vals[k]) for k in OHLCV)
    logc = np.log(c)
    r1 = logc - _shift(logc, 1)
    out: dict[str, np.ndarray] = {}
    for n in (1, 6, 42, 180):
        out[f"ret_{n}"] = logc - _shift(logc, n)
    for n in (6, 42, 180):
        out[f"vol_{n}"] = _rstd(r1, n)
    dc = c - _shift(c, 1)
    gain = np.where(r1 > 0, dc, 0.0)
    loss = np.where(r1 < 0, -dc, 0.0)
    gain = np.where(np.isnan(r1), np.nan, gain)
    loss = np.where(np.isnan(r1), np.nan, loss)
    ag, al = _rmean(gain, 14), _rmean(loss, 14)
    tot = ag + al
    with np.errstate(invalid="ignore", divide="ignore"):
        rsi = np.where(tot > 0, 100.0 * ag / tot, 50.0)
    out["rsi_14"] = np.where(np.isnan(tot), np.nan, rsi)
    for n in (6, 42, 180):
        out[f"sma_dev_{n}"] = c / _rmean(c, n) - 1.0
    lv = np.log1p(v)
    sd = _rstd(lv, 42)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["logvol_z_42"] = np.where(sd > 0, (lv - _rmean(lv, 42)) / sd, np.nan)
    rng = (h - l) / c
    out["range_1"] = rng
    out["range_6"] = _rmean(rng, 6)
    u = idx.tz_convert("UTC")
    hour = u.hour.to_numpy(dtype=np.float64)
    dow = u.dayofweek.to_numpy(dtype=np.float64)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for name, span in SPANS4.items():
        bad = _any_missing(missing, span)
        out[name] = np.where(bad, np.nan, out[name]).astype(np.float64)
    return pd.DataFrame(out, index=idx, columns=ALL4)


def compute_features_4h_live(hours: pd.DataFrame) -> pd.Series:
    """Features de la dernière bougie 4 h clôturée, à partir des dernières heures clôturées :
    agrégation des LOOKBACK4 dernières bougies 4 h clôturées puis même cœur que le lot."""
    w = aggregate_4h_live(hours, LOOKBACK4)
    return compute_features_4h(w).iloc[-1]


# ------------------------------------------------------------------ labels

def _sigma_scale(f4: pd.DataFrame, H: int) -> np.ndarray:
    return f4["vol_42"].to_numpy() * math.sqrt(H)


def tb_paths(df4: pd.DataFrame, H: int):
    """r_j = log(open4[t+1+j] / open4[t+1]) pour j = 1..H (bougies manquantes = NaN)."""
    miss = np.isnan(df4[OHLCV].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in df4.columns:
        miss |= df4["missing"].to_numpy(dtype=bool)
    o = np.where(miss, np.nan, df4["open"].to_numpy(dtype=np.float64))
    n = len(o)
    entry = np.full(n, np.nan)
    entry[: n - 1] = o[1:]
    R = np.full((H, n), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(1, H + 1):
            px = np.full(n, np.nan)
            if n > 1 + j:
                px[: n - 1 - j] = o[1 + j:]
            R[j - 1] = np.log(px / entry)
    return R


def tb_first_hits(df4: pd.DataFrame, H: int, u: np.ndarray, d: np.ndarray):
    R = tb_paths(df4, H)
    n = R.shape[1]
    j_up = np.full(n, np.inf)
    j_dn = np.full(n, np.inf)
    with np.errstate(invalid="ignore"):
        for j in range(1, H + 1):
            r = R[j - 1]
            hu = (r >= u) & np.isinf(j_up)
            hd = (r <= -d) & np.isinf(j_dn)
            j_up[hu] = j
            j_dn[hd] = j
    return j_up, j_dn


def tb_barriers(df4: pd.DataFrame, f4: pd.DataFrame, spec: dict):
    H = int(spec["H"])
    n = len(df4)
    out = []
    for side in ("up", "down"):
        b = spec[side]
        if b["type"] == "vol":
            x = float(b["k"]) * _sigma_scale(f4, H)
        elif b["type"] == "fixed_log":
            x = np.full(n, float(b["v"]))
        elif b["type"] == "fixed":
            r = float(b["r"])
            x = np.full(n, math.log(1.0 + r) if side == "up" else -math.log(1.0 - r))
        else:
            raise ValueError(b)
        out.append(x)
    u = np.maximum(out[0], C_LOG)
    return u, out[1]


def make_label_4h(df4: pd.DataFrame, f4: pd.DataFrame, spec: dict) -> pd.Series:
    H = int(spec["H"])
    fwd = compute_labels(df4, horizon=H)["fwd_ret"].to_numpy()
    kind = spec["kind"]
    if kind == "net":
        y = (fwd > C_LOG).astype(np.float64)
        y[np.isnan(fwd)] = np.nan
    elif kind == "dir":
        y = (fwd > 0).astype(np.float64)
        y[np.isnan(fwd)] = np.nan
    elif kind == "tb":
        u, d = tb_barriers(df4, f4, spec)
        j_up, j_dn = tb_first_hits(df4, H, u, d)
        y = ((j_up < np.inf) & (j_up < j_dn)).astype(np.float64)
        bad = np.isnan(fwd) | np.isnan(u) | np.isnan(d)
        if spec.get("expire_nan"):
            bad |= np.isinf(j_up) & np.isinf(j_dn)
        y[bad] = np.nan
    else:
        raise ValueError(spec)
    return pd.Series(y, index=df4.index, name="label")


# ------------------------------------------------------------------ projection 4 h -> 1 h

def project_to_1h(sig4: np.ndarray, idx4: pd.DatetimeIndex, idx1: pd.DatetimeIndex) -> pd.Series:
    """signal_1h[T+3h] = signal_4h[T] (décision à la clôture de la bougie 4 h T), NaN ailleurs."""
    out = np.full(len(idx1), np.nan)
    pos = idx1.get_indexer(idx4 + 3 * H1)
    ok = pos >= 0
    out[pos[ok]] = np.asarray(sig4, dtype=np.float64)[ok]
    return pd.Series(out, index=idx1, name="signal")


# ------------------------------------------------------------------ autotests (données synthétiques)

def synthetic_hours(n: int = 6000, seed: int = 0, start="2020-06-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="h")
    rng = np.random.default_rng(seed)
    c = 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    v = np.exp(rng.normal(5, 1, n))
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)
    df["missing"] = False
    for a, b in ((2001, 2003), (3606, 3610), (4500, 4500)):  # trous : dans une bougie 4 h, à cheval, isolé
        df.iloc[a:b + 1, :5] = np.nan
        df.iloc[a:b + 1, df.columns.get_loc("missing")] = True
    return df


def selftest(log=print, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df1 = synthetic_hours(seed=seed)
    a = aggregate_4h(df1)
    # 1. définition, vérifiée bougie par bougie contre une boucle naïve
    bad = 0
    for T in a.index[::7]:
        hrs = df1.loc[T: T + 3 * H1]
        if hrs["missing"].any():
            ok = bool(a.loc[T, "missing"]) and np.isnan(a.loc[T, "close"])
        else:
            ok = (a.loc[T, "open"] == hrs["open"].iloc[0] and a.loc[T, "high"] == hrs["high"].max()
                  and a.loc[T, "low"] == hrs["low"].min() and a.loc[T, "close"] == hrs["close"].iloc[-1]
                  and a.loc[T, "volume"] == ((hrs["volume"].iloc[0] + hrs["volume"].iloc[1]) + hrs["volume"].iloc[2])
                  + hrs["volume"].iloc[3] and not a.loc[T, "missing"])
        bad += int(not ok)
    assert (a.index.hour % 4 == 0).all()
    log(f"[agrégation] {len(a)} bougies 4 h (synthétique, {len(df1)} heures), alignées 00/04/../20 UTC : "
        f"{(a.index.hour % 4 == 0).all()} ; missing {int(a['missing'].sum())} (attendu 4 : trous 2001-2003, "
        f"3606-3610 à cheval sur 2 bougies, 4500) ; contrôle naïf {len(a.index[::7])} bougies : {bad} écart(s)")
    assert bad == 0 and int(a["missing"].sum()) == 4
    f_b = compute_features_4h(a)
    # 2. anti-fuite : heures postérieures à la clôture de t modifiées
    nleak, npts = 0, 30
    pts = np.sort(rng.choice(np.arange(250, len(a) - 10), npts, replace=False))
    sens = 0
    for i in pts:
        T = a.index[i]
        cut = df1.index.get_loc(T + 4 * H1)  # première heure après la clôture de t
        d2 = df1.copy()
        k = len(d2) - cut
        for col in ("open", "high", "low", "close"):
            d2.iloc[cut:, d2.columns.get_loc(col)] = d2[col].iloc[cut:].to_numpy() * np.exp(rng.normal(0, 0.05, k))
        d2.iloc[cut:, d2.columns.get_loc("volume")] = rng.exponential(100.0, k)
        a2 = aggregate_4h(d2)
        f2 = compute_features_4h(a2)
        same_bar = a.iloc[: i + 1].equals(a2.iloc[: i + 1])
        same_f = np.array_equal(f_b.iloc[: i + 1].to_numpy(), f2.iloc[: i + 1].to_numpy(), equal_nan=True)
        nleak += int(not (same_bar and same_f))
        d3 = df1.copy()  # sensibilité : la dernière heure de t est bien lue
        d3.iloc[cut - 1, d3.columns.get_loc("close")] *= 1.01
        sens += int(not np.array_equal(compute_features_4h(aggregate_4h(d3)).iloc[i].to_numpy(),
                                       f_b.iloc[i].to_numpy(), equal_nan=True))
    log(f"[anti-fuite agrégation + features] {npts} bougies t : heures > clôture de t modifiées -> bougies <= t "
        f"et features <= t identiques : {npts - nleak}/{npts} ; sensibilité (close de la dernière heure de t "
        f"x1,01 change les features de t) : {sens}/{npts}")
    assert nleak == 0 and sens > 0
    # 3. égalité direct (dernières heures clôturées) = lot
    need = 4 * LOOKBACK4 + 3
    hs = rng.choice(np.arange(need + 10, len(df1)), 200, replace=False)
    neq, n_agg = 0, 0
    for j in hs:
        hours = df1.iloc[j - need + 1: j + 1]  # heures clôturées disponibles à la fin de l'heure j
        live_bars = aggregate_4h_live(hours, LOOKBACK4)
        Tl = live_bars.index[-1]
        assert Tl + 3 * H1 <= df1.index[j]  # dernière bougie 4 h clôturée
        n_agg += int(live_bars.equals(a.loc[live_bars.index]))
        fl = compute_features_4h_live(hours)
        neq += int(np.array_equal(fl.to_numpy(), f_b.loc[Tl].to_numpy(), equal_nan=True))
    log(f"[direct = lot] 200 instants (heure quelconque, fenêtre des {need} dernières heures clôturées) : "
        f"bougies 4 h identiques {n_agg}/200 ; features bit à bit identiques {neq}/200")
    assert neq == 200 and n_agg == 200
    # 4. labels : anti-fuite et sensibilité
    specs = [{"kind": "net", "H": 6}, {"kind": "net", "H": 30}, {"kind": "dir", "H": 12},
             {"kind": "tb", "H": 6, "up": {"type": "vol", "k": 1.0}, "down": {"type": "vol", "k": 1.0}},
             {"kind": "tb", "H": 30, "up": {"type": "vol", "k": 1.0}, "down": {"type": "vol", "k": 1.0},
              "expire_nan": True},
             # barrières fixes larges (vol synthétique 2 %/4 h) : sinon toujours touchées avant H et
             # la sensibilité à t+1+H est nulle par construction
             {"kind": "tb", "H": 12, "up": {"type": "fixed", "r": 0.12}, "down": {"type": "fixed", "r": 0.12}},
             {"kind": "tb", "H": 12, "up": {"type": "fixed_log", "v": 0.10},
              "down": {"type": "fixed_log", "v": 0.10}}]
    for spec in specs:
        H = spec["H"]
        y0 = make_label_4h(a, f_b, spec)
        sens = 0
        for i in pts[:20]:
            cut4 = i + 2 + H  # premières bougies 4 h > t+1+H
            Tcut = a.index[cut4]
            d2 = df1.copy()
            c1 = d2.index.get_loc(Tcut)
            k = len(d2) - c1
            for col in ("open", "high", "low", "close"):
                d2.iloc[c1:, d2.columns.get_loc(col)] = d2[col].iloc[c1:].to_numpy() * np.exp(rng.normal(0, 0.05, k))
            a2 = aggregate_4h(d2)
            y2 = make_label_4h(a2, compute_features_4h(a2), spec)
            assert np.array_equal(y0.iloc[: i + 1].to_numpy(), y2.iloc[: i + 1].to_numpy(), equal_nan=True), \
                f"FUITE label {spec} i={i}"
            hit = False
            for fac in (0.5, 2.0):
                d4 = df1.copy()
                T_exit = a.index[i + 1 + H]
                d4.loc[T_exit, "open"] = df1.loc[T_exit, "open"] * fac
                a4 = aggregate_4h(d4)
                y4 = make_label_4h(a4, compute_features_4h(a4), spec)
                hit |= not np.array_equal(y0.iloc[: i + 1].to_numpy(), y4.iloc[: i + 1].to_numpy(), equal_nan=True)
            sens += int(hit)
        log(f"[anti-fuite label] {json_spec(spec)} : 20 points, données > t+1+H modifiées -> labels <= t identiques : "
            f"OK ; sensibilité open4[t+1+H] x0,5 / x2 : {sens}/20 ; taux y=1 {np.nanmean(y0.to_numpy()):.4f}")
        assert sens > 0
    # 5. exemple numérique triple barrière 4 h
    idx = pd.date_range("2020-01-01", periods=60 * 4, freq="h", tz="UTC")
    o = np.full(len(idx), 100.0)
    seq = {41: 100.0, 42: 101.0, 43: 104.0, 44: 99.0, 45: 96.0}  # opens 4 h des bougies 41..45
    for b, px in seq.items():
        o[4 * b: 4 * b + 4] = px
    d = pd.DataFrame({"open": o, "high": o * 1.001, "low": o * 0.999, "close": o, "volume": 1.0}, index=idx)
    a5 = aggregate_4h(d)
    f5 = compute_features_4h(a5)
    sp = {"kind": "tb", "H": 4, "up": {"type": "fixed", "r": 0.03}, "down": {"type": "fixed", "r": 0.03}}
    y5 = make_label_4h(a5, f5, sp)
    yn = make_label_4h(a5, f5, {"kind": "net", "H": 4})
    log(f"[exemple TB 4 h fixe 3 %/3 % H=4] t=40 : {y5.iloc[40]} (attendu 1 : +4 % à j=2) ; t=41 : {y5.iloc[41]} "
        f"(attendu 0 : log(104/101) = +2,9 % < log(1,03), puis log(96/101) = -5,1 % : basse touchée à j=3) ; "
        f"net H=4 t=40 : {yn.iloc[40]} (attendu 0 : log(96/100) < C)")
    assert y5.iloc[40] == 1.0 and y5.iloc[41] == 0.0 and yn.iloc[40] == 0.0
    # 6. projection 4 h -> 1 h
    s4 = np.array([1.0, 0.0, np.nan, 1.0])
    p = project_to_1h(s4, a.index[:4], df1.index[:20])
    want = np.full(20, np.nan)
    want[[3, 7, 15]] = [1.0, 0.0, 1.0]
    assert np.array_equal(p.to_numpy(), want, equal_nan=True)
    log("[projection] signal_1h[T+3h] = signal_4h[T], NaN ailleurs : OK")
    log("auto-test agg4h : OK")


def json_spec(spec: dict) -> str:
    import json
    return json.dumps(spec, separators=(",", ":"))
