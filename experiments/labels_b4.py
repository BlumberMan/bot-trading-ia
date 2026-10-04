"""Labels du brief P3-B4 (consigne d'Iyad) : net de coûts et triple barrière.

Convention de P1 (bot.labels) : décision à la clôture de t, entrée à l'ouverture de t+1,
sortie à une ouverture (comme le moteur bot.backtest). Coût aller-retour = 2 x 0,15 %.
    C = log(1 + 0,0030)

1. Net de coûts (kind "net", horizon H) :
    y_t = 1 si log(open[t+1+H] / open[t+1]) > C, sinon 0.
    NaN si une bougie de [t+1, t+1+H] manque ou est hors données (même règle que
    bot.labels.compute_labels, dont fwd_ret est réutilisé tel quel).

2. Triple barrière (kind "tb", barrière temporelle H) : trajectoire aux OUVERTURES
    r_j = log(open[t+1+j] / open[t+1]), j = 1..H
    barrière haute  u_t = max(U_t, C)   (U_t : fixe log(1+a) ou k_up * sigma_t * sqrt(H))
    barrière basse  d_t = D_t > 0       (D_t : fixe -log(1-b) ou k_dn * sigma_t * sqrt(H))
    sigma_t = vol_168 de bot.features (écart-type des rendements log horaires des 168
    bougies <= t), donc calculée sur des données <= t.
    j_up = premier j tel que r_j >= u_t ; j_dn = premier j tel que r_j <= -d_t.
    y_t = 1 si j_up existe et (j_dn n'existe pas ou j_up < j_dn), sinon 0 (barrière basse
    touchée en premier, ou barrière temporelle H atteinte sans toucher la haute).
    Comme u_t >= C, y_t = 1 implique un rendement brut > coût aller-retour.
    NaN si une bougie de [t+1, t+1+H] manque / hors données, ou si sigma_t est NaN
    (barrières proportionnelles à la volatilité).

Données lues pour le label de t : open[t+1 .. t+1+H] (et données <= t pour sigma_t).
Purge = H+1 bougies (bot.split, horizon=H), identique au label de P1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bot.features import compute_features
from bot.labels import compute_labels

COST_RT = 2 * 0.0015
C_LOG = math.log(1.0 + COST_RT)


def net_cost_labels(df: pd.DataFrame, H: int) -> pd.Series:
    fwd = compute_labels(df, horizon=H)["fwd_ret"]
    y = (fwd > C_LOG).astype(np.float64)
    y[fwd.isna()] = np.nan
    return y.rename("label")


def _barrier(spec: dict, sigma: np.ndarray, H: int, side: str) -> np.ndarray:
    n = len(sigma)
    if spec["type"] == "fixed":
        r = float(spec["r"])
        v = math.log(1.0 + r) if side == "up" else -math.log(1.0 - r)
        return np.full(n, v)
    if spec["type"] == "vol":
        return float(spec["k"]) * sigma * math.sqrt(H)
    raise ValueError(spec)


def triple_barrier_labels(df: pd.DataFrame, H: int, up: dict, down: dict) -> pd.Series:
    valid = compute_labels(df, horizon=H)["fwd_ret"].notna().to_numpy()
    o = df["open"].to_numpy(dtype=np.float64)
    miss = np.isnan(df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in df.columns:
        miss |= df["missing"].to_numpy(dtype=bool)
    o = np.where(miss, np.nan, o)
    n = len(o)
    sigma = compute_features(df)["vol_168"].to_numpy()
    u = np.maximum(_barrier(up, sigma, H, "up"), C_LOG)
    d = _barrier(down, sigma, H, "down")
    entry = np.full(n, np.nan)
    entry[: n - 1] = o[1:]
    j_up = np.full(n, np.inf)
    j_dn = np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(1, H + 1):
            px = np.full(n, np.nan)
            if n > 1 + j:
                px[: n - 1 - j] = o[1 + j:]
            r = np.log(px / entry)
            hit_u = (r >= u) & np.isinf(j_up)
            hit_d = (r <= -d) & np.isinf(j_dn)
            j_up[hit_u] = j
            j_dn[hit_d] = j
    y = ((j_up < np.inf) & (j_up < j_dn)).astype(np.float64)
    bad = ~valid | np.isnan(u) | np.isnan(d)
    y[bad] = np.nan
    return pd.Series(y, index=df.index, name="label")


def make_label(df: pd.DataFrame, spec: dict) -> pd.Series:
    if spec["kind"] == "net":
        return net_cost_labels(df, int(spec["H"]))
    if spec["kind"] == "tb":
        return triple_barrier_labels(df, int(spec["H"]), spec["up"], spec["down"])
    raise ValueError(spec)


# ------------------------------------------------------------------ autotest (données synthétiques)

def selftest(df: pd.DataFrame, specs: list[dict], log=print, n_points: int = 25, seed: int = 0) -> None:
    """Anti-fuite : modifier les données > t+1+H ne change aucun label <= t.

    Sensibilité : modifier open[t+1+H] (x0,5 ou x2) change au moins un label sur l'ensemble des points
    (la fenêtre lue va bien jusqu'à t+1+H)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    for spec in specs:
        H = int(spec["H"])
        y0 = make_label(df, spec)
        changed_sens = 0
        pts = np.sort(rng.choice(np.arange(300, n - H - 5), n_points, replace=False))
        for t in pts:
            d2 = df.copy()
            cut = t + 2 + H  # premières lignes > t+1+H
            k = n - cut
            for c in ("open", "high", "low", "close"):
                d2.iloc[cut:, d2.columns.get_loc(c)] = d2[c].iloc[cut:].to_numpy() * np.exp(rng.normal(0, 0.05, k))
            d2.iloc[cut:, d2.columns.get_loc("volume")] = rng.exponential(100.0, k)
            y2 = make_label(d2, spec)
            a, b = y0.iloc[: t + 1].to_numpy(), y2.iloc[: t + 1].to_numpy()
            assert np.array_equal(a, b, equal_nan=True), f"FUITE {spec} t={t}"
            hit = False
            for fac in (0.5, 2.0):
                d4 = df.copy()
                d4.iloc[t + 1 + H, d4.columns.get_loc("open")] = df["open"].iloc[t + 1 + H] * fac
                y4 = make_label(d4, spec)
                hit |= not np.array_equal(y0.iloc[: t + 1].to_numpy(), y4.iloc[: t + 1].to_numpy(), equal_nan=True)
            changed_sens += int(hit)
        rate = float(np.nanmean(y0.to_numpy()))
        log(f"[anti-fuite] {spec} : {n_points} points t, données > t+1+H modifiées -> labels <= t "
            f"identiques : OK ; sensibilité (open[t+1+H] x0,5 ou x2 change un label <= t) : {changed_sens}/{n_points} ; "
            f"taux label=1 synthétique {rate:.4f}")
        assert changed_sens > 0
