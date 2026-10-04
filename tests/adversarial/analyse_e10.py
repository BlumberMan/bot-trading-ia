"""Avocat du diable P3-A1 : tests sur le signal OOS de E10 reproduit (reports/adversarial/E10_adv_base_*).

Usage : .venv\\Scripts\\python tests\\adversarial\\analyse_e10.py [--draws 1000] [--seed 20261004]
Prérequis : tests/adversarial/run_variant.py base (signal complet sauvegardé).
Données : bot.split.load_dataset() par défaut, assertion index max <= 2025-09-30 23:00 UTC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from bot.backtest import run_backtest
from bot.baseline import run_buy_and_hold
from bot.metrics import daily_returns, sharpe_daily, summarize
from bot.split import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
A = pd.Timestamp("2021-01-01 00:00", tz="UTC")
B = MAX_INDEX
COST = 0.0015
CAND = 0.6694973455033209
FOLDS = [("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"), ("2022-01-01", "2022-06-30"),
         ("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
         ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-09-30")]


def ts(s, end=False):
    return pd.Timestamp(f"{s} {'23:00' if end else '00:00'}", tz="UTC")


def segments(h: np.ndarray):
    """(début, fin exclue) des segments h > 0."""
    x = np.concatenate([[0], (h > 0).astype(np.int8), [0]])
    d = np.diff(x)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def random_h(rng, n: int, k: int, L: int) -> np.ndarray:
    """k segments longs (longueurs >= 1, somme L) séparés par des plats (internes >= 1)."""
    seg = np.diff(np.concatenate([[0], np.sort(rng.choice(np.arange(1, L), k - 1, replace=False)), [L]]))
    R = n - L - (k - 1)
    cuts = np.sort(rng.choice(np.arange(R + k), k, replace=False))
    gaps = np.diff(np.concatenate([[-1], cuts, [R + k]])) - 1
    gaps[1:-1] += 1
    h = np.zeros(n)
    pos = 0
    for i in range(k):
        pos += gaps[i]
        h[pos: pos + seg[i]] = 1.0
        pos += seg[i]
    assert pos + gaps[k] == n
    return h


def bt_from_h(h_win: np.ndarray, df: pd.DataFrame, a_i: int, cost_mult=1.0, start=A, end=B):
    """Signal tel que la position tenue (exec_delay=1) sur la fenêtre = h_win."""
    sig = np.zeros(len(df))
    sig[a_i - 1: a_i - 1 + len(h_win)] = h_win
    return run_backtest(pd.Series(sig, index=df.index), df["open"], start=start, end=end,
                        exec_delay=1, cost_per_side=COST, cost_multiplier=cost_mult)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} <= {MAX_INDEX} ({len(df)} lignes)")
    o = df["open"]
    sig = pd.read_csv(OUT / "E10_adv_base_signal_full.csv", index_col=0, parse_dates=True)["signal"]
    sig.index = pd.DatetimeIndex(sig.index).tz_convert("UTC") if sig.index.tz else sig.index.tz_localize("UTC")
    assert sig.index.equals(df.index)
    res = run_backtest(sig, o, start=A, end=B, exec_delay=1, cost_per_side=COST)
    m = summarize(res)
    print(f"[repro] E10 concat : Sharpe {m['sharpe']:.6f} trades {m['trades']} expo {m['exposure']:.6f} "
          f"DD {m['max_drawdown']:.4f} rdt {m['total_return']:.4f}")
    assert abs(m["sharpe"] - CAND) < 1e-12
    bh = run_buy_and_hold(df, A, B)
    print(f"[B&H] Sharpe {sharpe_daily(bh.equity):.6f}")
    a_i = int(df.index.searchsorted(A))
    n = len(res.position)
    h0 = res.position.to_numpy()
    s0, e0 = segments(h0)
    K, L = len(s0), int((h0 > 0).sum())
    nan_open = int(o.loc[A:B].isna().sum())
    print(f"[cible] segments {K}, barres exposées {L}/{n}, opens NaN dans la fenêtre {nan_open}, "
          f"position héritée avant 2021 = {res.initial_position}")
    out = {"repro": m}

    # ---------------------------------------------------------------- 2. aléatoire même trades / expo
    rng = np.random.default_rng(args.seed)
    shs, trs, exs = [], [], []
    for d in range(args.draws):
        r = bt_from_h(random_h(rng, n, K, L), df, a_i)
        shs.append(sharpe_daily(r.equity))
        trs.append(len(r.trades))
        exs.append(float((r.position > 0).mean()))
    shs = np.array(shs)
    pct = float(np.mean(shs < m["sharpe"]) * 100)
    print(f"\n=== T2 aléatoire (seed {args.seed}, {args.draws} tirages, {K} segments, expo {L / n:.4f}, coûts 0,0015 x1) ===")
    print(f"trades tirés min/max {min(trs)}/{max(trs)} ; expo min/max {min(exs):.4f}/{max(exs):.4f}")
    print(f"Sharpe aléatoire : moyenne {shs.mean():.4f} écart-type {shs.std(ddof=1):.4f} "
          f"p5 {np.percentile(shs, 5):.4f} p50 {np.percentile(shs, 50):.4f} p95 {np.percentile(shs, 95):.4f} "
          f"max {shs.max():.4f}")
    print(f"E10 {m['sharpe']:.4f} -> percentile {pct:.1f} ; > p95 : {m['sharpe'] > np.percentile(shs, 95)}")
    out["random"] = {"seed": args.seed, "draws": args.draws, "mean": shs.mean(), "p95": np.percentile(shs, 95),
                     "percentile": pct}

    # ---------------------------------------------------------------- 2b. B&H même exposition
    print("\n=== T2b B&H avec la même exposition ===")
    frac = L / n
    rc = run_backtest(pd.Series(frac, index=df.index), o, start=A, end=B, initial_position=0.0)
    print(f"B&H fractionnaire constant {frac:.4f} : Sharpe {sharpe_daily(rc.equity):.4f} "
          f"rdt {rc.equity.iloc[-1] - 1:.4f} DD {summarize(rc)['max_drawdown']:.4f}")
    # stratifié par fold : même nb de segments et même exposition que E10 DANS chaque fold
    idx = res.position.index
    plan = []
    for a_s, b_s in FOLDS:
        msk = (idx >= ts(a_s)) & (idx <= ts(b_s, True))
        hf = h0[msk]
        sf, _ = segments(hf)
        plan.append((int(msk.sum()), max(len(sf), 1), int((hf > 0).sum())))
    print("plan par fold (barres, segments, barres exposées) :", plan)
    rng2 = np.random.default_rng(args.seed + 1)
    shs2 = []
    for d in range(args.draws):
        parts = []
        for nb, k, lf in plan:
            if lf == 0:
                parts.append(np.zeros(nb))
            elif k == 1:
                st = rng2.integers(0, nb - lf + 1)
                x = np.zeros(nb); x[st:st + lf] = 1.0
                parts.append(x)
            else:
                parts.append(random_h(rng2, nb, k, lf))
        shs2.append(sharpe_daily(bt_from_h(np.concatenate(parts), df, a_i).equity))
    shs2 = np.array(shs2)
    pct2 = float(np.mean(shs2 < m["sharpe"]) * 100)
    print(f"stratifié par fold (seed {args.seed + 1}, {args.draws} tirages) : moyenne {shs2.mean():.4f} "
          f"p50 {np.percentile(shs2, 50):.4f} p95 {np.percentile(shs2, 95):.4f} ; E10 percentile {pct2:.1f}")
    out["bh_same_expo"] = {"constant_frac_sharpe": sharpe_daily(rc.equity), "stratified_mean": shs2.mean(),
                           "stratified_p95": np.percentile(shs2, 95), "stratified_percentile": pct2}

    # ---------------------------------------------------------------- 3/4. signal figé : délai, coûts
    print("\n=== T3/T4 signal E10 figé (modèles et seuils inchangés) ===")
    for lab, kw in [("exec_delay=2", dict(exec_delay=2)), ("coûts x2", dict(cost_multiplier=2.0)),
                    ("coûts x3", dict(cost_multiplier=3.0))]:
        r = run_backtest(sig, o, start=A, end=B, cost_per_side=COST, **{"exec_delay": 1, **kw})
        s = summarize(r)
        print(f"{lab:<14}: Sharpe {s['sharpe']:.4f} trades {s['trades']} expo {s['exposure']:.4f} "
              f"DD {s['max_drawdown']:.4f} rdt {s['total_return']:.4f}")
        out[f"frozen_{lab}"] = s["sharpe"]

    # ---------------------------------------------------------------- 5. sous-périodes
    print("\n=== T5a par année (signal continu, position héritée) ===")
    for y in range(2021, 2026):
        e = ts(f"{y}-12-31", True) if y < 2025 else B
        r = run_backtest(sig, o, start=ts(f"{y}-01-01"), end=e, cost_per_side=COST)
        rb = run_buy_and_hold(df, ts(f"{y}-01-01"), e)
        s = summarize(r)
        print(f"{y} : E10 Sharpe {s['sharpe']:+.4f} rdt {s['total_return']:+.4f} trades {s['trades']} "
              f"expo {s['exposure']:.3f} | B&H Sharpe {sharpe_daily(rb.equity):+.4f} rdt {rb.equity.iloc[-1] - 1:+.4f}")
    print("\n=== T5b par régime (défini a priori : rendement B&H du mois civil, open 1re bougie -> open "
          "1re bougie du mois suivant ; hausse > +5 %, baisse < -5 %, range sinon) ===")
    dr = daily_returns(res.equity)
    drb = daily_returns(bh.equity)
    mo = o.loc[A:B].resample("MS").first()
    nxt = mo.shift(-1)
    nxt.iloc[-1] = o.loc[B]  # dernier mois : marquage au dernier open de la période dev
    mret = nxt / mo - 1
    reg = pd.Series(np.where(mret > 0.05, "hausse", np.where(mret < -0.05, "baisse", "range")), index=mo.index)
    day_reg = reg.reindex(dr.index.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")).to_numpy()
    for g in ("hausse", "baisse", "range"):
        mk = day_reg == g
        x, xb = dr[mk], drb[mk]
        print(f"{g:<7}: {int((reg == g).sum())} mois, {int(mk.sum())} jours | E10 Sharpe "
              f"{x.mean() / x.std(ddof=1) * math.sqrt(365):+.4f} rdt composé {np.prod(1 + x) - 1:+.4f} | "
              f"B&H Sharpe {xb.mean() / xb.std(ddof=1) * math.sqrt(365):+.4f} rdt composé {np.prod(1 + xb) - 1:+.4f}")

    # ---------------------------------------------------------------- 6b. sans le meilleur mois
    print("\n=== T6b sans le meilleur mois ===")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    best = mr.idxmax()
    print(f"meilleur mois E10 : {best.strftime('%Y-%m')} rdt {mr.max():+.4f} ; top 3 : "
          + ", ".join(f"{i.strftime('%Y-%m')} {v:+.4f}" for i, v in mr.sort_values(ascending=False).head(3).items()))
    dmonth = dr.index.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    x = dr[dmonth != best]
    sh_drop = x.mean() / x.std(ddof=1) * math.sqrt(365)
    x0 = dr.where(dmonth != best, 0.0)
    sh_flat = x0.mean() / x0.std(ddof=1) * math.sqrt(365)
    top3 = mr.sort_values(ascending=False).head(3).index
    x3 = dr[~np.isin(dmonth, top3)]
    print(f"Sharpe sans les jours du meilleur mois : {sh_drop:.4f} ; meilleur mois mis à plat : {sh_flat:.4f} ; "
          f"sans les 3 meilleurs mois : {x3.mean() / x3.std(ddof=1) * math.sqrt(365):.4f}")
    out["no_best_month"] = sh_drop

    # ---------------------------------------------------------------- 7. DSR
    print("\n=== T7 Sharpe déflaté (Bailey & López de Prado 2014), recalcul indépendant ===")
    reg_lines = (ROOT / "experiments" / "REGISTRE.md").read_text(encoding="utf-8").splitlines()
    N = len(reg_lines)
    ids = [l.split("|")[1].strip() for l in reg_lines]
    srd = []
    for i in ids:
        j = json.loads((ROOT / "experiments" / "results" / f"{i}.json").read_text(encoding="utf-8"))
        s_ann = float(j["concatenated"]["model"]["sharpe"])
        srd.append(s_ann / math.sqrt(365))  # SR journalier = Sharpe annualisé / sqrt(365)
    srd = np.array(srd)
    V = srd.var(ddof=1)
    r_ = dr.to_numpy()
    T = len(r_)
    sr = r_.mean() / r_.std(ddof=1)
    g3, g4 = skew(r_, bias=False), kurtosis(r_, fisher=False, bias=False)
    gam = 0.5772156649015329
    sr0 = math.sqrt(V) * ((1 - gam) * norm.ppf(1 - 1 / N) + gam * norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    dsr = norm.cdf((sr - sr0) * math.sqrt(T - 1) / den)
    psr = norm.cdf(sr * math.sqrt(T - 1) / den)
    print(f"N = {N} lignes de REGISTRE.md ; SR journaliers des essais : {np.round(srd, 5).tolist()}")
    print(f"V = {V:.6e} ; SR0 journalier = {sr0:.6f} (annualisé {sr0 * math.sqrt(365):.4f})")
    print(f"SR E10 journalier = {sr:.6f} ; T = {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; dénominateur {den:.6f}")
    print(f"DSR = {dsr:.6f} ; PSR(0) = {psr:.6f}")
    out["dsr"] = {"N": N, "V": V, "sr0": sr0, "sr": sr, "T": T, "dsr": dsr, "psr0": psr}
    (OUT / "analyse_e10.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
