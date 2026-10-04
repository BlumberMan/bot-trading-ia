"""Avocat du diable P3-A2 : tests sur le signal OOS de E17 reproduit par MA relance
(reports/adversarial/E17_adv_base_signal_full.csv, produit par run_nested_variant.py base)
et sur mes variantes re-tunées (reports/adversarial/E17_adv_*.json).

Usage : .venv\\Scripts\\python tests\\adversarial\\analyse_e17.py [--draws 1000] [--seed 777]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_e10 import FOLDS, bt_from_h, random_h, segments, ts  # noqa: E402

from bot.backtest import run_backtest  # noqa: E402
from bot.baseline import run_buy_and_hold  # noqa: E402
from bot.metrics import daily_returns, sharpe_daily, summarize  # noqa: E402
from bot.split import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "adversarial"
RES = ROOT / "experiments" / "results"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
A = pd.Timestamp("2021-01-01 00:00", tz="UTC")
B = MAX_INDEX
COST = 0.0015
DECL = 0.235865


def sh_of(p: Path):
    j = json.loads(p.read_text(encoding="utf-8"))
    m = j["concatenated"]["model"]
    return j, float(m["sharpe"])


def ann(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * math.sqrt(365)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} <= {MAX_INDEX} ({len(df)} lignes)")
    o = df["open"]
    sig = pd.read_csv(OUT / "E17_adv_base_signal_full.csv", index_col=0, parse_dates=True)["signal"]
    sig.index = pd.DatetimeIndex(sig.index).tz_convert("UTC") if sig.index.tz else sig.index.tz_localize("UTC")
    assert sig.index.equals(df.index)
    res = run_backtest(sig, o, start=A, end=B, exec_delay=1, cost_per_side=COST)
    m = summarize(res)
    S = float(m["sharpe"])
    jb, Sb = sh_of(OUT / "E17_adv_base.json")
    print(f"\n=== T0 reproduction (ma relance complète, seed 42) ===")
    print(f"Sharpe {S:.6f} (JSON {Sb:.6f}) trades {m['trades']} expo {m['exposure']:.6f} DD {m['max_drawdown']:.4f} "
          f"rdt total {m['total_return']:.4f} ; déclaré 0.235865 : |écart| {abs(S - DECL):.2e}")
    jr = json.loads((RES / "E17.json").read_text(encoding="utf-8"))
    same_choice = [(a["chosen_config"], a["best_hp"], a["best_map"]) == (b["chosen_config"], b["best_hp"], b["best_map"])
                   for a, b in zip(jb["folds"], jr["folds"])]
    print(f"choix par fold identiques au chercheur : {same_choice}")
    bh = run_buy_and_hold(df, A, B)
    print(f"B&H Sharpe {sharpe_daily(bh.equity):.6f} rdt {bh.equity.iloc[-1] - 1:.4f}")
    out = {"S": S, "repro": m}

    # ------------------------------------------------------------ T1 labels mélangés
    print("\n=== T1 labels mélangés, procédure emboîtée complète ===")
    pool = []
    for s in range(1, 11):
        j, v = sh_of(RES / f"E17_shuf{s}.json")
        pool.append(("chercheur", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    mine = []
    for s in range(11, 16):
        j, v = sh_of(OUT / f"E17_adv_shuf{s}.json")
        mine.append(v)
        pool.append(("avocat", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    for who, s, v, t, e in pool:
        print(f"  {who:<9} seed {s:>2} : Sharpe {v:+.4f} trades {t} expo {e:.3f}")
    allv = np.array([p[2] for p in pool])
    mine = np.array(mine)
    p95 = float(np.percentile(allv, 95))
    z = (S - allv.mean()) / allv.std(ddof=1)
    print(f"mes 5 seeds : moyenne {mine.mean():+.4f} écart-type {mine.std(ddof=1):.4f}")
    print(f"pool 15 seeds : moyenne {allv.mean():+.4f} écart-type {allv.std(ddof=1):.4f} p95 {p95:+.4f} max {allv.max():+.4f}")
    print(f"E17 {S:+.4f} : z {z:+.3f} ; seeds mélangées >= E17 : {int((allv >= S).sum())}/15 ; S > p95 : {S > p95}")
    out["shuffle"] = {"mine_mean": mine.mean(), "pool_mean": allv.mean(), "pool_sd": allv.std(ddof=1), "p95": p95, "z": z}

    # ------------------------------------------------------------ T2 aléatoire
    a_i = int(df.index.searchsorted(A))
    n = len(res.position)
    h0 = res.position.to_numpy()
    s0, _ = segments(h0)
    K, L = len(s0), int((h0 > 0).sum())
    print(f"\n[cible] segments {K}, barres exposées {L}/{n}, position héritée avant 2021 = {res.initial_position}")
    rng = np.random.default_rng(args.seed)
    shs, trs = [], []
    for _ in range(args.draws):
        r = bt_from_h(random_h(rng, n, K, L), df, a_i)
        shs.append(sharpe_daily(r.equity))
        trs.append(len(r.trades))
    shs = np.array(shs)
    pct = float(np.mean(shs < S) * 100)
    print(f"=== T2 aléatoire (seed {args.seed}, {args.draws} tirages, {K} segments, expo {L / n:.4f}) ===")
    print(f"trades tirés min/max {min(trs)}/{max(trs)} ; Sharpe moyenne {shs.mean():.4f} p50 {np.percentile(shs, 50):.4f} "
          f"p95 {np.percentile(shs, 95):.4f} max {shs.max():.4f} ; E17 percentile {pct:.1f}")
    out["random"] = {"percentile": pct, "p95": np.percentile(shs, 95)}

    # ------------------------------------------------------------ T2b B&H même exposition
    frac = L / n
    rc = run_backtest(pd.Series(frac, index=df.index), o, start=A, end=B, initial_position=0.0)
    src = summarize(rc)
    print(f"\n=== T2b B&H même exposition ===\nB&H fractionnaire constant {frac:.4f} : Sharpe {sharpe_daily(rc.equity):.4f} "
          f"rdt {rc.equity.iloc[-1] - 1:+.4f} DD {src['max_drawdown']:.4f} | E17 rdt {m['total_return']:+.4f} "
          f"DD {m['max_drawdown']:.4f}")
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
    for _ in range(args.draws):
        parts = []
        for nb, k, lf in plan:
            if lf == 0:
                parts.append(np.zeros(nb))
            elif k == 1:
                st = rng2.integers(0, nb - lf + 1)
                x = np.zeros(nb)
                x[st:st + lf] = 1.0
                parts.append(x)
            else:
                parts.append(random_h(rng2, nb, k, lf))
        shs2.append(sharpe_daily(bt_from_h(np.concatenate(parts), df, a_i).equity))
    shs2 = np.array(shs2)
    pct2 = float(np.mean(shs2 < S) * 100)
    print(f"stratifié par fold (seed {args.seed + 1}, {args.draws} tirages) : moyenne {shs2.mean():.4f} "
          f"p95 {np.percentile(shs2, 95):.4f} ; E17 percentile {pct2:.1f}")
    out["bh_expo"] = {"const": sharpe_daily(rc.equity), "strat_pct": pct2}

    # ------------------------------------------------------------ T3/T4 figé + re-tuné
    print("\n=== T3/T4 signal E17 figé (modèles et seuils inchangés) ===")
    for lab, kw in [("exec_delay=2", dict(exec_delay=2)), ("coûts x2", dict(cost_multiplier=2.0)),
                    ("coûts x3", dict(cost_multiplier=3.0))]:
        r = run_backtest(sig, o, start=A, end=B, cost_per_side=COST, **{"exec_delay": 1, **kw})
        s = summarize(r)
        print(f"{lab:<14}: Sharpe {s['sharpe']:+.4f} ({s['sharpe'] / S * 100:.0f} % de E17) trades {s['trades']} "
              f"DD {s['max_drawdown']:.4f} rdt {s['total_return']:+.4f}")
        out[f"frozen {lab}"] = s["sharpe"]
    print("(features retardées d'une bougie avec modèles et seuils figés = signal décalé d'une bougie = exec t+2 ci-dessus)")
    print("\n=== T3/T4/T6a procédure emboîtée re-tunée (mes relances) ===")
    for v in ("delay2", "featlag1", "cost2", "cost3", "drop_sma_dev_24"):
        j, x = sh_of(OUT / f"E17_adv_{v}.json")
        mm = j["concatenated"]["model"]
        print(f"{v:<16}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de E17) trades {mm['trades']} expo {mm['exposure']:.3f} "
              f"DD {mm['max_drawdown']:.4f} rdt {mm['total_return']:+.4f} folds>0 {j['folds_sharpe_positive']}/9 ; "
              f"choix {';'.join(r['chosen_config'] for r in j['folds'])}")
        out[f"retuned {v}"] = x
    for v in ("delay2", "featlag1", "drop_sma_dev_24"):
        _, x = sh_of(RES / f"E17_{v}.json")
        print(f"  (chercheur {v} : {x:+.4f})")
    print(f"feature la plus importante de MA base : {jb['most_important_feature']}")

    # ------------------------------------------------------------ T5
    print("\n=== T5a par année (signal continu, position héritée) ===")
    for y in range(2021, 2026):
        e = ts(f"{y}-12-31", True) if y < 2025 else B
        r = run_backtest(sig, o, start=ts(f"{y}-01-01"), end=e, cost_per_side=COST)
        rb = run_buy_and_hold(df, ts(f"{y}-01-01"), e)
        s = summarize(r)
        print(f"{y} : E17 Sharpe {s['sharpe']:+.4f} rdt {s['total_return']:+.4f} trades {s['trades']} "
              f"expo {s['exposure']:.3f} | B&H Sharpe {sharpe_daily(rb.equity):+.4f} rdt {rb.equity.iloc[-1] - 1:+.4f}")
    print("\n=== T5b par régime a priori (mois civil B&H : hausse > +5 %, baisse < -5 %, range sinon) ===")
    dr = daily_returns(res.equity)
    drb = daily_returns(bh.equity)
    mo = o.loc[A:B].resample("MS").first()
    nxt = mo.shift(-1)
    nxt.iloc[-1] = o.loc[B]
    mret = nxt / mo - 1
    reg = pd.Series(np.where(mret > 0.05, "hausse", np.where(mret < -0.05, "baisse", "range")), index=mo.index)
    dmonth = dr.index.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    day_reg = reg.reindex(dmonth).to_numpy()
    for g in ("hausse", "baisse", "range"):
        mk = day_reg == g
        x, xb = dr[mk], drb[mk]
        print(f"{g:<7}: {int((reg == g).sum())} mois, {int(mk.sum())} jours | E17 Sharpe {ann(x):+.4f} "
              f"rdt composé {np.prod(1 + x) - 1:+.4f} | B&H Sharpe {ann(xb):+.4f} rdt composé {np.prod(1 + xb) - 1:+.4f}")

    # ------------------------------------------------------------ T6b
    print("\n=== T6b sans le meilleur mois ===")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    best = mr.idxmax()
    print(f"meilleur mois E17 : {best.strftime('%Y-%m')} rdt {mr.max():+.4f} ; top 3 : "
          + ", ".join(f"{i.strftime('%Y-%m')} {v:+.4f}" for i, v in mr.sort_values(ascending=False).head(3).items()))
    top3 = mr.sort_values(ascending=False).head(3).index
    sh_drop = ann(dr[dmonth != best])
    print(f"Sharpe sans les jours du meilleur mois : {sh_drop:.4f} ({sh_drop / S * 100:.0f} % de E17) ; "
          f"sans les 3 meilleurs mois : {ann(dr[~np.isin(dmonth, top3)]):.4f}")
    out["no_best_month"] = sh_drop

    # ------------------------------------------------------------ T7 DSR
    print("\n=== T7 Sharpe déflaté (Bailey & López de Prado 2014) ===")
    reg_lines = (ROOT / "experiments" / "REGISTRE.md").read_text(encoding="utf-8").splitlines()
    N = len(reg_lines)
    ids = [ln.split("|")[1].strip() for ln in reg_lines]
    srd = np.array([float(json.loads((RES / f"{i}.json").read_text(encoding="utf-8"))["concatenated"]["model"]["sharpe"])
                    / math.sqrt(365) for i in ids])
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
    print(f"N = {N} lignes (ids {ids[0]}..{ids[-1]}) ; V = {V:.6e} ; SR0 journalier {sr0:.6f} (annualisé "
          f"{sr0 * math.sqrt(365):.4f})")
    print(f"SR journalier E17 {sr:.6f} ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; DSR = {dsr:.6f} ; PSR(0) = {psr:.6f}")
    out["dsr"] = {"N": N, "dsr": dsr, "psr0": psr, "sr0": sr0}
    (OUT / "analyse_e17.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
