"""Avocat du diable P3-A3 : tests sur le signal OOS de E24 reproduit par MA relance
(reports/adversarial/E24_adv_base_signal_full.csv / _proba_test.csv, produits par run_b4_variant.py base)
et sur mes variantes re-tunées (reports/adversarial/E24_adv_*.json).

Usage : .venv\\Scripts\\python tests\\adversarial\\analyse_e24.py [--draws 1000] [--seed 2424]
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

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_e10 import FOLDS, bt_from_h, random_h, segments, ts  # noqa: E402
from run_b4_variant import e24_barriers, tb_paths  # noqa: E402

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
DECL = 0.799410


def sh_of(p: Path):
    j = json.loads(p.read_text(encoding="utf-8"))
    return j, float(j["concatenated"]["model"]["sharpe"])


def ann(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * math.sqrt(365)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = ~(np.isnan(y) | np.isnan(p))
    return float(roc_auc_score(y[m], p[m])), int(m.sum())


def line(j, x, S, lab):
    mm = j["concatenated"]["model"]
    print(f"{lab:<16}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) trades {mm['trades']} expo {mm['exposure']:.3f} "
          f"DD {mm['max_drawdown']:.4f} rdt {mm['total_return']:+.4f} folds>0 {j['folds_sharpe_positive']}/9 ; "
          f"Sharpe/fold {';'.join('%.3f' % r['model']['sharpe'] for r in j['folds'])} ; "
          f"top feature {j['most_important_feature']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2424)
    args = ap.parse_args()
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} <= {MAX_INDEX} ({len(df)} lignes)")
    o = df["open"]
    sig = pd.read_csv(OUT / "E24_adv_base_signal_full.csv", index_col=0, parse_dates=True)["signal"]
    sig.index = pd.DatetimeIndex(sig.index).tz_convert("UTC") if sig.index.tz else sig.index.tz_localize("UTC")
    assert sig.index.equals(df.index)
    res = run_backtest(sig, o, start=A, end=B, exec_delay=1, cost_per_side=COST)
    m = summarize(res)
    S = float(m["sharpe"])
    jb, Sb = sh_of(OUT / "E24_adv_base.json")
    print("\n=== T0 reproduction (ma relance complète, seed 42) ===")
    print(f"Sharpe {S:.6f} (JSON {Sb:.6f}) trades {m['trades']} expo {m['exposure']:.6f} DD {m['max_drawdown']:.4f} "
          f"rdt total {m['total_return']:.4f} ; déclaré {DECL} : |écart| {abs(S - DECL):.2e}")
    jr = json.loads((RES / "E24.json").read_text(encoding="utf-8"))
    same = [(a["best_hp"], a["best_map"]) == (b["best_hp"], b["best_map"]) for a, b in zip(jb["folds"], jr["folds"])]
    print(f"choix (hp, seuils) par fold identiques au chercheur : {same}")
    bh = run_buy_and_hold(df, A, B)
    print(f"B&H Sharpe {sharpe_daily(bh.equity):.6f} rdt {bh.equity.iloc[-1] - 1:.4f}")
    print("importances moyennes (ma base) : " + ", ".join(
        f"{k} {v:.4f}" for k, v in sorted(jb["importance_norm_mean"].items(), key=lambda kv: -kv[1]) if v > 0))
    out = {"S": S, "repro": m}

    # ------------------------------------------------------------ T1
    print("\n=== T1 labels mélangés, procédure complète ===")
    pool = []
    for s in range(1, 11):
        j, v = sh_of(RES / f"E24_shuf{s}.json")
        pool.append(("chercheur", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    mine = []
    for s in range(11, 16):
        j, v = sh_of(OUT / f"E24_adv_shuf{s}.json")
        mine.append(v)
        pool.append(("avocat", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    for who, s, v, t, e in pool:
        print(f"  {who:<9} seed {s:>2} : Sharpe {v:+.4f} trades {t} expo {e:.3f}")
    allv, mine = np.array([p[2] for p in pool]), np.array(mine)
    p95 = float(np.percentile(allv, 95))
    z = (S - allv.mean()) / allv.std(ddof=1)
    print(f"mes 5 seeds : moyenne {mine.mean():+.4f} écart-type {mine.std(ddof=1):.4f} max {mine.max():+.4f}")
    print(f"pool 15 seeds : moyenne {allv.mean():+.4f} écart-type {allv.std(ddof=1):.4f} p95 {p95:+.4f} max {allv.max():+.4f}")
    print(f"E24 {S:+.4f} : z {z:+.3f} ; seeds >= E24 : {int((allv >= S).sum())}/15 ; S > p95 : {S > p95}")
    out["shuffle"] = {"pool_mean": allv.mean(), "pool_sd": allv.std(ddof=1), "p95": p95, "z": z}

    # ------------------------------------------------------------ T2
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
          f"p95 {np.percentile(shs, 95):.4f} max {shs.max():.4f} ; E24 percentile {pct:.1f}")
    out["random"] = {"percentile": pct, "p95": np.percentile(shs, 95)}

    # ------------------------------------------------------------ T2b
    frac = L / n
    rc = run_backtest(pd.Series(frac, index=df.index), o, start=A, end=B, initial_position=0.0)
    src = summarize(rc)
    print(f"\n=== T2b B&H même exposition ===\nB&H fractionnaire constant {frac:.4f} : Sharpe {sharpe_daily(rc.equity):.4f} "
          f"rdt {rc.equity.iloc[-1] - 1:+.4f} DD {src['max_drawdown']:.4f} | E24 Sharpe {S:.4f} rdt {m['total_return']:+.4f} "
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
          f"p95 {np.percentile(shs2, 95):.4f} ; E24 percentile {pct2:.1f}")
    out["bh_expo"] = {"const": sharpe_daily(rc.equity), "strat_pct": pct2}

    # ------------------------------------------------------------ T3/T4 figé
    print("\n=== T3/T4 signal E24 figé (modèles et seuils inchangés) ===")
    for lab, kw in [("exec_delay=2", dict(exec_delay=2)), ("coûts x2", dict(cost_multiplier=2.0)),
                    ("coûts x3", dict(cost_multiplier=3.0))]:
        r = run_backtest(sig, o, start=A, end=B, cost_per_side=COST, **{"exec_delay": 1, **kw})
        s = summarize(r)
        print(f"{lab:<14}: Sharpe {s['sharpe']:+.4f} ({s['sharpe'] / S * 100:.0f} % de S) trades {s['trades']} "
              f"DD {s['max_drawdown']:.4f} rdt {s['total_return']:+.4f}")
        out[f"frozen {lab}"] = s["sharpe"]
    print("(features retardées d'une bougie, modèles et seuils figés = signal décalé d'une bougie = exec t+2 ci-dessus)")

    # ------------------------------------------------------------ re-tunés
    print("\n=== T3/T4/T6a/C1/C2 procédure b4 re-tunée (mes relances) ===")
    for v in ("delay2", "featlag1", "cost2", "cost3", f"drop_{jb['most_important_feature']}", "only_vol168",
              "rule_vol", "tbdir", "tbfixed", "nodow", "dowplac51", "dowplac52", "dowplac53"):
        p = OUT / f"E24_adv_{v}.json"
        if not p.exists():
            print(f"{v:<16}: ABSENT")
            continue
        j, x = sh_of(p)
        line(j, x, S, v)
        out[f"retuned {v}"] = x
    pl = [out.get(f"retuned dowplac{s}") for s in (51, 52, 53)]
    if all(v is not None for v in pl):
        print(f"placebo jour de semaine : moyenne {np.mean(pl):+.4f} ({np.mean(pl) / S * 100:.0f} % de S)")
    for v in ("delay2", "featlag1", "cost2", "cost3", "drop_vol_168"):
        p = RES / f"E24_{v}.json"
        if p.exists():
            print(f"  (chercheur {v} : {sh_of(p)[1]:+.4f})")

    # ------------------------------------------------------------ C1e décomposition de l'AUC
    print("\n=== C1e décomposition de l'AUC OOS des probas E24 (ma base) ===")
    pt = pd.read_csv(OUT / "E24_adv_base_proba_test.csv", index_col=0, parse_dates=True)["p"]
    pt.index = pd.DatetimeIndex(pt.index).tz_convert("UTC") if pt.index.tz else pt.index.tz_localize("UTC")
    assert pt.index.equals(df.index)
    u, d = e24_barriers(df)
    ju, jd = tb_paths(df, 24, u, d)
    import labels_b4
    y = labels_b4.make_label(df, {"kind": "tb", "H": 24, "up": {"type": "vol", "k": 1.0},
                                  "down": {"type": "vol", "k": 1.0}}).to_numpy()
    ok = ~np.isnan(y)
    hit = np.where(ok, ((ju < np.inf) | (jd < np.inf)).astype(float), np.nan)
    dirv = np.where(ok & (hit == 1), (ju < jd).astype(float), np.nan)
    p = pt.to_numpy()
    tmask = (df.index >= A) & (df.index <= B)
    pz = np.where(tmask, p, np.nan)
    a_y, n_y = auc(y, pz)
    a_h, n_h = auc(hit, pz)
    a_d, n_d = auc(dirv, pz)
    vol = df.index.to_series().map(lambda _: 0).to_numpy(dtype=float)
    from bot.features import compute_features
    vol = compute_features(df)["vol_168"].to_numpy()
    a_vy, _ = auc(y, np.where(tmask, -vol, np.nan))
    a_vh, _ = auc(hit, np.where(tmask, -vol, np.nan))
    a_vd, _ = auc(dirv, np.where(tmask, -vol, np.nan))
    print(f"OOS 2021-01 -> 2025-09 : AUC(label) {a_y:.4f} (n {n_y}) ; AUC(hit) {a_h:.4f} (n {n_h}, taux hit "
          f"{np.nanmean(hit[tmask]):.4f}) ; AUC(direction | hit) {a_d:.4f} (n {n_d}, taux haut d'abord {np.nanmean(dirv[tmask]):.4f})")
    print(f"règle -vol_168 sur les mêmes lignes : AUC(label) {a_vy:.4f} ; AUC(hit) {a_vh:.4f} ; AUC(direction | hit) {a_vd:.4f}")
    rho = pd.Series(pz).corr(pd.Series(np.where(tmask, -vol, np.nan)), method="spearman")
    print(f"Spearman(proba E24, -vol_168) OOS : {rho:.4f}")
    out["auc"] = {"label": a_y, "hit": a_h, "dir": a_d, "vol_label": a_vy, "vol_hit": a_vh, "vol_dir": a_vd, "rho": rho}

    # ------------------------------------------------------------ T5
    print("\n=== T5a par année (signal continu, position héritée) ===")
    for yy in range(2021, 2026):
        e = ts(f"{yy}-12-31", True) if yy < 2025 else B
        r = run_backtest(sig, o, start=ts(f"{yy}-01-01"), end=e, cost_per_side=COST)
        rb = run_buy_and_hold(df, ts(f"{yy}-01-01"), e)
        s = summarize(r)
        print(f"{yy} : E24 Sharpe {s['sharpe']:+.4f} rdt {s['total_return']:+.4f} trades {s['trades']} "
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
        print(f"{g:<7}: {int((reg == g).sum())} mois, {int(mk.sum())} jours | E24 Sharpe {ann(x):+.4f} "
              f"rdt composé {np.prod(1 + x) - 1:+.4f} | B&H Sharpe {ann(xb):+.4f} rdt composé {np.prod(1 + xb) - 1:+.4f}")

    # ------------------------------------------------------------ T6b
    print("\n=== T6b sans le meilleur mois ===")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    best = mr.idxmax()
    top3 = mr.sort_values(ascending=False).head(3).index
    print(f"meilleur mois E24 : {best.strftime('%Y-%m')} rdt {mr.max():+.4f} ; top 3 : "
          + ", ".join(f"{i.strftime('%Y-%m')} {v:+.4f}" for i, v in mr.sort_values(ascending=False).head(3).items()))
    sh_drop = ann(dr[dmonth != best])
    print(f"Sharpe sans les jours du meilleur mois : {sh_drop:.4f} ({sh_drop / S * 100:.0f} % de S) ; "
          f"sans les 3 meilleurs mois : {ann(dr[~np.isin(dmonth, top3)]):.4f}")
    out["no_best_month"] = sh_drop

    # ------------------------------------------------------------ T7
    print("\n=== T7 Sharpe déflaté (Bailey & López de Prado 2014), recalcul indépendant ===")
    reg_lines = [ln for ln in (ROOT / "experiments" / "REGISTRE.md").read_text(encoding="utf-8").splitlines() if ln.strip()]
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
    # variante : variance restreinte aux essais E04-E28 (hors E01-E03 très négatifs)
    V2 = srd[3:].var(ddof=1)
    sr0b = math.sqrt(V2) * ((1 - gam) * norm.ppf(1 - 1 / N) + gam * norm.ppf(1 - 1 / (N * math.e)))
    dsr2 = norm.cdf((sr - sr0b) * math.sqrt(T - 1) / den)
    print(f"N = {N} lignes (ids {ids[0]}..{ids[-1]}) ; V = {V:.6e} ; SR0 journalier {sr0:.6f} (annualisé "
          f"{sr0 * math.sqrt(365):.4f})")
    print(f"SR journalier E24 {sr:.6f} ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; DSR = {dsr:.6f} ; PSR(0) = {psr:.6f}")
    print(f"(sensibilité) V sur E04-E28 seulement = {V2:.6e} ; SR0 annualisé {sr0b * math.sqrt(365):.4f} ; DSR = {dsr2:.6f}")
    out["dsr"] = {"N": N, "dsr": dsr, "psr0": psr, "sr0": sr0, "dsr_v_e04": dsr2}
    (OUT / "analyse_e24.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
