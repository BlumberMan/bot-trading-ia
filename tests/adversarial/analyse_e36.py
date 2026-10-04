"""Avocat du diable P3-A4 : tests sur le signal OOS de E36 reproduit par MA relance
(reports/adversarial/E36_adv_base_signal_1h.parquet / _proba_4h.parquet, produits par run_b5_variant.py base)
et sur mes variantes re-tunées (reports/adversarial/E36_adv_*.json).

Usage : .venv\\Scripts\\python tests\\adversarial\\analyse_e36.py [--draws 1000]
Seeds : T2a 3636, T2b 3637, T2c 3638, LT(a) 3639, LT(b) 3640.
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))
from analyse_e10 import FOLDS, bt_from_h, random_h, segments, ts  # noqa: E402

import agg4h  # noqa: E402
from bot.backtest import run_backtest  # noqa: E402
from bot.baseline import run_buy_and_hold  # noqa: E402
from bot.features import compute_features  # noqa: E402
from bot.metrics import daily_returns, sharpe_daily, summarize  # noqa: E402
from bot.split import load_dataset  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
RES = ROOT / "experiments" / "results"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
A = pd.Timestamp("2021-01-01 00:00", tz="UTC")
B = MAX_INDEX
H1 = pd.Timedelta(hours=1)
COST = 0.0015
DECL = 1.175146
H = 30
V = 0.061901


def sh_of(p: Path):
    j = json.loads(p.read_text(encoding="utf-8"))
    return j, float(j["concatenated"]["model"]["sharpe"])


def ann(x):
    x = np.asarray(x, dtype=float)
    return x.mean() / x.std(ddof=1) * math.sqrt(365)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = ~(np.isnan(y) | np.isnan(p))
    return float(roc_auc_score(y[m], p[m])), int(m.sum())


def place(rng, nb: int, lengths: np.ndarray) -> np.ndarray:
    """Segments de longueurs données (ordre permuté) placés uniformément, >= 1 bloc plat entre segments."""
    k = len(lengths)
    x = np.zeros(nb)
    if k == 0:
        return x
    seg = rng.permutation(lengths)
    R = nb - int(seg.sum()) - (k - 1)
    assert R >= 0
    cuts = np.sort(rng.choice(np.arange(R + k), k, replace=False))
    gaps = np.diff(np.concatenate([[-1], cuts, [R + k]])) - 1
    gaps[1:-1] += 1
    pos = 0
    for i in range(k):
        pos += gaps[i]
        x[pos: pos + seg[i]] = 1.0
        pos += seg[i]
    assert pos + gaps[k] == nb
    return x


def seg_lengths(h: np.ndarray) -> np.ndarray:
    s, e = segments(h)
    return (e - s).astype(int)


def line(j, x, S, lab):
    mm = j["concatenated"]["model"]
    print(f"{lab:<14}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) trades {mm['trades']} expo {mm['exposure']:.3f} "
          f"DD {mm['max_drawdown']:.4f} rdt {mm['total_return']:+.4f} folds>0 {j['folds_sharpe_positive']}/9 ; "
          f"Sharpe/fold {';'.join('%.3f' % float(r['model']['sharpe']) for r in j['folds'])} ; "
          f"top feature {j['most_important_feature']}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} <= {MAX_INDEX} ({len(df)} lignes)")
    o = df["open"]
    sig = pd.read_parquet(OUT / "E36_adv_base_signal_1h.parquet")["signal"]
    pr = pd.read_parquet(OUT / "E36_adv_base_proba_4h.parquet")
    assert sig.index.equals(df.index)
    res = run_backtest(sig, o, start=A, end=B, exec_delay=1, cost_per_side=COST)
    m = summarize(res)
    S = float(m["sharpe"])
    jb, Sb = sh_of(OUT / "E36_adv_base.json")
    print("\n=== T0 reproduction (ma relance complète, seed 42) ===")
    print(f"Sharpe {S:.6f} (JSON {Sb:.6f}) trades {m['trades']} expo {m['exposure']:.6f} DD {m['max_drawdown']:.4f} "
          f"rdt total {m['total_return']:.4f} ; déclaré {DECL} : |écart| {abs(S - DECL):.2e}")
    jr = json.loads((RES / "E36.json").read_text(encoding="utf-8"))
    same = [(a["best_hp"], a["best_map"]) == (b["best_hp"], b["best_map"]) for a, b in zip(jb["folds"], jr["folds"])]
    print(f"choix (hp, seuils) par fold identiques au chercheur : {same}")
    bh = run_buy_and_hold(df, A, B)
    SBH = sharpe_daily(bh.equity)
    print(f"B&H Sharpe {SBH:.6f} rdt {bh.equity.iloc[-1] - 1:.4f}")
    print("importances moyennes (ma base) : " + ", ".join(
        f"{k} {v:.4f}" for k, v in sorted(jb["importance_norm_mean"].items(), key=lambda kv: -kv[1]) if v > 0))
    print(f"par fold : trades {[r['model']['trades'] for r in jb['folds']]} ; trades de validation ayant servi au "
          f"choix {[r['best_val_trades'] for r in jb['folds']]} ; Sharpe val {[round(r['best_val_sharpe'], 3) for r in jb['folds']]}")
    out = {"S": S, "repro": m}

    # ------------------------------------------------------------ T1
    print("\n=== T1 labels mélangés, procédure complète re-tunée ===")
    pool = []
    for s in range(1, 11):
        j, v = sh_of(RES / f"E36_shuf{s}.json")
        pool.append(("chercheur", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    mine = []
    for s in range(11, 16):
        j, v = sh_of(OUT / f"E36_adv_shuf{s}.json")
        mine.append(v)
        pool.append(("avocat", s, v, j["concatenated"]["model"]["trades"], j["concatenated"]["model"]["exposure"]))
    for who, s, v, t, e in pool:
        print(f"  {who:<9} seed {s:>2} : Sharpe {v:+.4f} trades {t} expo {e:.3f}")
    allv, mine = np.array([p[2] for p in pool]), np.array(mine)
    p95 = float(np.percentile(allv, 95))
    z = (S - allv.mean()) / allv.std(ddof=1)
    zm = (S - mine.mean()) / mine.std(ddof=1)
    print(f"mes 5 seeds : moyenne {mine.mean():+.4f} écart-type {mine.std(ddof=1):.4f} max {mine.max():+.4f} ; "
          f"z sur mes seeds seules {zm:+.3f}")
    print(f"pool 15 seeds : moyenne {allv.mean():+.4f} écart-type {allv.std(ddof=1):.4f} p95 {p95:+.4f} max {allv.max():+.4f}")
    print(f"E36 {S:+.4f} : z {z:+.3f} ; seeds >= E36 : {int((allv >= S).sum())}/15 ; S > p95 : {S > p95} ; "
          f"seeds >= B&H {SBH:.4f} : {int((allv >= SBH).sum())}/15 ; seeds >= 0,85 S : {int((allv >= 0.85 * S).sum())}/15")
    out["shuffle"] = {"pool_mean": allv.mean(), "pool_sd": allv.std(ddof=1), "p95": p95, "z": z}

    # ------------------------------------------------------------ T2a grille 1 h
    a_i = int(df.index.searchsorted(A))
    n = len(res.position)
    h0 = res.position.to_numpy()
    s0, e0 = segments(h0)
    K, L = len(s0), int((h0 > 0).sum())
    print(f"\n[cible] segments {K}, heures exposées {L}/{n}, position héritée avant 2021 = {res.initial_position}")
    rng = np.random.default_rng(3636)
    shs, trs = [], []
    for _ in range(args.draws):
        r = bt_from_h(random_h(rng, n, K, L), df, a_i)
        shs.append(sharpe_daily(r.equity))
        trs.append(len(r.trades))
    shs = np.array(shs)
    pct = float(np.mean(shs < S) * 100)
    print(f"=== T2a aléatoire grille 1 h (seed 3636, {args.draws} tirages, {K} segments, expo {L / n:.4f}) ===")
    print(f"trades tirés min/max {min(trs)}/{max(trs)} ; Sharpe moyenne {shs.mean():.4f} p50 {np.percentile(shs, 50):.4f} "
          f"p95 {np.percentile(shs, 95):.4f} max {shs.max():.4f} ; E36 percentile {pct:.1f}")
    out["T2a"] = {"percentile": pct, "p95": np.percentile(shs, 95)}

    # ------------------------------------------------------------ T2b / T2c grille 4 h
    assert n % 4 == 0 and A.hour == 0
    blk = h0.reshape(-1, 4)
    nonconst = int((blk.min(axis=1) != blk.max(axis=1)).sum())
    h4 = (blk.mean(axis=1) >= 0.5).astype(float)
    L4 = seg_lengths(h4)
    nb = len(h4)
    print(f"\n=== T2b aléatoire grille 4 h (seed 3637, {args.draws} tirages) ===")
    print(f"blocs 4 h OOS {nb} ; blocs à position non constante (trous de prix) {nonconst} ; segments 4 h {len(L4)} ; "
          f"blocs exposés {int(L4.sum())} (expo {L4.sum() / nb:.4f}) ; durées (blocs 4 h) triées {sorted(L4.tolist())}")
    r_chk = bt_from_h(np.repeat(h4, 4), df, a_i)
    print(f"contrôle : position 4 h reconstruite rebacktestée -> Sharpe {sharpe_daily(r_chk.equity):.4f} "
          f"trades {len(r_chk.trades)} (S {S:.4f})")
    rng = np.random.default_rng(3637)
    s4, t4, x4 = [], [], []
    for _ in range(args.draws):
        r = bt_from_h(np.repeat(place(rng, nb, L4), 4), df, a_i)
        s4.append(sharpe_daily(r.equity))
        t4.append(len(r.trades))
        x4.append(float((r.position > 0).mean()))
    s4 = np.array(s4)
    pct4 = float(np.mean(s4 < S) * 100)
    print(f"trades tirés min/max {min(t4)}/{max(t4)} ; expo min/max {min(x4):.4f}/{max(x4):.4f}")
    print(f"Sharpe aléatoire 4 h : moyenne {s4.mean():.4f} sd {s4.std(ddof=1):.4f} p5 {np.percentile(s4, 5):.4f} "
          f"p50 {np.percentile(s4, 50):.4f} p95 {np.percentile(s4, 95):.4f} max {s4.max():.4f} ; E36 percentile {pct4:.1f} ; "
          f"tirages >= B&H : {int((s4 >= SBH).sum())}/{args.draws}")
    out["T2b"] = {"percentile": pct4, "p95": np.percentile(s4, 95)}

    idx4 = res.position.index[::4]
    plan = []
    for a_s, b_s in FOLDS:
        mk = (idx4 >= ts(a_s)) & (idx4 <= ts(b_s, True))
        plan.append((int(mk.sum()), seg_lengths(h4[mk])))
    print(f"\n=== T2c aléatoire grille 4 h stratifié par fold (seed 3638, {args.draws} tirages) ===")
    print("plan par fold (blocs, durées des morceaux) : " + " | ".join(f"{a}:{b.tolist()}" for a, b in plan))
    rng = np.random.default_rng(3638)
    s5, t5 = [], []
    for _ in range(args.draws):
        hh = np.concatenate([place(rng, a, b) for a, b in plan])
        r = bt_from_h(np.repeat(hh, 4), df, a_i)
        s5.append(sharpe_daily(r.equity))
        t5.append(len(r.trades))
    s5 = np.array(s5)
    pct5 = float(np.mean(s5 < S) * 100)
    print(f"trades tirés min/max {min(t5)}/{max(t5)} ; Sharpe moyenne {s5.mean():.4f} p50 {np.percentile(s5, 50):.4f} "
          f"p95 {np.percentile(s5, 95):.4f} max {s5.max():.4f} ; E36 percentile {pct5:.1f}")
    out["T2c"] = {"percentile": pct5, "p95": np.percentile(s5, 95)}

    # ------------------------------------------------------------ T2d
    frac = L / n
    rc = run_backtest(pd.Series(frac, index=df.index), o, start=A, end=B, initial_position=0.0)
    src_ = summarize(rc)
    print(f"\n=== T2d B&H même exposition ===\nB&H fractionnaire constant {frac:.4f} : Sharpe {sharpe_daily(rc.equity):.4f} "
          f"rdt {rc.equity.iloc[-1] - 1:+.4f} DD {src_['max_drawdown']:.4f} | E36 Sharpe {S:.4f}")
    out["T2d"] = sharpe_daily(rc.equity)

    # ------------------------------------------------------------ T3/T4 figé
    print("\n=== T3/T4 signal E36 figé (modèles et seuils inchangés) ===")
    for lab, kw in [("exec +1 bougie 4 h (delay 5 h)", dict(exec_delay=5)), ("exec +1 h (delay 2 h)", dict(exec_delay=2)),
                    ("coûts x2", dict(cost_multiplier=2.0)), ("coûts x3", dict(cost_multiplier=3.0))]:
        r = run_backtest(sig, o, start=A, end=B, cost_per_side=COST, **{"exec_delay": 1, **kw})
        s = summarize(r)
        print(f"{lab:<31}: Sharpe {s['sharpe']:+.4f} ({s['sharpe'] / S * 100:.0f} % de S) trades {s['trades']} "
              f"DD {s['max_drawdown']:.4f} rdt {s['total_return']:+.4f}")
        out[f"frozen {lab}"] = s["sharpe"]
    print("(features retardées d'une bougie 4 h avec modèles et seuils figés ~ signal décalé d'une bougie 4 h = ligne delay 5 h)")

    # ------------------------------------------------------------ re-tunés
    print("\n=== Variantes re-tunées (procédure b5.run complète, mes relances) ===")
    for v in ("delay2", "featlag1", "cost2", "cost3", f"drop_{jb['most_important_feature']}", "tbv08", "tbv12",
              "rule_vol_180", "rule_vol_42", "onlyvols", "tbdir"):
        p = OUT / f"E36_adv_{v}.json"
        if not p.exists():
            print(f"{v:<14}: ABSENT")
            continue
        j, x = sh_of(p)
        line(j, x, S, v)
        out[f"retuned {v}"] = x
    for v in ("delay2", "featlag1", "cost2", "cost3", "drop_vol_180", "rule_vol_180", "rule_vol_42", "tbdir"):
        p = RES / f"E36_{v}.json"
        if p.exists():
            print(f"  (chercheur {v} : {sh_of(p)[1]:+.4f})")

    # ------------------------------------------------------------ LT faible nombre de trades
    print("\n=== LT faible nombre de trades ===")
    tr = res.trades.copy()
    nr = tr["net_return"].to_numpy()
    print(f"trades {len(tr)} ; rendement net/trade : moyenne {nr.mean():+.4f} médiane {np.median(nr):+.4f} "
          f"sd {nr.std(ddof=1):.4f} ; gagnants {int((nr > 0).sum())} ; top 5 {np.sort(nr)[::-1][:5].round(4).tolist()} ; "
          f"pire 5 {np.sort(nr)[:5].round(4).tolist()} ; somme log top 3 / somme log totale "
          f"{np.sort(np.log1p(nr))[::-1][:3].sum() / np.log1p(nr).sum():.3f}")
    rng = np.random.default_rng(3639)
    bm, bs = [], []
    for _ in range(10000):
        x = nr[rng.integers(0, len(nr), len(nr))]
        bm.append(x.mean())
        bs.append(x.mean() / x.std(ddof=1))
    bm, bs = np.array(bm), np.array(bs)
    print(f"(a) bootstrap par trade (seed 3639, 10000) : moyenne/trade IC95 [{np.percentile(bm, 2.5):+.4f}, "
          f"{np.percentile(bm, 97.5):+.4f}] ; Sharpe par trade {nr.mean() / nr.std(ddof=1):.4f} IC95 "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] ; P(moyenne <= 0) {np.mean(bm <= 0):.4f}")
    # (b) Sharpe journalier par blocs de trades
    hidx = res.position.index
    tid = np.full(len(hidx), -1)
    for i, row in tr.iterrows():
        mk = (hidx >= row["entry_time"]) & (hidx <= row["exit_time"])
        tid[mk & (tid < 0)] = i
    dr = daily_returns(res.equity)
    days = hidx.floor("D")
    day_tid = pd.Series(np.where(tid >= 0, tid, 10 ** 6), index=hidx).groupby(days).min()
    day_tid = day_tid.where(day_tid < 10 ** 6, -1).reindex(dr.index).to_numpy()
    flat = dr.to_numpy()[day_tid < 0]
    print(f"(b) jours {len(dr)} ; jours rattachés à un trade {int((day_tid >= 0).sum())} ; jours sans activité "
          f"{len(flat)} (|rdt| max {np.abs(flat).max() if len(flat) else 0:.2e})")
    blocks = [dr.to_numpy()[day_tid == i] for i in range(len(tr))]
    rng = np.random.default_rng(3640)
    bsh = []
    ND = len(dr)
    for _ in range(10000):
        pick = rng.integers(0, len(tr), len(tr))
        xs = np.concatenate([blocks[i] for i in pick])
        nf = max(ND - len(xs), 0)
        arr = np.concatenate([xs, np.zeros(nf)])
        bsh.append(ann(arr))
    bsh = np.array(bsh)
    print(f"    bootstrap par trade du Sharpe journalier annualisé (seed 3640, 10000) : médiane {np.median(bsh):.4f} "
          f"IC95 [{np.percentile(bsh, 2.5):+.4f}, {np.percentile(bsh, 97.5):+.4f}] ; P(Sharpe <= 0) {np.mean(bsh <= 0):.4f} ; "
          f"P(Sharpe <= B&H {SBH:.4f}) {np.mean(bsh <= SBH):.4f} ; P(Sharpe <= 1,0) {np.mean(bsh <= 1.0):.4f}")
    out["LT"] = {"mean_ci": [np.percentile(bm, 2.5), np.percentile(bm, 97.5)],
                 "sharpe_ci": [np.percentile(bsh, 2.5), np.percentile(bsh, 97.5)]}
    # (c) sans le(s) meilleur(s) trade(s)
    order = np.argsort(-nr)
    for kk in (1, 3, 5):
        hh = h0.copy()
        for i in order[:kk]:
            row = tr.iloc[i]
            mk = (hidx >= row["entry_time"]) & (hidx < row["exit_time"])
            hh[mk] = 0.0
        r = bt_from_h(hh, df, a_i)
        sx = sharpe_daily(r.equity)
        print(f"(c) sans les {kk} meilleur(s) trade(s) {[(str(tr.iloc[i]['entry_time'])[:10], round(float(nr[i]), 4)) for i in order[:kk]]} : "
              f"Sharpe {sx:+.4f} ({sx / S * 100:.0f} % de S) trades {len(r.trades)} rdt {r.equity.iloc[-1] - 1:+.4f}")
        out[f"LT no top{kk}"] = sx
    print(f"(info GONOGO palier 4) trades OOS {m['trades']} (seuil >= 200) ; DD max {m['max_drawdown'] * 100:.2f} % (seuil < 25 %)")

    # ------------------------------------------------------------ NA features NaN
    print("\n=== NA features NaN (position figée) ===")
    df4 = agg4h.aggregate_4h(df)
    F4 = agg4h.compute_features_4h(df4)[agg4h.F4]
    assert pr.index.equals(df4.index)
    p4 = pr["p"].to_numpy()
    cov = (df4.index >= A - 8 * H1) & (df4.index <= B - 3 * H1)
    fnan = F4.isna().any(axis=1).to_numpy()
    oos_dec = (df4.index >= A - 4 * H1) & (df4.index <= B - 7 * H1)
    print(f"bougies 4 h de décision OOS {int(oos_dec.sum())} ; features NaN {int((fnan & oos_dec).sum())} "
          f"({(fnan & oos_dec).sum() / oos_dec.sum() * 100:.2f} %) ; proba NaN {int((np.isnan(p4) & oos_dec).sum())} ; "
          f"proba NaN <=> features NaN : {np.array_equal(np.isnan(p4[oos_dec]), fnan[oos_dec])}")
    print("NaN par feature (décisions OOS) : " + ", ".join(
        f"{c} {int(F4[c].isna().to_numpy()[oos_dec].sum())}" for c in agg4h.F4))
    nanpos = pd.Series(np.isnan(p4), index=df4.index)
    gov = (hidx.floor("4h") - 4 * H1)
    frozen = nanpos.reindex(gov).to_numpy(dtype=bool)
    expo = h0 > 0
    lr = np.log1p(res.bar_return.to_numpy())
    fe = frozen & expo
    share_x = fe.sum() / expo.sum()
    share_p = lr[fe].sum() / lr.sum()
    print(f"heures exposées {int(expo.sum())} ; dont figées (bougie de décision à proba NaN) {int(fe.sum())} "
          f"(part de l'exposition {share_x * 100:.2f} %) ; log-P&L figé {lr[fe].sum():+.4f} / total {lr.sum():+.4f} "
          f"(part du P&L {share_p * 100:.2f} %) ; Sharpe des seules heures non figées (heures figées mises à 0) "
          f"{ann(pd.Series(np.where(fe, 0.0, res.bar_return.to_numpy()), index=hidx).add(1).groupby(days).prod().sub(1)):.4f}")
    sig4b = pr["sig4"].to_numpy().copy()
    sig4b[cov & np.isnan(p4)] = 0.0
    s1b = agg4h.project_to_1h(sig4b, df4.index, df.index)
    rb = run_backtest(s1b, o, start=A, end=B, exec_delay=1, cost_per_side=COST)
    sbm = summarize(rb)
    print(f"variante NaN -> plat : Sharpe {sbm['sharpe']:+.4f} ({sbm['sharpe'] / S * 100:.0f} % de S) trades {sbm['trades']} "
          f"expo {sbm['exposure']:.4f} DD {sbm['max_drawdown']:.4f} rdt {sbm['total_return']:+.4f}")
    out["NA"] = {"share_expo": share_x, "share_pnl": share_p, "nan_flat": sbm["sharpe"]}

    # ------------------------------------------------------------ C1d / C1e
    print("\n=== C1d décomposition de l'AUC OOS des probas E36 (ma base) ===")
    y = pr["y_E36"].to_numpy()
    om = np.where(df4["missing"].to_numpy(), np.nan, df4["open"].to_numpy())
    nn = len(om)
    u = max(V, agg4h.C_LOG)
    hit = np.full(nn, np.nan)
    dirv = np.full(nn, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(nn - H - 1):
            if np.isnan(om[t + 1: t + 2 + H]).any():
                continue
            r = np.log(om[t + 2: t + 2 + H] / om[t + 1])
            up, dn = np.flatnonzero(r >= u), np.flatnonzero(r <= -V)
            ju = up[0] if len(up) else np.inf
            jd = dn[0] if len(dn) else np.inf
            hit[t] = float(ju < np.inf or jd < np.inf)
            if hit[t] == 1:
                dirv[t] = float(ju < jd)
    tmask = (df4.index >= A) & (df4.index <= B)
    pz = np.where(tmask, p4, np.nan)
    a_y, n_y = auc(y, pz)
    a_h, n_h = auc(hit, pz)
    a_d, n_d = auc(dirv, pz)
    print(f"OOS : AUC(label) {a_y:.4f} (n {n_y}) ; AUC(hit) {a_h:.4f} (n {n_h}, taux hit {np.nanmean(hit[tmask]):.4f}) ; "
          f"AUC(direction | hit) {a_d:.4f} (n {n_d}, taux haut d'abord {np.nanmean(dirv[tmask]):.4f})")
    for fn in ("vol_42", "vol_180"):
        for sg in (-1, 1):
            q = np.where(tmask, sg * F4[fn].to_numpy(), np.nan)
            print(f"  règle {'+' if sg > 0 else '-'}{fn} : AUC(label) {auc(y, q)[0]:.4f} AUC(hit) {auc(hit, q)[0]:.4f} "
                  f"AUC(dir | hit) {auc(dirv, q)[0]:.4f}")
    print(f"Spearman(proba, vol_180) OOS {pd.Series(pz).corr(pd.Series(np.where(tmask, F4['vol_180'].to_numpy(), np.nan)), method='spearman'):+.4f} ; "
          f"Spearman(proba, ret_180) {pd.Series(pz).corr(pd.Series(np.where(tmask, F4['ret_180'].to_numpy(), np.nan)), method='spearman'):+.4f}")
    out["auc"] = {"label": a_y, "hit": a_h, "dir": a_d}
    print("\n=== C1e seuil fixe 0,027683 ===")
    f1 = compute_features(df)["vol_168"]
    med = float(np.nanmedian((f1.loc[: pd.Timestamp("2020-12-31 23:00", tz="UTC")] * math.sqrt(24)).to_numpy()))
    med_ft = float(np.nanmedian((f1.loc[: pd.Timestamp("2020-12-26 19:00", tz="UTC")] * math.sqrt(24)).to_numpy()))
    print(f"médiane vol_168 x sqrt(24) sur 1 h < 2021-01-01 = {med:.7f} (déclaré 0,027683, |écart| {abs(med - 0.027683):.1e}) ; "
          f"jusqu'à train_end fold 1 (2020-12-26 19:00) = {med_ft:.7f} ; v = round(0,027683 x sqrt(5), 6) = "
          f"{agg4h.fixed_log_rule(30)} (config 0,061901)")

    # ------------------------------------------------------------ T5
    print("\n=== T5a par année (signal continu, position héritée) ===")
    neg = le_bh = 0
    for yy in range(2021, 2026):
        e = ts(f"{yy}-12-31", True) if yy < 2025 else B
        r = run_backtest(sig, o, start=ts(f"{yy}-01-01"), end=e, cost_per_side=COST)
        rbh = run_buy_and_hold(df, ts(f"{yy}-01-01"), e)
        s = summarize(r)
        sb = sharpe_daily(rbh.equity)
        neg += s["sharpe"] < 0
        le_bh += s["sharpe"] <= sb
        print(f"{yy} : E36 Sharpe {s['sharpe']:+.4f} rdt {s['total_return']:+.4f} trades {s['trades']} "
              f"expo {s['exposure']:.3f} DD {s['max_drawdown']:.4f} | B&H Sharpe {sb:+.4f} rdt {rbh.equity.iloc[-1] - 1:+.4f}")
    print(f"années Sharpe < 0 : {neg} ; années Sharpe <= B&H : {le_bh}")
    print("\n=== T5b par régime a priori (mois civil B&H : hausse > +5 %, baisse < -5 %, range sinon) ===")
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
        print(f"{g:<7}: {int((reg == g).sum())} mois, {int(mk.sum())} jours | E36 Sharpe {ann(x):+.4f} "
              f"rdt composé {np.prod(1 + x) - 1:+.4f} | B&H Sharpe {ann(xb):+.4f} rdt composé {np.prod(1 + xb) - 1:+.4f}")

    # ------------------------------------------------------------ T6b
    print("\n=== T6b sans le meilleur mois ===")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    best = mr.idxmax()
    top3 = mr.sort_values(ascending=False).head(3).index
    print(f"meilleur mois E36 : {best.strftime('%Y-%m')} rdt {mr.max():+.4f} ; top 3 : "
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
    Vv = srd.var(ddof=1)
    r_ = dr.to_numpy()
    T = len(r_)
    sr = r_.mean() / r_.std(ddof=1)
    g3, g4 = skew(r_, bias=False), kurtosis(r_, fisher=False, bias=False)
    gam = 0.5772156649015329

    def dsr_of(NN, VV):
        s0 = math.sqrt(VV) * ((1 - gam) * norm.ppf(1 - 1 / NN) + gam * norm.ppf(1 - 1 / (NN * math.e)))
        den = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
        return s0, norm.cdf((sr - s0) * math.sqrt(T - 1) / den), norm.cdf(sr * math.sqrt(T - 1) / den)
    sr0, dsr, psr = dsr_of(N, Vv)
    print(f"N = {N} lignes (ids {ids[0]}..{ids[-1]}) ; V = {Vv:.6e} ; SR0 journalier {sr0:.6f} (annualisé "
          f"{sr0 * math.sqrt(365):.4f})")
    print(f"SR journalier E36 {sr:.6f} (annualisé {sr * math.sqrt(365):.4f}) ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; "
          f"DSR = {dsr:.6f} ; PSR(0) = {psr:.6f}")
    V2 = srd[3:].var(ddof=1)
    print(f"(sensibilité) V sur E04-E39 = {V2:.6e} -> SR0 annualisé {dsr_of(N, V2)[0] * math.sqrt(365):.4f} DSR {dsr_of(N, V2)[1]:.6f} ; "
          f"V sur E29-E39 (bougies 4 h) = {srd[28:].var(ddof=1):.6e} -> DSR {dsr_of(N, srd[28:].var(ddof=1))[1]:.6f}")
    extra = sorted({p.stem for p in list(RES.glob("E*_*.json")) + list(OUT.glob("E*_adv_*.json"))
                    if not any(s in p.stem for s in ("shuf", "veto", "report", "_adv_base"))})
    print(f"(info) variantes de stratégie évaluées sur le même OOS hors registre (fichiers JSON, hors labels mélangés) : "
          f"{len(extra)} ; N_eff = {N + len(extra)} -> DSR {dsr_of(N + len(extra), Vv)[1]:.6f}")
    out["dsr"] = {"N": N, "dsr": dsr, "psr0": psr, "sr0": sr0}
    (OUT / "analyse_e36.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
