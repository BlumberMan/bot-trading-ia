r"""Avocat du diable P3-A6, candidat E49 (règle A / barrière vol). Critères : reports/adversarial/criteres_E49.txt.

Usage (OMP/MKL/OPENBLAS_NUM_THREADS=1) :
    .venv\Scripts\python tests\adversarial\adv_e49.py t1       T1-règles seeds 101..200 + contrôle de reconstruction
    .venv\Scripts\python tests\adversarial\adv_e49.py t2       T2a/T2b/T2c (seeds 4901, 4902)
    .venv\Scripts\python tests\adversarial\adv_e49.py battery  T3, T4, T5, T6, T7, BR, LT, PH, F1
    .venv\Scripts\python tests\adversarial\adv_e49.py t8       fuite statique et dynamique (seed 4903)
    .venv\Scripts\python tests\adversarial\adv_e49.py t9       antériorité ba68dde / 60e7a4a
Sorties : reports/adversarial/E49_adv_<test>.txt / .json. Données : load_dataset() par défaut (harness.load_dev),
garde index max <= 2025-09-30 23:00 UTC. Importe experiments/rules.py et rules_report.py sans les modifier.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
ADV = ROOT / "tests" / "adversarial"
OUT = ROOT / "reports" / "adversarial"
for p_ in (str(EXP), str(ADV), str(ROOT / "src")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import harness  # noqa: E402
import rules  # noqa: E402
import rules_report  # noqa: E402
from bot.baseline import run_buy_and_hold  # noqa: E402
from bot.metrics import daily_returns, sharpe_daily, summarize  # noqa: E402

A, B = rules.CONCAT_START, rules.CONCAT_END
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((EXP / "configs" / "E49.json").read_text(encoding="utf-8"))
S_DECL = 0.606844
NPROC = 9
FOLDS = [("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"), ("2022-01-01", "2022-06-30"),
         ("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
         ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"), ("2025-01-01", "2025-09-30")]


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(str(s))


def save(name, log, obj):
    (OUT / f"E49_adv_{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    (OUT / f"E49_adv_{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float) + "\n",
                                              encoding="utf-8")


def flag(b):
    return "FAILLE TROUVÉE" if b else "RÉSISTE"


def load(log=None):
    df = harness.load_dev(log or (lambda s="": None))
    assert df.index.max() <= MAX_INDEX, "données réservées"
    return df


def sh_concat(target, df, delay=1, cost_mult=1.0, a=A, b=B):
    return summarize(rules.bt(target, df, a, b, delay=delay, cost_mult=cost_mult))


def base_run(df):
    r = rules.build(df, CFG)
    m = sh_concat(r["target"], df)
    return r, m


def ann(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(365))


# ================================================================== T1

def my_synthetic(df: pd.DataFrame, seed, block=720) -> pd.DataFrame:
    """Réimplémentation indépendante : rendements log relatifs au dernier close disponible, blocs permutés."""
    cols = ["open", "high", "low", "close"]
    X = df[cols].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    miss = df["missing"].to_numpy(dtype=bool) | np.isnan(X).any(axis=1)
    n = len(df)
    lastc = np.empty(n)
    cur = np.nan
    for t in range(n):  # lastc[t] = dernier close disponible strictement avant t
        lastc[t] = cur
        if not miss[t]:
            cur = X[t, 3]
    R = np.full((n, 4), np.nan)
    good = ~miss
    good[0] = False
    R[good] = np.log(X[good] / lastc[good, None])
    nb = int(math.ceil((n - 1) / block))
    perm = np.arange(nb) if seed is None else np.random.default_rng(seed).permutation(nb)
    src = np.concatenate([np.arange(1 + q * block, min(1 + (q + 1) * block, n)) for q in perm])
    Y = np.full((n, 4), np.nan)
    V = np.full(n, np.nan)
    M = np.zeros(n, dtype=bool)
    Y[0] = X[0]
    V[0] = vol[0]
    P = X[0, 3]
    for t in range(1, n):
        s_ = src[t - 1]
        if miss[s_]:
            M[t] = True
            continue
        Y[t] = P * np.exp(R[s_])
        V[t] = vol[s_]
        P = Y[t, 3]
    out = pd.DataFrame(Y, index=df.index, columns=cols)
    out["volume"] = V
    out["missing"] = M
    return out


def _t1_worker(seed):
    df = load()
    syn = rules_report.make_synthetic(df, seed)
    mine = my_synthetic(df, seed)
    cols = ["open", "high", "low", "close"]
    a, b = syn[cols].to_numpy(), mine[cols].to_numpy()
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(syn["missing"].to_numpy(),
                                                                                  mine["missing"].to_numpy()))
    err = float(np.nanmax(np.abs(a / b - 1)))
    lr0 = np.log(df["close"] / df["close"].ffill().shift(1)).dropna().to_numpy()
    lr1 = np.log(syn["close"] / syn["close"].ffill().shift(1)).dropna().to_numpy()
    ms = bool(len(lr0) == len(lr1) and np.allclose(np.sort(lr0), np.sort(lr1), atol=1e-10, rtol=0))
    r = rules.build(syn, CFG)
    m = sh_concat(r["target"], syn)
    return {"seed": seed, "sharpe": float(m["sharpe"]), "trades": int(m["trades"]), "expo": float(m["exposure"]),
            "ret": float(m["total_return"]), "err_vs_mine": err, "same_nan": same_nan, "multiset": ms}


def t1():
    log = Log()
    df = load(log)
    r0, m0 = base_run(df)
    S = float(m0["sharpe"])
    log(f"=== T1-règles E49 : S reproduit {S:+.9f} (trades {m0['trades']}) ; blocs 720 ; seeds 101..200 ===")
    s_id_rr = rules.concat_sharpe(rules_report.make_synthetic(df, None), CFG)
    ident = my_synthetic(df, None)
    s_id_me = rules.concat_sharpe(ident, CFG)
    cols = ["open", "high", "low", "close"]
    err_id = float(np.nanmax(np.abs(ident[cols].to_numpy() / df[cols].to_numpy() - 1)))
    log(f"[r] identité make_synthetic : Sharpe {s_id_rr:+.9f} (écart {abs(s_id_rr - S):.2e}) ; identité mienne : "
        f"Sharpe {s_id_me:+.9f} (écart {abs(s_id_me - S):.2e}) ; écart relatif max série mienne vs origine {err_id:.2e}")
    with Pool(NPROC) as pool:
        res = pool.map(_t1_worker, list(range(101, 201)))
    for x in res:
        log(f"  seed {x['seed']:>3} : Sharpe {x['sharpe']:+.4f} trades {x['trades']} expo {x['expo']:.3f} rdt "
            f"{x['ret']:+.4f} | écart vs ma reconstruction {x['err_vs_mine']:.1e} NaN identiques {x['same_nan']} "
            f"multiset {x['multiset']}")
    v = np.array([x["sharpe"] for x in res])
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    p95 = float(np.percentile(v, 95))
    z = (S - mu) / sd
    nge = int((v >= S).sum())
    max_err = max(x["err_vs_mine"] for x in res)
    recon_ok = (abs(s_id_rr - S) <= 1e-9 and abs(s_id_me - S) <= 1e-9 and max_err <= 1e-10
                and all(x["same_nan"] and x["multiset"] for x in res))
    f_s = S <= p95 or z < 1.645
    log(f"[r] reconstruction : écart max make_synthetic vs mienne {max_err:.2e} ; NaN identiques "
        f"{all(x['same_nan'] for x in res)} ; multiset conservé {all(x['multiset'] for x in res)} -> "
        f"{flag(not recon_ok)}")
    log(f"[s] 100 séries (101..200) : moyenne {mu:+.4f} sd {sd:.4f} min {v.min():+.4f} max {v.max():+.4f} p95 "
        f"{p95:+.4f} ; NaN {int(np.isnan(v).sum())} ; séries >= S {nge}/100 ; z {z:+.3f} -> {flag(f_s)}")
    old = json.loads((EXP / "results" / "rules_t1" / "E49_t1.json").read_text(encoding="utf-8"))
    pool_v = np.concatenate([np.array(old["sharpes"], dtype=float), v])
    log(f"[info] pool 200 (1..100 chercheur + 101..200) : moyenne {pool_v.mean():+.4f} sd {pool_v.std(ddof=1):.4f} "
        f"p95 {np.percentile(pool_v, 95):+.4f} ; séries >= S {int((pool_v >= S).sum())}/200 ; z "
        f"{(S - pool_v.mean()) / pool_v.std(ddof=1):+.3f}")
    save("t1", log, {"S": S, "identity_rr": s_id_rr, "identity_mine": s_id_me, "seeds": list(range(101, 201)),
                     "rows": res, "mean": mu, "sd": sd, "p95": p95, "z": z, "n_ge": nge, "recon_faille": not recon_ok,
                     "faille": f_s, "pool200_p95": float(np.percentile(pool_v, 95)),
                     "pool200_nge": int((pool_v >= S).sum())})


# ================================================================== T2

_G = {}


def _t2_init():
    _G["df"] = load()
    _G["a_i"] = int(_G["df"].index.searchsorted(A))


def _t2_worker(h8):
    import analyse_e10 as ae
    r = ae.bt_from_h(h8.astype(float), _G["df"], _G["a_i"])
    return float(sharpe_daily(r.equity)), int(len(r.trades)), float((r.position > 0).mean())


def t2():
    import analyse_e10 as ae
    log = Log()
    df = load(log)
    r0, m0 = base_run(df)
    S = float(m0["sharpe"])
    res = rules.bt(r0["target"], df, A, B)
    h0 = res.position.to_numpy()
    s0, _ = ae.segments(h0)
    n, K, L = len(h0), len(s0), int((h0 > 0).sum())
    a_i = int(df.index.searchsorted(A))
    chk = ae.bt_from_h(h0, df, a_i)
    log(f"=== T2 E49 : S {S:+.6f} ; segments {K} ; barres exposées {L}/{n} (expo {L / n:.4f}) ; contrôle "
        f"bt_from_h(h0) Sharpe {sharpe_daily(chk.equity):+.6f} ===")
    rng = np.random.default_rng(4901)
    H = [ae.random_h(rng, n, K, L).astype(np.int8) for _ in range(1000)]
    with Pool(NPROC, initializer=_t2_init) as pool:
        ra = pool.map(_t2_worker, H, chunksize=20)
    v = np.array([x[0] for x in ra])
    pa = float(np.mean(v < S) * 100)
    fa = pa <= 95
    log(f"T2a grille 1 h (seed 4901, 1000) : trades {min(x[1] for x in ra)}..{max(x[1] for x in ra)} ; expo "
        f"{min(x[2] for x in ra):.4f}..{max(x[2] for x in ra):.4f} ; moyenne {v.mean():+.4f} sd {v.std(ddof=1):.4f} "
        f"p50 {np.percentile(v, 50):+.4f} p95 {np.percentile(v, 95):+.4f} max {v.max():+.4f} ; S percentile {pa:.1f} "
        f"-> {flag(fa)}")
    idx = res.position.index
    plan = []
    for a_s, b_s in FOLDS:
        msk = (idx >= pd.Timestamp(f"{a_s} 00:00", tz="UTC")) & (idx <= pd.Timestamp(f"{b_s} 23:00", tz="UTC"))
        hf = h0[msk]
        sf, _ = ae.segments(hf)
        plan.append((int(msk.sum()), max(len(sf), 1), int((hf > 0).sum())))
    assert sum(p[0] for p in plan) == n
    log(f"plan par fold (barres, segments, barres exposées) : {plan}")
    rng2 = np.random.default_rng(4902)
    H2 = []
    for _ in range(1000):
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
                parts.append(ae.random_h(rng2, nb, k, lf))
        H2.append(np.concatenate(parts).astype(np.int8))
    with Pool(NPROC, initializer=_t2_init) as pool:
        rb = pool.map(_t2_worker, H2, chunksize=20)
    v2 = np.array([x[0] for x in rb])
    pb = float(np.mean(v2 < S) * 100)
    fb = pb <= 95
    log(f"T2b stratifié par fold (seed 4902, 1000) : moyenne {v2.mean():+.4f} sd {v2.std(ddof=1):.4f} p50 "
        f"{np.percentile(v2, 50):+.4f} p95 {np.percentile(v2, 95):+.4f} ; S percentile {pb:.1f} -> {flag(fb)}")
    from bot.backtest import run_backtest
    rc = run_backtest(pd.Series(L / n, index=df.index), df["open"], start=A, end=B, initial_position=0.0)
    cf = float(sharpe_daily(rc.equity))
    fc = S <= cf
    log(f"T2c B&H fractionnaire constant {L / n:.4f} : Sharpe {cf:+.4f} rdt {rc.equity.iloc[-1] - 1:+.4f} ; S {S:+.4f} "
        f"-> {flag(fc)}")
    save("t2", log, {"S": S, "K": K, "L": L, "n": n, "t2a": {"seed": 4901, "pct": pa, "p95": float(np.percentile(v, 95)),
                     "mean": float(v.mean()), "faille": fa},
                     "t2b": {"seed": 4902, "pct": pb, "p95": float(np.percentile(v2, 95)), "mean": float(v2.mean()),
                             "faille": fb}, "t2c": {"sharpe": cf, "faille": fc}})


# ================================================================== batterie

def a_signal(df1, lags=(20, 60, 120), thr=2, lag_days=0):
    """Ma réimplémentation de la famille A (lags et seuil variables) ; lag_days = retard des features en jours."""
    n = len(df1)
    d = rules.daily_bars(df1)
    C = d["close"].to_numpy(dtype=float)
    nd = len(C)
    s30 = rules.sigma30(C)
    votes = np.zeros(nd)
    bad = np.isnan(C).copy()
    for L in lags:
        ref = np.full(nd, np.nan)
        ref[L:] = C[:-L]
        r = C / ref - 1.0
        bad |= np.isnan(r)
        votes += np.where(np.isnan(r), 0.0, (r > 0).astype(float))
    s = np.where(bad, np.nan, (votes >= thr).astype(float))
    if lag_days:
        s = np.concatenate([np.full(lag_days, np.nan), s[:-lag_days]])
        s30 = np.concatenate([np.full(lag_days, np.nan), s30[:-lag_days]])
    pos = rules.decision_positions(df1, d.index, 24)
    sig = np.full(n, np.nan)
    s30h = np.full(n, np.nan)
    sig[pos], s30h[pos] = s, s30
    return sig, s30h


def run_sig(df, sig, s30, b_scale=1.0, horizon=rules.H_TIME, delay=1):
    b = rules.barrier(s30, "vol", b_scale)
    tg, tr = rules.run_engine(df["open"].to_numpy(), df["close"].to_numpy(), sig, b, horizon, delay)
    return tg, tr


def battery():
    log = Log()
    df = load(log)
    r0, m0 = base_run(df)
    S = float(m0["sharpe"])
    tg0 = r0["target"]
    out = {"S": S}
    log(f"=== Batterie adversariale E49 : S reproduit {S:+.6f} ; trades {m0['trades']} ; DD {m0['max_drawdown']:.4f} ===")
    sig_me, s30_me = a_signal(df)
    same = np.array_equal(sig_me, r0["raw"], equal_nan=True) and np.array_equal(s30_me, r0["sigma30"], equal_nan=True)
    log(f"[contrôle] ma réimplémentation de la famille A == rules.family_signal : {same}")
    assert same

    def rec(name, x, fx, extra=""):
        log(f"  {name:<52}: Sharpe {x:+.4f} ({x / S * 100:5.0f} % de S){extra} -> {flag(fx)}")
        out[name] = {"sharpe": float(x), "faille": bool(fx)}

    log("\n--- T3 retards ---")
    tg2, tr2 = run_sig(df, sig_me, s30_me, delay=2)
    m = sh_concat(tg2, df, delay=2)
    x = m["sharpe"]
    rec("T3a t+2 moteur re-simulé", x, x <= 0 or x < 0.5 * S or x > 1.10 * S, f" trades {m['trades']} rdt {m['total_return']:+.4f}")
    m = sh_concat(tg0, df, delay=2)
    x = m["sharpe"]
    rec("T3b t+2 cibles figées (exec_delay=2)", x, x <= 0 or x < 0.5 * S or x > 1.10 * S, f" trades {m['trades']}")
    sl, s30l = a_signal(df, lag_days=1)
    tgl, _ = run_sig(df, sl, s30l)
    m = sh_concat(tgl, df)
    x = m["sharpe"]
    rec("T3c features retardées d'1 jour (moteur t+1)", x, x <= 0 or x < 0.5 * S or x > 1.10 * S,
        f" trades {m['trades']} rdt {m['total_return']:+.4f}")

    log("\n--- T4 coûts (cibles figées) ---")
    for k in (2.0, 3.0):
        m = sh_concat(tg0, df, cost_mult=k)
        x = m["sharpe"]
        rec(f"T4 coûts x{k:.0f}", x, x <= 0 or x < 0.5 * S, f" rdt {m['total_return']:+.4f} DD {m['max_drawdown']:.4f}")

    log("\n--- T5a années ---")
    yrs = {}
    for y in range(2021, 2026):
        a = pd.Timestamp(f"{y}-01-01 00:00", tz="UTC")
        e = pd.Timestamp(f"{y}-12-31 23:00", tz="UTC") if y < 2025 else B
        m = sh_concat(tg0, df, a=a, b=e)
        bh = sharpe_daily(run_buy_and_hold(df, a, e).equity)
        yrs[y] = (float(m["sharpe"]), float(bh))
        log(f"  {y} : Sharpe {m['sharpe']:+.4f} rdt {m['total_return']:+.4f} trades {m['trades']} expo "
            f"{m['exposure']:.3f} | B&H {bh:+.4f}")
    neg = sum(s_ < 0 for s_, _ in yrs.values())
    le = sum(s_ <= b_ for s_, b_ in yrs.values())
    f = neg >= 2 or le >= 4
    log(f"  années Sharpe < 0 : {neg} ; années Sharpe <= B&H : {le}/5 -> {flag(f)}")
    out["T5a"] = {"years": yrs, "neg": neg, "le_bh": le, "faille": f}

    log("\n--- T5b régimes ---")
    res = rules.bt(tg0, df, A, B)
    dr = daily_returns(res.equity)
    drb = daily_returns(run_buy_and_hold(df, A, B).equity)
    o = df["open"]
    mo = o.loc[A:B].resample("MS").first()
    nxt = mo.shift(-1)
    nxt.iloc[-1] = o.loc[B]
    mret = nxt / mo - 1
    reg = pd.Series(np.where(mret > 0.05, "hausse", np.where(mret < -0.05, "baisse", "range")), index=mo.index)
    dmonth = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1, tz="UTC") for t in dr.index])
    day_reg = reg.reindex(dmonth).to_numpy()
    rg = {}
    for g in ("hausse", "baisse", "range"):
        mk = day_reg == g
        rg[g] = {"months": int((reg == g).sum()), "days": int(mk.sum()), "sharpe": ann(dr[mk]),
                 "compound": float(np.prod(1 + dr[mk]) - 1), "bh_sharpe": ann(drb[mk]),
                 "bh_compound": float(np.prod(1 + drb[mk]) - 1)}
        log(f"  {g:<7}: {rg[g]['months']} mois {rg[g]['days']} j | Sharpe {rg[g]['sharpe']:+.4f} composé "
            f"{rg[g]['compound']:+.4f} | B&H Sharpe {rg[g]['bh_sharpe']:+.4f} composé {rg[g]['bh_compound']:+.4f}")
    f = rg["range"]["sharpe"] <= 0 or rg["hausse"]["sharpe"] <= 0 or rg["baisse"]["compound"] <= -0.5
    log(f"  -> {flag(f)}")
    out["T5b"] = {**rg, "faille": f}

    log("\n--- T6a sans un horizon (vote 2 sur 2 ; info 1 sur 2 et horizon seul) ---")
    t6 = {}
    for drop in (20, 60, 120):
        keep = tuple(L for L in (20, 60, 120) if L != drop)
        for thr in (2, 1):
            sg, s3 = a_signal(df, lags=keep, thr=thr)
            tg, _ = run_sig(df, sg, s3)
            m = sh_concat(tg, df)
            t6[(drop, thr)] = float(m["sharpe"])
            log(f"  sans L={drop:<3} {keep} vote >= {thr} : Sharpe {m['sharpe']:+.4f} trades {m['trades']} expo "
                f"{m['exposure']:.3f} rdt {m['total_return']:+.4f}")
    for L in (20, 60, 120):
        sg, s3 = a_signal(df, lags=(L,), thr=1)
        tg, _ = run_sig(df, sg, s3)
        m = sh_concat(tg, df)
        log(f"  [info] horizon seul L={L:<3} : Sharpe {m['sharpe']:+.4f} trades {m['trades']}")
    dom = min((20, 60, 120), key=lambda L: t6[(L, 2)])
    x = t6[(dom, 2)]
    rec(f"T6a sans l'horizon dominant (L={dom}, vote 2/2)", x, x <= 0 or x < 0.5 * S)

    log("\n--- T6b sans le meilleur mois ---")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    top = mr.sort_values(ascending=False)
    bm = top.index[0]
    x = ann(dr[dmonth != bm])
    x3 = ann(dr[~dmonth.isin(top.index[:3])])
    log(f"  meilleurs mois : " + ", ".join(f"{i.strftime('%Y-%m')} {v_:+.4f}" for i, v_ in top.head(3).items()))
    rec(f"T6b sans le meilleur mois ({bm.strftime('%Y-%m')})", x, x <= 0 or x < 0.5 * S, f" ; [info] sans 3 meilleurs {x3:+.4f}")

    log("\n--- T7 DSR (N = lignes de REGISTRE.md) ---")
    lines = (EXP / "REGISTRE.md").read_text(encoding="utf-8").splitlines()
    ids = [ln.split("|")[1].strip() for ln in lines]
    N = len(lines)
    sr_i, sr_ii = [], []
    for i in ids:
        j = json.loads((EXP / "results" / f"{i}.json").read_text(encoding="utf-8"))
        s_ann = float(j["concatenated"]["model"]["sharpe"])
        sr_ii.append(s_ann / math.sqrt(365))
        sr_i.append(float(j["daily_oos"]["sr_daily"]) if "daily_oos" in j else s_ann / math.sqrt(365))
    rr = dr.to_numpy()
    T = len(rr)
    sr = rr.mean() / rr.std(ddof=1)
    g3 = float(skew(rr, bias=False))
    g4 = float(kurtosis(rr, fisher=False, bias=False))
    gam = 0.5772156649015329
    t7 = {}
    for lab, arr in (("(i) sr_daily", sr_i), ("(ii) Sharpe/sqrt(365)", sr_ii)):
        Vv = float(np.var(arr, ddof=1))
        e_max = (1 - gam) * norm.ppf(1 - 1 / N) + gam * norm.ppf(1 - 1 / (N * math.e))
        sr0 = math.sqrt(Vv) * e_max
        den = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
        dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))
        psr = float(norm.cdf(sr * math.sqrt(T - 1) / den))
        t7[lab] = {"N": N, "V": Vv, "sr0": sr0, "sr0_ann": sr0 * math.sqrt(365), "dsr": dsr, "psr0": psr}
        log(f"  {lab} : N {N} ; V {Vv:.6e} ; E[max] {e_max:.6f} ; SR0 journalier {sr0:.6f} (ann. {sr0 * math.sqrt(365):.4f}) ;"
            f" SR {sr:.6f} (ann. {sr * math.sqrt(365):.4f}) ; T {T} ; skew {g3:.4f} ; kurt {g4:.4f} ; DSR {dsr:.6f} ; "
            f"PSR(0) {psr:.6f}")
    f = min(v_["dsr"] for v_ in t7.values()) < 0.95
    log(f"  -> {flag(f)}")
    out["T7"] = {**t7, "faille": f, "sr": sr, "T": T, "skew": g3, "kurt": g4}

    log("\n--- BR barrières ±20 % (re-simulées) ---")
    for nm, kw in (("BR b x0,8", {"b_scale": 0.8}), ("BR b x1,2", {"b_scale": 1.2}),
                   ("BR temps 96 h", {"horizon": 96}), ("BR temps 144 h", {"horizon": 144})):
        tg, _ = run_sig(df, sig_me, s30_me, **kw)
        m = sh_concat(tg, df)
        x = m["sharpe"]
        rec(nm, x, x <= 0 or x < 0.7 * S, f" trades {m['trades']}")

    log("\n--- LT faible nombre de trades ---")
    trs = res.trades
    tr_r = trs["net_return"].to_numpy()
    rng = np.random.default_rng(4904)
    bs = rng.choice(tr_r, size=(10000, len(tr_r)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    f = lo <= 0
    log(f"  (a) {len(tr_r)} trades ; moyenne {tr_r.mean():+.5f} médiane {np.median(tr_r):+.5f} ; IC95 moyenne "
        f"[{lo:+.5f} ; {hi:+.5f}] ; P(moy <= 0) {np.mean(bs <= 0):.4f} (seed 4904, 10000) -> {flag(f)}")
    out["LTa"] = {"ci": [lo, hi], "faille": bool(f)}
    rng = np.random.default_rng(4905)
    nT, blk = len(rr), 20
    nblk = int(math.ceil(nT / blk))
    shs = np.empty(10000)
    for q in range(10000):
        st = rng.integers(0, nT, nblk)
        ii = (st[:, None] + np.arange(blk)[None, :]).ravel()[:nT] % nT
        x_ = rr[ii]
        shs[q] = x_.mean() / x_.std(ddof=1) * math.sqrt(365)
    lo, hi = np.percentile(shs, [2.5, 97.5])
    f = lo <= 0
    log(f"  (b) bootstrap circulaire blocs 20 j du Sharpe : IC95 [{lo:+.4f} ; {hi:+.4f}] ; P(Sharpe <= 0) "
        f"{np.mean(shs <= 0):.4f} (seed 4905, 10000) -> {flag(f)}")
    out["LTb"] = {"ci": [lo, hi], "faille": bool(f)}
    trades = r0["trades"]
    a_i = int(df.index.searchsorted(A))
    seg_pos = [int(df.index.get_loc(t)) for t in trs["entry_time"]]
    order = np.argsort(-tr_r)

    def eng_trade(gpos):
        for t in trades:
            if t["entry"] is not None and t["entry"] <= gpos and (t["exit"] is None or gpos < t["exit"]):
                return t
        raise AssertionError(gpos)

    for k in (1, 3, 5):
        tg = tg0.copy()
        rm = []
        for q in order[:k]:
            t = eng_trade(seg_pos[q])
            i1 = t["exit_dec"] if t["exit_dec"] is not None else len(tg)
            tg[t["dec"]:i1] = 0.0
            rm.append(f"{df.index[t['entry']]} {tr_r[q]:+.4f}")
        m = sh_concat(tg, df)
        x = m["sharpe"]
        fx = (x < 0.5 * S) if k == 1 else (x <= 0 or x < 0.5 * S)
        rec(f"LTc sans les {k} meilleurs trades", x, fx, f" trades {m['trades']} ; retirés {', '.join(rm)}")

    log("\n--- PH phase horaire de la décision ---")
    ph = {}
    for k in (0, 4, 8, 12, 16, 20):
        d2 = df.copy()
        d2.index = df.index + pd.Timedelta(hours=k)
        r2 = rules.build(d2, CFG)
        m = sh_concat(r2["target"], df)
        ph[k] = float(m["sharpe"])
        log(f"  décision {(23 - k) % 24:02d}:00 UTC (décalage {k:>2} h) : Sharpe {m['sharpe']:+.4f} trades {m['trades']} "
            f"expo {m['exposure']:.3f}")
    assert abs(ph[0] - S) < 1e-12
    vals = [ph[k] for k in (4, 8, 12, 16, 20)]
    f = min(vals) <= 0 or float(np.median(vals)) < 0.5 * S
    log(f"  5 phases : min {min(vals):+.4f} médiane {np.median(vals):+.4f} (0,5 S = {0.5 * S:+.4f}) -> {flag(f)}")
    out["PH"] = {"values": ph, "faille": bool(f)}

    log("\n--- F1 sans le fold 1 ---")
    a2 = pd.Timestamp("2021-07-01 00:00", tz="UTC")
    m = sh_concat(tg0, df, a=a2)
    bh = sharpe_daily(run_buy_and_hold(df, a2, B).equity)
    f = m["sharpe"] <= bh
    log(f"  2021-07-01 -> 2025-09-30 : Sharpe {m['sharpe']:+.4f} trades {m['trades']} rdt {m['total_return']:+.4f} | "
        f"B&H {bh:+.4f} -> {flag(f)}")
    out["F1"] = {"sharpe": float(m["sharpe"]), "bh": float(bh), "faille": bool(f)}
    save("battery", log, out)


# ================================================================== T8

def perturb_after(df, t0, seed):
    rng = np.random.default_rng(seed)
    d = df.copy()
    n = len(d)
    m = n - t0 - 1
    c0 = float(d["close"].iloc[:t0 + 1].dropna().iloc[-1])
    c = c0 * np.exp(np.cumsum(rng.normal(0, 0.01, m)))
    o = np.concatenate([[c0], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, m)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, m)))
    miss = d["missing"].to_numpy()[t0 + 1:] | d[["open", "close"]].iloc[t0 + 1:].isna().any(axis=1).to_numpy()
    for col, v in (("open", o), ("high", hi), ("low", lo), ("close", c)):
        d.iloc[t0 + 1:, d.columns.get_loc(col)] = np.where(miss, np.nan, v)
    return d


def t8():
    log = Log()
    df = load(log)
    log("=== T8 fuite E49 ===")
    log("--- (a) statique : grep dans src/, experiments/rules.py, rules_report.py, harness.py ---")
    pats = ["allow_holdout", "holdout_period", "HOLDOUT_END", "read_parquet", "read_csv", "2025-10", "2026-"]
    files = [EXP / "rules.py", EXP / "rules_report.py", EXP / "harness.py"] + sorted((ROOT / "src" / "bot").glob("*.py"))
    for fpath in files:
        for no, ln in enumerate(fpath.read_text(encoding="utf-8").splitlines(), 1):
            if any(p in ln for p in pats):
                log(f"  {fpath.relative_to(ROOT)}:{no}: {ln.strip()}")
    r0, m0 = base_run(df)
    res = rules.bt(r0["target"], df, A, B)
    log(f"--- (c) dernier open lu {res.end_mark_time} <= {MAX_INDEX} : {res.end_mark_time <= MAX_INDEX} ; dernière "
        f"décision : cible {r0['target'][-1]:.0f}, trade final {r0['trades'][-1]['reason']} ; positions tenues à la "
        f"dernière bougie {res.position.iloc[-1]:.0f} ===")
    t2_ = r0["target"].copy()
    t2_[-1] = 1 - t2_[-1]
    inv = np.array_equal(rules.bt(t2_, df, A, B).equity.to_numpy(), res.equity.to_numpy())
    log(f"  backtest invariant à la décision de la dernière clôture : {inv}")
    log("--- (b) dynamique : 30 coupures (seed 4903), heures > t0 remplacées par une marche aléatoire ---")
    rng = np.random.default_rng(4903)
    lo_i, hi_i = int(df.index.searchsorted(A)), int(df.index.searchsorted(pd.Timestamp("2025-09-29 00:00", tz="UTC")))
    t0s = np.sort(rng.choice(np.arange(lo_i, hi_i), 30, replace=False))
    all_ok, any_ch = True, False
    rows = []
    for i, t0 in enumerate(t0s):
        d = perturb_after(df, int(t0), 4903 + i)
        r1 = rules.build(d, CFG)
        ok_sig = np.array_equal(r0["raw"][:t0 + 1], r1["raw"][:t0 + 1], equal_nan=True)
        ok_b = np.array_equal(r0["b"][:t0 + 1], r1["b"][:t0 + 1], equal_nan=True)
        ok_tg = np.array_equal(r0["target"][:t0 + 1], r1["target"][:t0 + 1])
        e0 = [(t["dec"], t["entry"], t["exit_dec"], t["reason"]) for t in r0["trades"]
              if t["exit_dec"] is not None and t["exit_dec"] <= t0]
        e1 = [(t["dec"], t["entry"], t["exit_dec"], t["reason"]) for t in r1["trades"]
              if t["exit_dec"] is not None and t["exit_dec"] <= t0]
        ch = not np.array_equal(r0["raw"][t0 + 1:], r1["raw"][t0 + 1:], equal_nan=True)
        ok = ok_sig and ok_b and ok_tg and e0 == e1
        all_ok &= ok
        any_ch |= ch
        rows.append({"t0": str(df.index[t0]), "ok": bool(ok), "changed_after": bool(ch), "exits_le_t0": len(e0)})
        log(f"  {i:>2} t0 {df.index[t0]} : signal {ok_sig} b {ok_b} cibles {ok_tg} sorties<=t0 identiques "
            f"{e0 == e1} ({len(e0)}) ; signal > t0 modifié {ch}")
    f = not (all_ok and any_ch and inv and res.end_mark_time <= MAX_INDEX)
    log(f"  30/30 inchangés <= t0 : {all_ok} ; test non vide : {any_ch} -> {flag(f)}")
    save("t8", log, {"rows": rows, "all_ok": bool(all_ok), "any_changed": bool(any_ch), "inv_last": bool(inv),
                     "faille": bool(f)})


# ================================================================== T9

def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def t9():
    log = Log()
    log("=== T9 antériorité E49 ===")
    head = git("rev-parse", "HEAD")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", "ba68dde", head], cwd=ROOT).returncode == 0
    log(f"HEAD {head} ; ba68dde ancêtre de HEAD : {anc}")
    log(f"ba68dde : {git('log', '-1', '--format=%H %ci %s', 'ba68dde')}")
    log(f"60e7a4a : {git('log', '-1', '--format=%H %ci %s', '60e7a4a')}")
    log(f"f021b4b : {git('log', '-1', '--format=%H %ci %s', 'f021b4b')}")
    tree = git("ls-tree", "-r", "--name-only", "ba68dde").splitlines()
    pre = [p for p in tree if p.startswith("experiments/results/") and any(f"E{i}" in p for i in range(48, 56))]
    hist = git("log", "--format=%h", "ba68dde", "--", *[f"experiments/results/E{i}.json" for i in range(48, 56)],
               "experiments/results/rules_t1", "experiments/results/rules_E49")
    log(f"résultats E48..E55 dans l'arbre de ba68dde : {pre if pre else 'aucun'} ; dans l'historique jusqu'à ba68dde : "
        f"{hist if hist else 'aucun'}")
    j = json.loads((EXP / "results" / "E49.json").read_text(encoding="utf-8"))
    log(f"E49.json : commit {j['commit']} ; code_dirty {j['code_dirty']}")
    d_other = git("diff", "--name-only", "ba68dde", head, "--", "src", "experiments/harness.py",
                  "experiments/rules_report.py", *[f"experiments/configs/E{i}.json" for i in range(48, 56)])
    log(f"diff ba68dde..HEAD sur src, harness, rules_report, configs E48..E55 : {d_other if d_other else 'vide'}")
    d_rules = git("diff", "ba68dde", head, "--", "experiments/rules.py")
    log("diff ba68dde..HEAD sur experiments/rules.py :")
    for ln in d_rules.splitlines():
        log("    " + ln)
    hunks = [ln for ln in d_rules.splitlines() if ln.startswith("@@")]
    # lignes modifiées (numérotation des deux versions, -U0) toutes dans le corps de registry_line (ast)
    import ast
    import re

    def span(src):
        for nd in ast.parse(src).body:
            if isinstance(nd, ast.FunctionDef) and nd.name == "registry_line":
                return nd.lineno, nd.end_lineno
        raise AssertionError("registry_line absente")

    old_full = git("show", "ba68dde:experiments/rules.py")
    so, sn = span(old_full), span((EXP / "rules.py").read_text(encoding="utf-8"))
    only_reg = True
    for h_ in git("diff", "-U0", "ba68dde", head, "--", "experiments/rules.py").splitlines():
        mm = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", h_)
        if mm:
            o1, oc, n1, nc = int(mm[1]), int(mm[2] or 1), int(mm[3]), int(mm[4] or 1)
            if oc and not (so[0] <= o1 and o1 + oc - 1 <= so[1]):
                only_reg = False
            if nc and not (sn[0] <= n1 and n1 + nc - 1 <= sn[1]):
                only_reg = False
            log(f"hunk -U0 {h_.strip()} ; registry_line ba68dde lignes {so} ; HEAD lignes {sn}")
    calls = [ln for ln in (EXP / "rules.py").read_text(encoding="utf-8").splitlines() if "registry_line(" in ln]
    other_calls = git("grep", "-n", "registry_line(", "--", "experiments", "src", "scripts")
    log(f"hunks : {len(hunks)} ; seul registry_line touché : {only_reg} ; occurrences de registry_line( : {calls} ; "
        f"git grep : {other_calls}")
    # rules.py de ba68dde exécuté hors arbre
    scratch = Path(os.environ["ADV_SCRATCH"]).resolve()  # dossier hors du dépôt
    scratch.mkdir(parents=True, exist_ok=True)
    old_src = git("show", "ba68dde:experiments/rules.py")
    pold = scratch / "rules_ba68dde.py"
    pold.write_text(old_src + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("rules_ba68dde", pold)
    rold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rold)
    df = load(log)
    r_new = rules.build(df, CFG)
    r_old = rold.build(df, CFG)
    s_new = summarize(rules.bt(r_new["target"], df, A, B))
    s_old = summarize(rold.bt(r_old["target"], df, A, B))
    same_t = np.array_equal(r_new["target"], r_old["target"])
    same_tr = r_new["trades"] == r_old["trades"]
    log(f"rules.py ba68dde (hors arbre, {pold}) vs HEAD : cibles identiques {same_t} ; trades identiques {same_tr} "
        f"({len(r_old['trades'])}) ; Sharpe {s_old['sharpe']:+.12f} vs {s_new['sharpe']:+.12f} ; trades "
        f"{s_old['trades']} vs {s_new['trades']}")
    line49 = (EXP / "REGISTRE.md").read_text(encoding="utf-8").splitlines()[48]
    regen = rules.registry_line(j, line49.split("|")[2].strip())
    log(f"ligne 49 de REGISTRE.md == registry_line(E49.json) : {regen == line49}")
    # contrôle du garde-fou de register() : diff entre le commit du résultat et HEAD
    ch = git("diff", "--name-only", j["commit"], head, "--", *rules.CODE_PATHS)
    log(f"[info] register() exécuté à HEAD refuserait E49 (diff {j['commit'][:7]}..HEAD sur CODE_PATHS) : "
        f"{ch if ch else 'vide'}")
    f = not (anc and not pre and not hist and j["commit"].startswith("ba68dde") and j["code_dirty"] is False
             and not d_other and only_reg and len(calls) == 2 and same_t and same_tr
             and abs(s_old["sharpe"] - s_new["sharpe"]) <= 1e-12 and s_old["trades"] == s_new["trades"]
             and regen == line49)
    log(f"-> {flag(f)}")
    save("t9", log, {"faille": bool(f), "same_target": bool(same_t), "same_trades": bool(same_tr),
                     "registry_equal": regen == line49})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"t1": t1, "t2": t2, "battery": battery, "t8": t8, "t9": t9}[sys.argv[1]]()
