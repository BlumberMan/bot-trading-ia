r"""Tests du veto et batterie du cycle « règles » (brief P3-B9, points 5 à 7). N'est pas un essai.

Usage :
    .venv\Scripts\python experiments\rules_report.py --selftest          auto-tests (synthétique)
    .venv\Scripts\python experiments\rules_report.py --t1 E48            T1-règles, 100 séries mélangées par blocs
    .venv\Scripts\python experiments\rules_report.py --t2 E48            analyse_e10 importé (T2, T2b, figés, T5, T6b, DSR)
    .venv\Scripts\python experiments\rules_report.py --veto E48          synthèse T1 / T2 / T7 -> results/E48_veto.*
    .venv\Scripts\python experiments\rules_report.py --summary           tableau des 8 essais, éligibilité, candidat
    .venv\Scripts\python experiments\rules_report.py --full E48          batterie complète du candidat
--t2, --veto, --summary et --full exigent que les 8 lignes E48-E55 soient au registre (N = 55 pour le DSR).

ÉLIGIBILITÉ (point 5) : Sharpe net OOS concaténé S > -0,530.

T1-RÈGLES (remplace les labels mélangés, validé par Iyad) : période dev 1 h (load_dataset par défaut). Bougie t >= 1
décomposée en rc = log(close_t / P), ro = log(open_t / P), rh = log(high_t / P), rl = log(low_t / P), P = dernier
close disponible avant t (bougie manquante : tout NaN, elle voyage avec son bloc). Bougies 1..n-1 découpées en blocs
consécutifs de 720 (le dernier est plus court), blocs permutés (numpy default_rng(seed).permutation, seeds 1..100),
série reconstruite depuis le premier close (bougie 0 inchangée ; close manquant -> P inchangé). Même grille de dates,
mêmes folds, règle appliquée telle quelle (rules.build). Statistique : Sharpe concaténé 2021-01 -> 2025-09.
FAILLE si S <= p95 (numpy.percentile, interpolation linéaire) des 100 Sharpe. Fournis aussi : moyenne, écart-type,
z = (S - moyenne) / écart-type (ddof=1), nombre de séries >= S.

T2 : tests/adversarial/analyse_e10.py IMPORTÉ et exécuté tel quel (non modifié) ; seules ses constantes de module
OUT (dossier temporaire, signal réécrit au format CSV attendu) et CAND (= S, contrôle 1e-12) sont remplacées.
Aléatoire mêmes trades (segments) et même exposition : 1000 tirages grille 1 h (seed 20261004) et 1000 stratifiés par
fold (seed 20261005). FAILLE si l'un des deux percentiles <= 95.

T7 : DSR (Bailey & López de Prado 2014). N = nombre de lignes de experiments/REGISTRE.md ; V = variance (ddof=1) des
Sharpe journaliers (daily_oos.sr_daily des results/<ID>.json) de toutes les lignes ; SR, T, skew, kurtosis (non
excès) des rendements journaliers OOS de l'essai. SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))),
g = constante d'Euler ; DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2)).
FAILLE si DSR < 0,95.

CHOIX DU CANDIDAT (règle JOURNAL.md inchangée) : parmi E48-E55, Sharpe net OOS concaténé le plus élevé, à condition
qu'il dépasse celui de la baseline sur les mêmes folds (-0,5297) ; sinon « aucun candidat ».

BATTERIE DU CANDIDAT (point 6) ; drapeaux indicatifs (seuils de reports/adversarial/criteres_E24.txt), S = Sharpe
du candidat :
- coûts x2, x3 (même signal ; la règle ne dépend pas des coûts) : FAILLE si Sharpe <= 0.
- exécution à t+2 : (a) moteur re-simulé avec un délai de 2 h (E = open t+2, sorties à k+2 ; backtest exec_delay=2) ;
  (b) signal figé, backtest exec_delay=2. FAILLE si X <= 0, X < 0,5 S ou X > 1,10 S.
- barrières ±20 % : b x 0,8 et x 1,2 (fixe : log(1,027) x s ; vol : s x sigma30 x sqrt(5), plancher inchangé) ;
  en plus, barrière temporelle 96 et 144 h. FAILLE si X <= 0 ou X < 0,5 S.
- années 2021..2025 (signal continu, position héritée ; B&H par année) : FAILLE si Sharpe < 0 sur >= 2 années OU
  Sharpe <= B&H sur >= 4 années sur 5.
- régimes a priori (rendement B&H du mois civil, open 1re bougie -> open 1re bougie du mois suivant ; > +5 % hausse,
  < -5 % baisse, sinon range) : FAILLE si Sharpe range <= 0 OU Sharpe hausse <= 0 OU rendement composé baisse <= -50 %.
- sans le meilleur mois (jours du meilleur mois retirés) : FAILLE si X <= 0 ou X < 0,5 S.
- sans les 1, 3, 5 meilleurs trades OOS (cibles mises à 0 sur ces trades, re-backtest) : FAILLE si X <= 0 ou X < 0,5 S.
- IC bootstrap par trade : rendements nets des trades OOS, 10000 rééchantillonnages (seed 9), IC 95 % percentile de
  la moyenne et du rendement composé ; FAILLE si la borne basse de l'IC de la moyenne <= 0.
- B&H à exposition égale (analyse_e10, T2b) : FAILLE si S <= Sharpe du B&H fractionnaire constant OU percentile
  stratifié <= 95.
- nombre de trades OOS (informatif ; GONOGO exigera >= 200 au palier 4).
Sorties : results/rules_t1/<ID>_t1.json/.txt, results/rules_<ID>/analyse_e10.json + analyse.txt,
results/<ID>_veto.json/.txt, results/rules_summary.json/.txt, results/<ID>_battery.json/.txt.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
RES = EXP / "results"
sys.path.insert(0, str(EXP))

import harness  # noqa: E402
import rules  # noqa: E402

IDS = [f"E{i}" for i in range(rules.REGISTRY_FIRST, rules.REGISTRY_LAST + 1)]
ELIG = -0.530
BLOCK = 720
T1_SEEDS = range(1, 101)
EULER = 0.5772156649015329
BOOT_SEED, BOOT_B = 9, 10000


class Cap(io.StringIO):
    def reconfigure(self, **kw):
        pass


def fl(x):
    return float(x) if not isinstance(x, str) else float("nan")


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def flag(b):
    return "FAILLE" if b else "RÉSISTE"


def cfg_of(tid):
    return json.loads((EXP / "configs" / f"{tid}.json").read_text(encoding="utf-8"))


def need_full_registry():
    lines = rules.REGISTRE.read_text(encoding="utf-8").splitlines()
    ids = [ln.split("|")[1].strip() for ln in lines]
    if len(lines) != rules.REGISTRY_LAST or ids[-8:] != IDS:
        raise RuntimeError(f"registre : {len(lines)} lignes ; les 8 lignes E48-E55 doivent y être (N = 55)")
    return lines, ids


# ------------------------------------------------------------------ T1-règles : séries mélangées par blocs

def make_synthetic(df: pd.DataFrame, seed: int | None, block: int = BLOCK) -> pd.DataFrame:
    """seed None : permutation identité (contrôle : reconstruit la série d'origine)."""
    o, h, l, c = (df[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=np.float64)
    miss = np.isnan(np.c_[o, h, l, c]).any(axis=1) | df["missing"].to_numpy(dtype=bool)
    n = len(df)
    if miss[0]:
        raise ValueError("première bougie manquante")
    P = pd.Series(np.where(miss, np.nan, c)).ffill().shift(1).to_numpy()
    rel = np.full((n, 4), np.nan)
    ok = ~miss
    ok[0] = False
    for k, x in enumerate((o, h, l, c)):
        rel[ok, k] = np.log(x[ok] / P[ok])
    pos = np.arange(1, n)
    blocks = [pos[s:s + block] for s in range(0, n - 1, block)]
    perm = np.arange(len(blocks)) if seed is None else np.random.default_rng(seed).permutation(len(blocks))
    order = np.concatenate([blocks[q] for q in perm])
    r2, m2, v2 = rel[order], miss[order], v[order]
    rc = np.where(m2, 0.0, r2[:, 3])
    logc = math.log(c[0]) + np.cumsum(rc)
    logP = np.concatenate([[math.log(c[0])], logc[:-1]])
    out = np.full((n, 5), np.nan)
    out[0] = [o[0], h[0], l[0], c[0], v[0]]
    for k in range(4):
        out[1:, k] = np.where(m2, np.nan, np.exp(logP + r2[:, k]))
    out[1:, 4] = np.where(m2, np.nan, v2)
    syn = pd.DataFrame(out, index=df.index, columns=["open", "high", "low", "close", "volume"])
    syn["missing"] = np.concatenate([[False], m2])
    return syn


def t1(tid: str) -> int:
    log = harness.Tee()
    j = load(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    cfg = cfg_of(tid)
    log(f"=== T1-règles {tid} ({cfg['family']} / {cfg['barrier']}) : S = {S:+.6f} ; éligible (S > {ELIG}) : {S > ELIG} ===")
    if not S > ELIG:
        log("non éligible : T1 non calculé")
        return 0
    df = harness.load_dev(log)
    s_id = rules.concat_sharpe(make_synthetic(df, None), cfg)
    log(f"[contrôle] permutation identité : Sharpe {s_id:+.9f} vs S {S:+.9f} (écart {abs(s_id - S):.2e})")
    vals, trades = [], []
    for sd in T1_SEEDS:
        syn = make_synthetic(df, sd)
        r = rules.build(syn, cfg)
        m = rules.summarize(rules.bt(r["target"], syn, rules.CONCAT_START, rules.CONCAT_END))
        vals.append(fl(m["sharpe"]))
        trades.append(int(m["trades"]))
        log(f"  seed {sd:>3} : Sharpe concat {vals[-1]:+.4f} trades {trades[-1]} expo {m['exposure']:.3f} "
            f"rdt {m['total_return']:+.4f}")
    v = np.array(vals)
    mu, sd_ = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
    p95 = float(np.nanpercentile(v, 95))
    z = (S - mu) / sd_
    nge = int(np.sum(v >= S))
    f1 = S <= p95
    log(f"  100 séries : moyenne {mu:+.4f} écart-type {sd_:.4f} min {np.nanmin(v):+.4f} max {np.nanmax(v):+.4f} "
        f"p95 {p95:+.4f} ; NaN {int(np.isnan(v).sum())} ; trades moyens {np.mean(trades):.1f}")
    log(f"  {tid} S {S:+.4f} : z {z:+.3f} ; séries >= S : {nge}/100 ; S > p95 : {S > p95} -> {flag(f1)}")
    out = {"id": tid, "S": S, "identity_sharpe": s_id, "seeds": list(T1_SEEDS), "sharpes": vals, "trades": trades,
           "mean": mu, "std": sd_, "p95": p95, "z": z, "n_ge": nge, "faille": f1, "block": BLOCK}
    d = RES / "rules_t1"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tid}_t1.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    (d / f"{tid}_t1.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ T2 : analyse_e10 importé

def t2(tid: str) -> int:
    need_full_registry()
    j = load(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    if not S > ELIG:
        print(f"{tid} non éligible : T2 non calculé")
        return 0
    sys.path.insert(0, str(ROOT / "tests" / "adversarial"))
    import analyse_e10  # noqa: E402
    adv = RES / f"rules_{tid}"
    sig = pd.read_parquet(adv / "signal_1h.parquet", engine="pyarrow")[["signal"]]
    tmp = Path(tempfile.mkdtemp(prefix=f"rules_{tid}_"))
    try:
        sig.to_csv(tmp / "E10_adv_base_signal_full.csv")  # nom attendu par analyse_e10.py
        analyse_e10.OUT = tmp
        analyse_e10.CAND = S
        argv = sys.argv
        sys.argv = ["analyse_e10.py", "--draws", "1000", "--seed", "20261004"]
        buf = Cap()
        with redirect_stdout(buf):
            analyse_e10.main()
        sys.argv = argv
        shutil.copyfile(tmp / "analyse_e10.json", adv / "analyse_e10.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (adv / "analyse.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    return 0


# ------------------------------------------------------------------ T7

def dsr_block(j: dict, tid: str, p) -> dict:
    lines, ids = need_full_registry()
    N = len(lines)
    srs = np.array([load(i)["daily_oos"]["sr_daily"] for i in ids])
    V = float(np.var(srs, ddof=1))
    d = j["daily_oos"]
    sr, T, g3, g4 = d["sr_daily"], d["n_days"], d["skew"], d["kurtosis"]
    z1, z2 = norm.ppf(1 - 1 / N), norm.ppf(1 - 1 / (N * math.e))
    sr0 = math.sqrt(V) * ((1 - EULER) * z1 + EULER * z2)
    den = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    stat = (sr - sr0) * math.sqrt(T - 1) / den
    dsr = float(norm.cdf(stat))
    psr0 = float(norm.cdf(sr * math.sqrt(T - 1) / den))
    p(f"N = {N} lignes de REGISTRE.md ({ids[0]}..{ids[-1]}) ; SR journaliers : {np.round(srs, 5).tolist()}")
    p(f"V[SR_n] = {V:.6e} (écart-type {math.sqrt(V):.6f}) ; Phi^-1(1-1/N) = {z1:.6f} ; Phi^-1(1-1/(N e)) = {z2:.6f}")
    p(f"SR0 journalier = {sr0:.6f} (annualisé {sr0 * math.sqrt(365):.4f})")
    p(f"SR {tid} journalier {sr:.6f} (annualisé {sr * math.sqrt(365):.4f}) ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; "
      f"dénominateur {den:.6f} ; SR - SR0 {sr - sr0:+.6f} ; stat {stat:.4f}")
    p(f"DSR = {dsr:.6f} ; PSR(0) = {psr0:.6f} -> {flag(dsr < 0.95)}")
    return {"N": N, "V": V, "z1": z1, "z2": z2, "sr0": sr0, "sr0_ann": sr0 * math.sqrt(365), "sr": sr, "T": T,
            "skew": g3, "kurtosis": g4, "den": den, "stat": stat, "dsr": dsr, "psr0": psr0, "faille": dsr < 0.95}


def veto(tid: str) -> int:
    need_full_registry()
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    j = load(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    out = {"id": tid, "S": S, "eligible": S > ELIG}
    p(f"=== VETO {tid} : S {S:+.6f} ; trades {j['concatenated']['model']['trades']} ; éligible (S > {ELIG}) "
      f"{S > ELIG} ; commit {j['commit'][:7]} ===")
    if not S > ELIG:
        p("non éligible : tests du veto non calculés")
    else:
        t = json.loads((RES / "rules_t1" / f"{tid}_t1.json").read_text(encoding="utf-8"))
        assert abs(t["S"] - S) < 1e-12
        p(f"\nT1-règles (100 séries mélangées par blocs de {t['block']} bougies, seeds 1-100) : moyenne {t['mean']:+.4f} "
          f"écart-type {t['std']:.4f} p95 {t['p95']:+.4f} ; z {t['z']:+.3f} ; séries >= S {t['n_ge']}/100 ; "
          f"contrôle identité {t['identity_sharpe']:+.6f} -> {flag(t['faille'])}")
        out["T1"] = {k: t[k] for k in ("mean", "std", "p95", "z", "n_ge", "faille", "identity_sharpe")}
        a = json.loads((RES / f"rules_{tid}" / "analyse_e10.json").read_text(encoding="utf-8"))
        pr, ps = a["random"]["percentile"], a["bh_same_expo"]["stratified_percentile"]
        f2 = pr <= 95 or ps <= 95
        p(f"\nT2 aléatoire mêmes trades / même exposition (analyse_e10) : grille 1 h percentile {pr:.1f} (p95 aléatoire "
          f"{a['random']['p95']:+.4f}, moyenne {a['random']['mean']:+.4f}) ; stratifié par fold percentile {ps:.1f} "
          f"(p95 {a['bh_same_expo']['stratified_p95']:+.4f}) -> {flag(f2)}")
        out["T2"] = {"percentile": pr, "p95": a["random"]["p95"], "mean": a["random"]["mean"], "strat_percentile": ps,
                     "strat_p95": a["bh_same_expo"]["stratified_p95"], "faille": f2}
        p("\nT7 Sharpe déflaté :")
        out["T7"] = dsr_block(j, tid, p)
        p(f"(version analyse_e10 : DSR = {a['dsr']['dsr']:.6f}, N = {a['dsr']['N']})")
        out["T7"]["dsr_analyse_e10"] = a["dsr"]["dsr"]
        out["veto"] = {"T1": t["faille"], "T2": f2, "T7": out["T7"]["faille"]}
        p(f"\nVETO {tid} : T1 {flag(t['faille'])} ; T2 {flag(f2)} ; T7 {flag(out['T7']['faille'])}")
    (RES / f"{tid}_veto.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n",
                                          encoding="utf-8")
    (RES / f"{tid}_veto.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ synthèse et choix du candidat

def summary() -> int:
    need_full_registry()
    log = harness.Tee()
    rows = []
    log("ID  | famille | barrière | Sharpe concat | trades | DD max | expo | rdt tot | folds>0 | IS 2019-20 | "
        "haute/basse/temps OOS | éligible | égalité baseline (max |écart|)")
    for tid in IDS:
        j = load(tid)
        m = j["concatenated"]["model"]
        eq = max(abs(fl(v)) for v in j["baseline_equality"].values())
        e = j["exits_oos"]
        S = fl(m["sharpe"])
        rows.append((tid, S))
        log(f"{tid} | {j['config']['family']:<7} | {j['config']['barrier']:<8} | {S:+.4f} | {m['trades']:>4} | "
            f"{fl(m['max_drawdown']):.4f} | {fl(m['exposure']):.3f} | {fl(m['total_return']):+.4f} | "
            f"{j['folds_sharpe_positive']}/9 | {fl(j['in_sample_2019_2020']['sharpe']):+.4f} | "
            f"{e['haute']}/{e['basse']}/{e['temps']} | {S > ELIG} | {eq:.2e}")
    j0 = load(IDS[0])
    base = fl(j0["concatenated"]["baseline"]["sharpe"])
    bh = fl(j0["concatenated"]["buy_and_hold"]["sharpe"])
    best = max(rows, key=lambda r: r[1])
    cand = best[0] if best[1] > base else None
    log(f"\nbaseline {base:+.4f} ; B&H {bh:+.4f} ; meilleur essai {best[0]} {best[1]:+.4f} ; règle JOURNAL "
        f"(Sharpe concaténé le plus élevé, > baseline) -> candidat : {cand if cand else 'aucun candidat'}")
    out = {"rows": rows, "baseline": base, "buy_and_hold": bh, "best": best[0], "candidate": cand}
    (RES / "rules_summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    (RES / "rules_summary.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ batterie du candidat

def full(tid: str) -> int:
    need_full_registry()
    from bot.baseline import run_buy_and_hold
    from bot.metrics import daily_returns, sharpe_daily, summarize
    log = harness.Tee()
    p = log
    j = load(tid)
    cfg = cfg_of(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    df = harness.load_dev(log)
    A, B = rules.CONCAT_START, rules.CONCAT_END
    r0 = rules.build(df, cfg)
    sig = pd.read_parquet(RES / f"rules_{tid}" / "signal_1h.parquet", engine="pyarrow")
    assert np.array_equal(sig["signal"].to_numpy(), r0["target"])
    res = rules.bt(r0["target"], df, A, B)
    m0 = summarize(res)
    assert abs(m0["sharpe"] - S) < 1e-12
    out = {"id": tid, "S": S}
    p(f"=== Batterie {tid} ({cfg['family']} / {cfg['barrier']}) : S {S:+.6f} ; trades OOS {m0['trades']} ===")

    def line(name, x, fx, extra=""):
        p(f"  {name:<44}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S){extra} -> {flag(fx)}")
        out[name] = {"sharpe": x, "faille": bool(fx)}

    p("\n--- coûts (même signal) ---")
    for k in (2.0, 3.0):
        mm = summarize(rules.bt(r0["target"], df, A, B, cost_mult=k))
        line(f"coûts x{k:.0f}", mm["sharpe"], mm["sharpe"] <= 0, f" rdt {mm['total_return']:+.4f}")
    p("\n--- exécution à t+2 ---")
    r2 = rules.build(df, cfg, delay=2)
    mm = summarize(rules.bt(r2["target"], df, A, B, delay=2))
    x = mm["sharpe"]
    line("t+2 moteur re-simulé (E = open t+2)", x, x <= 0 or x < 0.5 * S or x > 1.10 * S, f" trades {mm['trades']}")
    mm = summarize(rules.bt(r0["target"], df, A, B, delay=2))
    x = mm["sharpe"]
    line("t+2 signal figé (exec_delay=2)", x, x <= 0 or x < 0.5 * S or x > 1.10 * S, f" trades {mm['trades']}")
    p("\n--- barrières ±20 % (re-simulées) ---")
    for nm, kw in (("barrières de prix x0,8", {"b_scale": 0.8}), ("barrières de prix x1,2", {"b_scale": 1.2}),
                   ("barrière temporelle 96 h (x0,8)", {"horizon": 96}),
                   ("barrière temporelle 144 h (x1,2)", {"horizon": 144})):
        rr = rules.build(df, cfg, **kw)
        mm = summarize(rules.bt(rr["target"], df, A, B))
        x = mm["sharpe"]
        line(nm, x, x <= 0 or x < 0.5 * S, f" trades {mm['trades']}")

    p("\n--- années (signal continu, position héritée) ---")
    yrs = {}
    for y in range(2021, 2026):
        a = pd.Timestamp(f"{y}-01-01 00:00", tz="UTC")
        e = pd.Timestamp(f"{y}-12-31 23:00", tz="UTC") if y < 2025 else B
        mm = summarize(rules.bt(r0["target"], df, a, e))
        bh = run_buy_and_hold(df, a, e)
        yrs[y] = (mm["sharpe"], sharpe_daily(bh.equity))
        p(f"  {y} : règle Sharpe {mm['sharpe']:+.4f} rdt {mm['total_return']:+.4f} trades {mm['trades']} expo "
          f"{mm['exposure']:.3f} | B&H Sharpe {yrs[y][1]:+.4f} rdt {bh.equity.iloc[-1] - 1:+.4f}")
    neg = sum(1 for s_, _ in yrs.values() if s_ < 0)
    le = sum(1 for s_, b_ in yrs.values() if s_ <= b_)
    fy = neg >= 2 or le >= 4
    p(f"  années à Sharpe < 0 : {neg} ; années à Sharpe <= B&H : {le}/5 -> {flag(fy)}")
    out["years"] = {"values": {str(k): v for k, v in yrs.items()}, "neg": neg, "le_bh": le, "faille": fy}

    p("\n--- régimes (mois civil B&H > +5 % hausse, < -5 % baisse, sinon range) ---")
    o = df["open"]
    bhc = run_buy_and_hold(df, A, B)
    dr, drb = daily_returns(res.equity), daily_returns(bhc.equity)
    mo = o.loc[A:B].resample("MS").first()
    nxt = mo.shift(-1)
    nxt.iloc[-1] = o.loc[B]
    mret = nxt / mo - 1
    reg = pd.Series(np.where(mret > 0.05, "hausse", np.where(mret < -0.05, "baisse", "range")), index=mo.index)
    dmonth = dr.index.tz_convert("UTC").tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    day_reg = reg.reindex(dmonth).to_numpy()
    rg = {}
    for g in ("hausse", "baisse", "range"):
        mk = day_reg == g
        xx, xb = dr[mk], drb[mk]
        rg[g] = {"months": int((reg == g).sum()), "days": int(mk.sum()),
                 "sharpe": float(xx.mean() / xx.std(ddof=1) * math.sqrt(365)), "compound": float(np.prod(1 + xx) - 1),
                 "bh_sharpe": float(xb.mean() / xb.std(ddof=1) * math.sqrt(365)), "bh_compound": float(np.prod(1 + xb) - 1)}
        p(f"  {g:<7}: {rg[g]['months']} mois, {rg[g]['days']} jours | règle Sharpe {rg[g]['sharpe']:+.4f} rdt composé "
          f"{rg[g]['compound']:+.4f} | B&H Sharpe {rg[g]['bh_sharpe']:+.4f} rdt composé {rg[g]['bh_compound']:+.4f}")
    fr = rg["range"]["sharpe"] <= 0 or rg["hausse"]["sharpe"] <= 0 or rg["baisse"]["compound"] <= -0.5
    p(f"  -> {flag(fr)}")
    out["regimes"] = {**rg, "faille": fr}

    p("\n--- sans le meilleur mois ---")
    meq = res.equity.resample("MS").last()
    mr = meq / meq.shift(1).fillna(1.0) - 1
    bm = mr.idxmax()
    xx = dr[dmonth != bm]
    x = float(xx.mean() / xx.std(ddof=1) * math.sqrt(365))
    line(f"sans le meilleur mois ({bm.strftime('%Y-%m')}, {mr.max():+.4f})", x, x <= 0 or x < 0.5 * S)

    p("\n--- sans les meilleurs trades OOS ---")
    trs = res.trades.copy()
    tdf = pd.read_parquet(RES / f"rules_{tid}" / "trades.parquet", engine="pyarrow")
    idx = df.index
    order = trs.sort_values("net_return", ascending=False)
    for k in (1, 3, 5):
        tg = r0["target"].copy()
        removed = []
        for _, row in order.head(k).iterrows():
            et = row["entry_time"]
            cand = tdf[(tdf["entry_time"] == et)] if not row["open_at_start"] else tdf[tdf["entry_time"] < A].tail(1)
            assert len(cand) == 1
            c0 = cand.iloc[0]
            i0 = idx.get_loc(c0["decision_time"])
            i1 = idx.get_loc(c0["exit_decision_time"]) if pd.notna(c0["exit_decision_time"]) else len(idx)
            tg[i0:i1] = 0.0
            removed.append(f"{et} {row['net_return']:+.4f}")
        mm = summarize(rules.bt(tg, df, A, B))
        x = mm["sharpe"]
        line(f"sans les {k} meilleurs trades", x, x <= 0 or x < 0.5 * S,
             f" trades {mm['trades']} rdt {mm['total_return']:+.4f} ; retirés : {', '.join(removed)}")

    p("\n--- IC bootstrap par trade (trades OOS, rendements nets) ---")
    rr_ = trs["net_return"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    bs = rng.choice(rr_, size=(BOOT_B, len(rr_)), replace=True)
    means = bs.mean(axis=1)
    comp = np.prod(1 + bs, axis=1) - 1
    lo, hi = np.percentile(means, [2.5, 97.5])
    clo, chi = np.percentile(comp, [2.5, 97.5])
    fb = lo <= 0
    p(f"  {len(rr_)} trades ; moyenne {rr_.mean():+.5f} ; médiane {np.median(rr_):+.5f} ; IC95 moyenne [{lo:+.5f} ; "
      f"{hi:+.5f}] ; P(moyenne <= 0) {np.mean(means <= 0):.4f} ; rendement composé {np.prod(1 + rr_) - 1:+.4f} IC95 "
      f"[{clo:+.4f} ; {chi:+.4f}] (seed {BOOT_SEED}, {BOOT_B} tirages) -> {flag(fb)}")
    out["bootstrap"] = {"n": int(len(rr_)), "mean": float(rr_.mean()), "ci_mean": [float(lo), float(hi)],
                        "p_mean_le0": float(np.mean(means <= 0)), "ci_compound": [float(clo), float(chi)], "faille": fb}

    p("\n--- B&H à exposition égale (analyse_e10, T2b) ---")
    a = json.loads((RES / f"rules_{tid}" / "analyse_e10.json").read_text(encoding="utf-8"))
    cf, sp = a["bh_same_expo"]["constant_frac_sharpe"], a["bh_same_expo"]["stratified_percentile"]
    fe = S <= cf or sp <= 95
    p(f"  exposition {m0['exposure']:.4f} ; B&H fractionnaire constant Sharpe {cf:+.4f} ; aléatoire stratifié par fold "
      f"percentile {sp:.1f} -> {flag(fe)}")
    out["bh_same_expo"] = {"constant_frac_sharpe": cf, "strat_percentile": sp, "faille": fe}
    p(f"\n--- trades OOS : {m0['trades']} (informatif : GONOGO exigera >= 200 au palier 4) ---")
    out["trades_oos"] = int(m0["trades"])
    fl_ = [k for k, v in out.items() if isinstance(v, dict) and v.get("faille")]
    p(f"\nFAILLES de la batterie : {len(fl_)} : {', '.join(fl_) if fl_ else 'aucune'}")
    out["failles"] = fl_
    (RES / f"{tid}_battery.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n",
                                             encoding="utf-8")
    (RES / f"{tid}_battery.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ auto-tests

def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ÉCHEC'}] {name}")

    print("=== auto-tests T1-règles (synthétique) ===")
    df = rules.synth_1h(120, seed=2)
    for t in (100, 101, 102, 1500):
        df.loc[df.index[t], ["open", "high", "low", "close", "volume"]] = np.nan
        df.loc[df.index[t], "missing"] = True
    ident = make_synthetic(df, None)
    cols = ["open", "high", "low", "close"]
    err = np.nanmax(np.abs(ident[cols].to_numpy() / df[cols].to_numpy() - 1))
    check(f"permutation identité : série reconstruite = série d'origine (écart relatif max {err:.2e})",
          err < 1e-10 and np.array_equal(np.isnan(ident["close"]), np.isnan(df["close"])))
    syn = make_synthetic(df, 7, block=240)
    check("même grille de dates, même nombre de bougies manquantes", syn.index.equals(df.index)
          and int(syn["missing"].sum()) == int(df["missing"].sum()))
    lr0 = np.log(df["close"] / df["close"].ffill().shift(1)).dropna().to_numpy()
    lr1 = np.log(syn["close"] / syn["close"].ffill().shift(1)).dropna().to_numpy()
    check("multiset des log-rendements close à close conservé", np.allclose(np.sort(lr0), np.sort(lr1), atol=1e-12))
    check("high >= max(open, close) et low <= min(open, close) conservés",
          bool(((syn["high"] >= syn[["open", "close"]].max(axis=1) - 1e-9) | syn["missing"]).all()
               and ((syn["low"] <= syn[["open", "close"]].min(axis=1) + 1e-9) | syn["missing"]).all()))
    s2 = make_synthetic(df, 7, block=240)
    check("seed fixée -> série identique", s2.equals(syn))
    check("seeds différentes -> séries différentes", not make_synthetic(df, 8, block=240).equals(syn))
    # blocs : à l'intérieur d'un bloc, les rendements relatifs se suivent comme dans l'origine
    perm = np.random.default_rng(7).permutation(len(range(0, len(df) - 1, 240)))
    q = int(perm[0])
    a0 = 1 + q * 240
    seg_o = lr0[:0]
    rel_o = np.log(df["close"].to_numpy()[a0 + 1:a0 + 50] / df["close"].to_numpy()[a0:a0 + 49])
    rel_s = np.log(syn["close"].to_numpy()[2:51] / syn["close"].to_numpy()[1:50])
    check("1er bloc synthétique = bloc d'origine perm[0] (rendements identiques)", np.allclose(rel_o, rel_s, atol=1e-12))
    _ = seg_o
    print(f"auto-tests : {'OK' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--t1")
    ap.add_argument("--t2")
    ap.add_argument("--veto")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--full")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.t1:
        return t1(a.t1)
    if a.t2:
        return t2(a.t2)
    if a.veto:
        return veto(a.veto)
    if a.summary:
        return summary()
    if a.full:
        return full(a.full)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
