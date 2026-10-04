"""Tests du veto (brief P3-B4, point 4) pour un essai éligible, batterie complète pour le candidat.

Usage :
    .venv\\Scripts\\python experiments\\b4_report.py --id E24 [--full]
Prérequis : results/<ID>.json (+ signal results/adv_<ID>/E10_adv_base_signal_full.csv),
results/<ID>_shuf1..10.json ; avec --full : <ID>_delay2, _featlag1, _cost2, _cost3, _drop_<top>.json.
À lancer APRÈS l'écriture de toutes les lignes du brief dans REGISTRE.md (N du DSR).

1. T1 labels mélangés (procédure complète, 10 seeds) : moyenne, écart-type, p95, z, rang.
2. tests/adversarial/analyse_e10.py IMPORTÉ et appelé tel quel (non modifié) ; seules ses constantes
   de module OUT (dossier du signal) et CAND (Sharpe attendu du signal relu, contrôle 1e-12) sont
   remplacées : T2 aléatoire mêmes trades / même exposition (1000 tirages) et stratifié par fold,
   T2b B&H fractionnaire même exposition, T3/T4 signal figé (t+2, coûts x2 / x3), T5a années,
   T5b régimes, T6b sans le meilleur mois, T7 DSR (sa version, Sharpe annualisés / sqrt(365)).
3. T7 DSR recalculé avec les SR journaliers de chaque essai du registre (comme e17_report.py).
4. --full : variantes re-tunées (procédure complète) delay2, featlag1, cost2, cost3, drop_<top>.
Les drapeaux FAILLE / RÉSISTE reprennent, à titre indicatif, les seuils de
reports/adversarial/criteres_E17.txt (T1 : S <= p95 ou z < 1,645 ; T2 : percentile <= 95 ;
T7 : DSR < 0,95 ; T3 : X <= 0, X < 0,5 S ou X > 1,10 S ; T4 : X <= 0 ; T6a/T6b : X <= 0 ou X < 0,5 S).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
RES = EXP / "results"
EULER = 0.5772156649015329


class Cap(io.StringIO):
    def reconfigure(self, **kw):
        pass


def fl(x):
    return float(x) if not isinstance(x, str) else float("nan")


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def flag(b):
    return "FAILLE" if b else "RÉSISTE"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    tid = args.id
    lines_out = []

    def p(s=""):
        print(s)
        lines_out.append(s)

    j = load(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    base = fl(j["concatenated"]["baseline"]["sharpe"])
    out = {"id": tid, "S": S, "baseline": base, "eligible": S > base}
    p(f"=== {tid} : Sharpe net OOS concaténé {S:+.6f} ; baseline {base:+.6f} ; éligible {S > base} ; trades "
      f"{j['concatenated']['model']['trades']} ; expo {j['concatenated']['model']['exposure']:.4f} ; commit {j['commit'][:7]} ===")

    # ------------------------------------------------ T1
    p(f"\n=== T1 labels mélangés, procédure complète ({args.seeds} seeds) ===")
    v = []
    for s in range(1, args.seeds + 1):
        js = load(f"{tid}_shuf{s}")
        m = js["concatenated"]["model"]
        v.append(fl(m["sharpe"]))
        extra = (" ; choix " + ";".join(r["chosen_config"] for r in js["folds"])) if js["config"].get("procedure") == "nested" else ""
        p(f"  seed {s:>2} : Sharpe concat {v[-1]:+.4f} trades {m['trades']} expo {m['exposure']:.3f}{extra}")
    v = np.array(v)
    mu, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
    p95 = float(np.nanpercentile(v, 95))
    z = (S - mu) / sd
    nge = int(np.sum(v >= S))
    f1 = (S <= p95) or (z < 1.645)
    p(f"  moyenne {mu:+.4f} écart-type {sd:.4f} min {np.nanmin(v):+.4f} max {np.nanmax(v):+.4f} p95 {p95:+.4f} ; "
      f"{tid} {S:+.4f} : z {z:+.3f} ; seeds >= {tid} : {nge}/{len(v)} ; S > p95 : {S > p95} -> {flag(f1)}")
    out["T1"] = {"sharpes": v.tolist(), "mean": mu, "std": sd, "p95": p95, "z": z, "n_ge": nge, "faille": f1}

    # ------------------------------------------------ analyse_e10 importé
    sys.path.insert(0, str(ROOT / "tests" / "adversarial"))
    import analyse_e10  # noqa: E402
    adv = RES / f"adv_{tid}"
    analyse_e10.OUT = adv
    analyse_e10.CAND = S
    argv = sys.argv
    sys.argv = ["analyse_e10.py", "--draws", "1000", "--seed", "20261004"]
    buf = Cap()
    with redirect_stdout(buf):
        analyse_e10.main()
    sys.argv = argv
    txt = buf.getvalue()
    (adv / "analyse.txt").write_text(txt, encoding="utf-8")
    p("\n=== tests/adversarial/analyse_e10.py appliqué au signal de " + tid + " (sortie brute) ===")
    for ln in txt.rstrip("\n").splitlines():
        p(ln)
    a = json.loads((adv / "analyse_e10.json").read_text(encoding="utf-8"))
    out["analyse_e10"] = a
    f2 = a["random"]["percentile"] <= 95
    f2s = a["bh_same_expo"]["stratified_percentile"] <= 95
    p(f"\nT2 aléatoire mêmes trades / même expo : percentile {a['random']['percentile']:.1f} -> {flag(f2)} ; "
      f"stratifié par fold : percentile {a['bh_same_expo']['stratified_percentile']:.1f} -> {flag(f2s)}")
    out["T2"] = {"percentile": a["random"]["percentile"], "p95": a["random"]["p95"],
                 "strat_percentile": a["bh_same_expo"]["stratified_percentile"], "faille": f2, "faille_strat": f2s}

    # ------------------------------------------------ T7 SR journaliers
    p("\n=== T7 Sharpe déflaté (Bailey & López de Prado 2014), SR journaliers de chaque essai du registre ===")
    reg = (EXP / "REGISTRE.md").read_text(encoding="utf-8").splitlines()
    ids = [ln.split("|")[1].strip() for ln in reg]
    N = len(reg)
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
    p(f"N = {N} lignes de REGISTRE.md ({ids[0]}..{ids[-1]}) ; V[SR_n] = {V:.6e} ; SR0 journalier {sr0:.6f} "
      f"(annualisé {sr0 * math.sqrt(365):.4f})")
    p(f"SR {tid} journalier {sr:.6f} ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; SR - SR0 {sr - sr0:+.6f} ; "
      f"stat {stat:.4f} ; DSR = {dsr:.6f} ; PSR(0) = {psr0:.6f} -> {flag(dsr < 0.95)}")
    p(f"(version analyse_e10 : DSR = {a['dsr']['dsr']:.6f}, N = {a['dsr']['N']})")
    out["T7"] = {"N": N, "V": V, "sr0": sr0, "sr": sr, "T": T, "skew": g3, "kurtosis": g4, "stat": stat,
                 "dsr": dsr, "psr0": psr0, "dsr_analyse_e10": a["dsr"]["dsr"], "faille": dsr < 0.95}

    if args.full:
        p("\n=== Batterie complète (candidat) : variantes re-tunées, procédure complète ===")
        top = j["most_important_feature"]
        out["variants"] = {}
        for var in ("delay2", "featlag1", "cost2", "cost3", f"drop_{top}"):
            jv = load(f"{tid}_{var}")
            m = jv["concatenated"]["model"]
            x = fl(m["sharpe"])
            if var in ("delay2", "featlag1"):
                fx = x <= 0 or x < 0.5 * S or x > 1.10 * S
                crit = "T3"
            elif var.startswith("cost"):
                fx = x <= 0
                crit = "T4"
            else:
                fx = x <= 0 or x < 0.5 * S
                crit = "T6a"
            p(f"  {crit} {var:<22}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) trades {m['trades']} expo "
              f"{m['exposure']:.3f} DD {m['max_drawdown']:.4f} folds>0 {jv['folds_sharpe_positive']}/9 -> {flag(fx)}")
            out["variants"][var] = {"sharpe": x, "trades": m["trades"], "faille": fx}
        for lab in ("exec_delay=2", "coûts x2", "coûts x3"):
            x = a[f"frozen_{lab}"]
            fx = (x <= 0 or x < 0.5 * S or x > 1.10 * S) if lab.startswith("exec") else x <= 0
            p(f"  signal figé {lab:<12}: Sharpe {x:+.4f} -> {flag(fx)}")
        x = a["no_best_month"]
        p(f"  T6b sans le meilleur mois : {x:+.4f} ({x / S * 100:.0f} % de S) -> {flag(x <= 0 or x < 0.5 * S)}")
        p("\n=== In-sample vs OOS par fold ===")
        p("fold | config | Sharpe IS | Sharpe val | Sharpe OOS | AUC train | AUC val | AUC test | taux y=1 train | test")
        for r in j["folds"]:
            p(f"{r['fold']:>4} | {r['chosen_config']:>6} | {fl(r['sharpe_in_sample']):>9.3f} | {fl(r['best_val_sharpe']):>10.3f} | "
              f"{fl(r['model']['sharpe']):>10.3f} | {fl(r['auc_train']):>9.4f} | {fl(r['auc_val']):>7.4f} | "
              f"{fl(r['auc_test']):>8.4f} | {fl(r['prior_train']):>14.4f} | {fl(r['label_rate_test']):.4f}")
        p("\n=== Importances (gain normalisé) ===")
        imp = j["importance_norm_mean"]
        feats = [k for k in sorted(imp, key=lambda q: -imp[q]) if imp[k] > 0]
        p("feature        | moyenne | " + " | ".join(f"f{r['fold']}" for r in j["folds"]))
        for fn in feats:
            p(f"{fn:<14} | {imp[fn]:.4f}  | " + " | ".join(f"{r['importance_norm'][fn]:.4f}" for r in j["folds"]))

    with open(RES / f"{tid}_veto.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
        fh.write("\n")
    with open(RES / f"{tid}_veto.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines_out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
