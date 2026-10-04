"""Pièces du candidat (brief P3-B2 §4) à partir des JSON déjà produits par harness.py.

Usage : .venv\\Scripts\\python experiments\\candidate_report.py
- applique la règle de choix (JOURNAL.md) aux essais du registre ;
- Sharpe déflaté (Bailey & López de Prado 2014) sur les rendements journaliers OOS concaténés ;
- test labels mélangés (results/<ID>_shuf<seed>.json) ;
- importances (gain) moyennes et par fold ; écart in-sample / OOS par fold.
Aucune donnée n'est relue : uniquement les JSON de experiments/results/.
Écrit experiments/results/candidate_report.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
EULER = 0.5772156649015329
BASELINE_SHARPE = -0.530


def f(x):
    return float(x) if not isinstance(x, str) else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    lines = [l for l in (EXP / "REGISTRE.md").read_text(encoding="utf-8").splitlines()]
    ids = [l.split("|")[1].strip() for l in lines]
    N = len(lines)
    trials = {i: json.loads((RES / f"{i}.json").read_text(encoding="utf-8")) for i in ids}
    sh = {i: f(t["concatenated"]["model"]["sharpe"]) for i, t in trials.items()}
    print(f"registre : {N} lignes ; Sharpe OOS concaténés :")
    for i in ids:
        print(f"  {i} : {sh[i]:+.4f}")
    best = max(ids, key=lambda i: sh[i])
    cand = best if sh[best] > BASELINE_SHARPE else None
    print(f"règle de choix : max = {best} ({sh[best]:+.4f}) ; > {BASELINE_SHARPE} : {cand is not None}")
    out = {"n_registry_lines": N, "sharpe_concat": sh, "candidate": cand}
    if cand is None:
        print("aucun candidat")
    else:
        c = trials[cand]
        d = c["daily_oos"]
        sr_ann = sh[cand]
        sr = d["sr_daily"]
        T = d["n_days"]
        g3, g4 = d["skew"], d["kurtosis"]
        srs = np.array([trials[i]["daily_oos"]["sr_daily"] for i in ids])
        V = float(np.var(srs, ddof=1))
        z1 = norm.ppf(1 - 1 / N)
        z2 = norm.ppf(1 - 1 / (N * math.e))
        sr0 = math.sqrt(V) * ((1 - EULER) * z1 + EULER * z2)
        denom = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
        stat = (sr - sr0) * math.sqrt(T - 1) / denom
        dsr = float(norm.cdf(stat))
        psr0 = float(norm.cdf(sr * math.sqrt(T - 1) / denom))
        print()
        print("=== Sharpe déflaté (Bailey & López de Prado 2014), unités journalières ===")
        print("SR0 = sqrt(V[SR_n]) * ((1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(N e)))")
        print("DSR = Φ( (SR - SR0) sqrt(T-1) / sqrt(1 - γ3 SR + (γ4-1)/4 SR²) )")
        print(f"N = {N} ; V[SR_n] (ddof=1, SR journaliers des {N} essais) = {V:.6e} ; sqrt(V) = {math.sqrt(V):.6f}")
        print(f"Φ⁻¹(1-1/N) = {z1:.6f} ; Φ⁻¹(1-1/(N e)) = {z2:.6f} ; γ (Euler) = {EULER}")
        print(f"SR0 (journalier) = {sr0:.6f} (annualisé ×√365 : {sr0 * math.sqrt(365):.4f})")
        print(f"SR candidat (journalier) = {sr:.6f} (annualisé {sr_ann:.4f}) ; T = {T} jours")
        print(f"skewness γ3 = {g3:.6f} ; kurtosis γ4 (non excédentaire) = {g4:.6f}")
        print(f"dénominateur = {denom:.6f} ; statistique = {stat:.6f}")
        print(f"DSR = {dsr:.6f} ; (pour mémoire PSR(SR*=0) = {psr0:.6f})")
        out["deflated_sharpe"] = {"N": N, "V_sr_daily": V, "z_1_1N": z1, "z_1_1Ne": z2,
                                  "sr0_daily": sr0, "sr0_annual": sr0 * math.sqrt(365),
                                  "sr_daily": sr, "sr_annual": sr_ann, "T_days": T,
                                  "skew": g3, "kurtosis": g4, "denominator": denom,
                                  "statistic": stat, "dsr": dsr, "psr0": psr0}

        print()
        print("=== Test labels mélangés (même procédure complète, labels du train permutés) ===")
        shuf = {}
        for p in sorted(RES.glob(f"{cand}_shuf*.json")):
            s = json.loads(p.read_text(encoding="utf-8"))
            m = s["concatenated"]["model"]
            shuf[s["shuffle_seed"]] = {"sharpe": f(m["sharpe"]), "trades": m["trades"],
                                       "exposure": m["exposure"], "max_drawdown": m["max_drawdown"],
                                       "auc_test": [f(r["auc_test"]) for r in s["folds"]]}
            print(f"  seed {s['shuffle_seed']} : Sharpe OOS concat {f(m['sharpe']):+.4f}, trades {m['trades']}, "
                  f"expo {m['exposure']:.3f}, DD {m['max_drawdown']:.2%}, AUC test moyen "
                  f"{np.nanmean(shuf[s['shuffle_seed']]['auc_test']):.4f}")
        v = np.array([x["sharpe"] for x in shuf.values()])
        rank = int(np.sum(v >= sr_ann))
        print(f"  moyenne {v.mean():+.4f}, écart-type {v.std(ddof=1):.4f}, min {v.min():+.4f}, max {v.max():+.4f} ; "
              f"candidat {sr_ann:+.4f} ; seeds mélangés >= candidat : {rank}/{len(v)}")
        out["shuffled"] = {"runs": shuf, "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                           "n_ge_candidate": rank}

        print()
        print("=== In-sample vs OOS par fold ===")
        print("fold | Sharpe IS | Sharpe OOS | AUC train | AUC val int. | AUC test | hp choisis | mapping")
        for r in c["folds"]:
            print(f"{r['fold']:>4} | {f(r['sharpe_in_sample']):>9.3f} | {f(r['model']['sharpe']):>10.3f} | "
                  f"{f(r['auc_train']):>9.4f} | {f(r['auc_val']):>12.4f} | {f(r['auc_test']):>8.4f} | "
                  f"{json.dumps({k: r['best_hp'][k] for k in ('n_estimators', 'num_leaves', 'min_child_samples')})} | "
                  f"{json.dumps(r['best_map'])}")
        print()
        print("=== Importances (gain LightGBM, modèle réentraîné de chaque fold) ===")
        feats = c["features"]
        print("feature          | moyenne    | " + " | ".join(f"f{r['fold']:<8}" for r in c["folds"]))
        order = sorted(feats, key=lambda k: -c["importance_gain_mean"][k])
        for k in order:
            print(f"{k:<16} | {c['importance_gain_mean'][k]:>10.1f} | " +
                  " | ".join(f"{r['importance_gain'][k]:>9.1f}" for r in c["folds"]))
        out["importance_gain_mean"] = c["importance_gain_mean"]
        out["importance_gain_folds"] = {r["fold"]: r["importance_gain"] for r in c["folds"]}
    with open(RES / "candidate_report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
