"""Pièces de E17 (brief P3-B3, KO C3-07 / C3-10 / C3-12), à partir des JSON de experiments/results/.

Usage : .venv\\Scripts\\python experiments\\e17_report.py
Prérequis : nested.py exécuté (base, shuf1..shuf10, delay2, featlag1, drop_<feature la plus importante>).
1. tests/adversarial/analyse_e10.py IMPORTÉ et appelé tel quel (non modifié) : seuls ses
   constantes de module OUT (dossier de lecture du signal / d'écriture) et CAND (Sharpe attendu
   du signal relu) sont remplacées pour pointer vers E17 : aléatoire mêmes trades / même
   exposition (1000 tirages + stratifié par fold), B&H fractionnaire, t+2 et coûts sur signal
   figé, années, régimes mensuels +/-5 %, sans meilleur mois, DSR (sa version).
2. labels mélangés (10 seeds) : moyenne, écart-type, z, percentile de E17.
3. variantes re-tunées (procédure emboîtée complète) : delay2, featlag1, drop_<feature>.
4. DSR (Bailey & López de Prado 2014) avec N = lignes de REGISTRE.md, valeurs intermédiaires.
5. choix internes, importances, in-sample / OOS.
Écrit experiments/results/adv_E17/analyse_E17.txt et experiments/results/E17_report.json.
"""

from __future__ import annotations

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
ADV = RES / "adv_E17"
EULER = 0.5772156649015329
BASELINE_SHARPE = -0.530


class Cap(io.StringIO):
    def reconfigure(self, **kw):
        pass


def f(x):
    return float(x) if not isinstance(x, str) else float("nan")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    e17 = json.loads((RES / "E17.json").read_text(encoding="utf-8"))
    cand = f(e17["concatenated"]["model"]["sharpe"])
    out = {"E17_sharpe": cand}

    # ------------------------------------------------ 1. batterie adversariale (script importé)
    sys.path.insert(0, str(ROOT / "tests" / "adversarial"))
    import analyse_e10  # noqa: E402
    analyse_e10.OUT = ADV
    analyse_e10.CAND = cand
    argv = sys.argv
    sys.argv = ["analyse_e10.py", "--draws", "1000", "--seed", "20261004"]
    buf = Cap()
    with redirect_stdout(buf):
        analyse_e10.main()
    sys.argv = argv
    txt = buf.getvalue()
    (ADV / "analyse_E17.txt").write_text(txt, encoding="utf-8")
    print("=== tests/adversarial/analyse_e10.py appliqué au signal E17 (sortie brute) ===")
    print(txt)
    out["analyse"] = json.loads((ADV / "analyse_e10.json").read_text(encoding="utf-8"))

    # ------------------------------------------------ 2. labels mélangés
    print("=== Labels mélangés, procédure emboîtée complète ===")
    sh = {}
    for s in range(1, 11):
        p = RES / f"E17_shuf{s}.json"
        j = json.loads(p.read_text(encoding="utf-8"))
        m = j["concatenated"]["model"]
        sh[s] = f(m["sharpe"])
        print(f"  seed {s:>2} : Sharpe concat {sh[s]:+.4f} trades {m['trades']} expo {m['exposure']:.3f} "
              f"choix {';'.join(r['chosen_config'] for r in j['folds'])}")
    v = np.array(list(sh.values()))
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    z = (cand - mu) / sd
    pct = float(np.mean(v < cand) * 100)
    print(f"  moyenne {mu:+.4f} écart-type {sd:.4f} min {v.min():+.4f} max {v.max():+.4f} ; "
          f"E17 {cand:+.4f} : z {z:+.3f}, percentile {pct:.0f} (seeds < E17 : {int(np.sum(v < cand))}/10)")
    out["shuffled"] = {"sharpe": sh, "mean": mu, "std": sd, "z": z, "percentile": pct}

    # ------------------------------------------------ 3. variantes re-tunées
    print("\n=== Variantes re-tunées (procédure emboîtée complète, tuning et évaluation perturbés) ===")
    top = e17["most_important_feature"]
    out["variants"] = {}
    for var in ("delay2", "featlag1", f"drop_{top}"):
        j = json.loads((RES / f"E17_{var}.json").read_text(encoding="utf-8"))
        m = j["concatenated"]["model"]
        print(f"  {var:<20}: Sharpe concat {f(m['sharpe']):+.4f} ({f(m['sharpe']) / cand * 100 if cand else float('nan'):.0f} % de E17) "
              f"trades {m['trades']} expo {m['exposure']:.3f} DD {m['max_drawdown']:.4f} "
              f"folds>0 {j['folds_sharpe_positive']}/9 ; choix {';'.join(r['chosen_config'] for r in j['folds'])}")
        out["variants"][var] = {"sharpe": f(m["sharpe"]), "trades": m["trades"], "exposure": m["exposure"]}

    # ------------------------------------------------ 4. DSR
    print("\n=== Sharpe déflaté (Bailey & López de Prado 2014), unités journalières ===")
    lines = (EXP / "REGISTRE.md").read_text(encoding="utf-8").splitlines()
    ids = [l.split("|")[1].strip() for l in lines]
    N = len(lines)
    srs = np.array([json.loads((RES / f"{i}.json").read_text(encoding="utf-8"))["daily_oos"]["sr_daily"] for i in ids])
    V = float(np.var(srs, ddof=1))
    d = e17["daily_oos"]
    sr, T, g3, g4 = d["sr_daily"], d["n_days"], d["skew"], d["kurtosis"]
    z1, z2 = norm.ppf(1 - 1 / N), norm.ppf(1 - 1 / (N * math.e))
    sr0 = math.sqrt(V) * ((1 - EULER) * z1 + EULER * z2)
    den = math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    stat = (sr - sr0) * math.sqrt(T - 1) / den
    dsr = float(norm.cdf(stat))
    psr0 = float(norm.cdf(sr * math.sqrt(T - 1) / den))
    print("SR0 = sqrt(V[SR_n]) * ((1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(N e)))")
    print("DSR = Φ( (SR - SR0) sqrt(T-1) / sqrt(1 - γ3 SR + (γ4-1)/4 SR²) )")
    print(f"N = {N} lignes de REGISTRE.md ({ids[0]}..{ids[-1]}) ; SR journaliers : {np.round(srs, 6).tolist()}")
    print(f"V[SR_n] (ddof=1) = {V:.6e} ; sqrt(V) = {math.sqrt(V):.6f}")
    print(f"Φ⁻¹(1-1/N) = {z1:.6f} ; Φ⁻¹(1-1/(N e)) = {z2:.6f}")
    print(f"SR0 journalier = {sr0:.6f} (annualisé {sr0 * math.sqrt(365):.4f})")
    print(f"SR E17 journalier = {sr:.6f} (annualisé {cand:.4f}) ; T = {T} ; skew γ3 = {g3:.6f} ; kurtosis γ4 = {g4:.6f}")
    print(f"SR - SR0 = {sr - sr0:+.6f} ; dénominateur = {den:.6f} ; statistique = {stat:.6f}")
    print(f"DSR = {dsr:.6f} ; PSR(0) = {psr0:.6f}")
    out["dsr"] = {"N": N, "V": V, "z1": z1, "z2": z2, "sr0": sr0, "sr": sr, "T": T, "skew": g3,
                  "kurtosis": g4, "den": den, "stat": stat, "dsr": dsr, "psr0": psr0}

    # ------------------------------------------------ 5. choix, importances, IS/OOS
    print("\n=== Choix internes par fold ===")
    for r in e17["folds"]:
        top3 = sorted(r["per_config_best"], key=lambda q: -f(q["best_val_sharpe"]))[:3]
        print(f"fold {r['fold']} : {r['chosen_config']} ({r['chosen_model']}, H={r['chosen_H']}) score val "
              f"{f(r['best_val_sharpe']):.4f} ; 2e/3e : " + ", ".join(
                  f"{q['config']} {f(q['best_val_sharpe']):.4f}" for q in top3[1:]))
    print("\n=== In-sample vs OOS ===")
    print("fold | config | Sharpe IS | Sharpe val | Sharpe OOS | AUC train | AUC val | AUC test")
    for r in e17["folds"]:
        print(f"{r['fold']:>4} | {r['chosen_config']:>6} | {f(r['sharpe_in_sample']):>9.3f} | "
              f"{f(r['best_val_sharpe']):>10.3f} | {f(r['model']['sharpe']):>10.3f} | {f(r['auc_train']):>9.4f} | "
              f"{f(r['auc_val']):>7.4f} | {f(r['auc_test']):>8.4f}")
    print("\n=== Importances normalisées (règle E17.json) ===")
    feats = sorted(e17["importance_norm_mean"], key=lambda q: -e17["importance_norm_mean"][q])
    print("feature        | moyenne | " + " | ".join(f"f{r['fold']}({r['importance_kind'][:4]})" for r in e17["folds"]))
    for fn in feats:
        print(f"{fn:<14} | {e17['importance_norm_mean'][fn]:.4f}  | " +
              " | ".join(f"{r['importance_norm'][fn]:>10.4f}" for r in e17["folds"]))
    elig = cand > BASELINE_SHARPE
    print(f"\nrègle JOURNAL : E17 {cand:+.4f} > baseline {BASELINE_SHARPE} : {elig}")
    out["eligible"] = elig
    with open(RES / "E17_report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
