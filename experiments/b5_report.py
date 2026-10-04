"""Tests du veto (brief P3-B5, point 6) pour un essai éligible ; batterie complète pour le candidat.

Usage :
    .venv\\Scripts\\python experiments\\b5_report.py --id E29 [--full]
Prérequis : results/<ID>.json, results/adv_<ID>/signal_1h.parquet et proba_4h.parquet, results/<ID>_shuf1..10.json ;
avec --full : results/<ID>_<variante>.json pour delay2, featlag1, cost2, cost3, drop_<top>, tbfixed (si label TB vol),
tbdir, rule_vol_42, rule_<top> (si différente), dowplac51..53 (nodow : sans objet si aucune feature dow).
À lancer APRÈS l'écriture de toutes les lignes du brief dans REGISTRE.md (N du DSR = nombre de lignes).

1. T1 labels mélangés (procédure complète, 10 seeds) : moyenne, écart-type, p95, z, rang.
2. tests/adversarial/analyse_e10.py IMPORTÉ et appelé tel quel (non modifié) ; seules ses constantes de module
   OUT (dossier temporaire où le signal parquet est réécrit au format CSV attendu) et CAND (Sharpe attendu,
   contrôle 1e-12) sont remplacées : T2 aléatoire mêmes trades / même exposition (1000 tirages, grille 1 h) et
   stratifié par fold, T2b B&H fractionnaire même exposition, T3/T4 signal figé (exec +1 h, coûts x2 / x3),
   T5a années, T5b régimes, T6b sans le meilleur mois, T7 DSR (sa version).
3. T7 DSR recalculé avec les SR journaliers de chaque essai du registre (comme b4_report.py).
4. --full : signal figé exécuté une bougie 4 h plus tard (exec_delay 5 h) ; variantes re-tunées ; contrôles de
   circularité (seuils fixes, direction seule, règle sans apprentissage, décomposition de l'AUC hit / direction)
   et de calendrier ; IS / OOS par fold ; importances.
Drapeaux FAILLE / RÉSISTE indicatifs, seuils de reports/adversarial/criteres_E24.txt (T1 : S <= p95 ou z < 1,645 ;
T2 : percentile <= 95 ; T7 : DSR < 0,95 ; T3 : X <= 0, X < 0,5 S ou X > 1,10 S ; T4 : X <= 0 ; T6a / T6b / tbfixed /
tbdir : X <= 0 ou X < 0,5 S ; rule : X >= 0,75 S ; placebo : moyenne < 0,5 S).
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
EULER = 0.5772156649015329
sys.path.insert(0, str(EXP))


class Cap(io.StringIO):
    def reconfigure(self, **kw):
        pass


def fl(x):
    return float(x) if not isinstance(x, str) else float("nan")


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def flag(b):
    return "FAILLE" if b else "RÉSISTE"


def dsr_block(j, tid, p):
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
    p(f"SR {tid} journalier {sr:.6f} (annualisé {sr * math.sqrt(365):.4f}) ; T {T} ; skew {g3:.4f} ; kurtosis {g4:.4f} ; "
      f"SR - SR0 {sr - sr0:+.6f} ; stat {stat:.4f} ; DSR = {dsr:.6f} ; PSR(0) = {psr0:.6f} -> {flag(dsr < 0.95)}")
    return {"N": N, "V": V, "sr0": sr0, "sr": sr, "T": T, "skew": g3, "kurtosis": g4, "stat": stat,
            "dsr": dsr, "psr0": psr0, "faille": dsr < 0.95}


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
        assert js["shuffle_seed"] == s and js["config"]["id"] == tid
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
    sig = pd.read_parquet(adv / "signal_1h.parquet", engine="pyarrow")
    tmp = Path(tempfile.mkdtemp(prefix=f"b5_{tid}_"))
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

    # ------------------------------------------------ T7
    p("\n=== T7 Sharpe déflaté (Bailey & López de Prado 2014), SR journaliers de chaque essai du registre ===")
    out["T7"] = dsr_block(j, tid, p)
    out["T7"]["dsr_analyse_e10"] = a["dsr"]["dsr"]
    p(f"(version analyse_e10 : DSR = {a['dsr']['dsr']:.6f}, N = {a['dsr']['N']})")
    out["veto"] = {"T1": out["T1"]["faille"], "T2": out["T2"]["faille"], "T7": out["T7"]["faille"]}
    p(f"\nVETO {tid} : T1 {flag(out['veto']['T1'])} ; T2 {flag(out['veto']['T2'])} ; T7 {flag(out['veto']['T7'])}")

    if args.full:
        full(tid, j, S, a, sig, p, out)

    with open(RES / f"{tid}_veto.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
        fh.write("\n")
    with open(RES / f"{tid}_veto.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines_out) + "\n")
    return 0


def full(tid, j, S, a, sig, p, out):
    import agg4h
    import harness
    from bot.backtest import run_backtest
    from bot.metrics import summarize
    from sklearn.metrics import roc_auc_score

    p("\n=== Batterie complète (candidat) ===")
    log = harness.Tee()
    df1 = harness.load_dev(log)
    p(log.lines[-1])
    s1 = sig["signal"]
    assert s1.index.equals(df1.index)
    res = run_backtest(s1, df1["open"], start=harness.CONCAT_START, end=harness.CONCAT_END, exec_delay=1,
                       cost_per_side=0.0015)
    m0 = summarize(res)
    assert abs(m0["sharpe"] - S) < 1e-12
    r5 = summarize(run_backtest(s1, df1["open"], start=harness.CONCAT_START, end=harness.CONCAT_END, exec_delay=5,
                                cost_per_side=0.0015))
    x = r5["sharpe"]
    fx = x <= 0 or x < 0.5 * S or x > 1.10 * S
    p(f"  T3 signal figé exécuté 1 bougie 4 h plus tard (exec_delay 5 h) : Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) "
      f"trades {r5['trades']} -> {flag(fx)}")
    out["frozen_delay_4h"] = {"sharpe": x, "faille": fx}
    for lab in ("exec_delay=2", "coûts x2", "coûts x3"):
        x = a[f"frozen_{lab}"]
        fx = (x <= 0 or x < 0.5 * S or x > 1.10 * S) if lab.startswith("exec") else x <= 0
        p(f"  signal figé {lab:<12} (analyse_e10, grille 1 h) : Sharpe {x:+.4f} -> {flag(fx)}")
        out[f"frozen_{lab}_flag"] = fx
    x = a["no_best_month"]
    p(f"  T6b sans le meilleur mois : {x:+.4f} ({x / S * 100:.0f} % de S) -> {flag(x <= 0 or x < 0.5 * S)}")

    p("\n=== Variantes re-tunées (procédure complète b5.run) ===")
    top = j["most_important_feature"]
    cfg = j["config"]
    nested = cfg.get("procedure") == "nested"
    kinds = {c["label"]["kind"] for c in j["effective_configs"]}
    tbvol = any(c["label"]["kind"] == "tb" and c["label"]["up"]["type"] == "vol" for c in j["effective_configs"])
    has_dow = any("dow_sin" in c["features"] for c in j["effective_configs"])
    plan = [("delay2", "T3"), ("featlag1", "T3"), ("cost2", "T4"), ("cost3", "T4"), (f"drop_{top}", "T6a")]
    if tbvol:
        plan.append(("tbfixed", "C1 seuils fixes"))
    plan.append(("tbdir", "C1 direction seule"))
    if not nested:
        plan.append(("rule_vol_42", "C1 règle sans apprentissage"))
        if top != "vol_42":
            plan.append((f"rule_{top}", "C1 règle sans apprentissage"))
    if has_dow:
        plan.append(("nodow", "C2 sans jour de semaine"))
    plan += [("dowplac51", "C2 placebo"), ("dowplac52", "C2 placebo"), ("dowplac53", "C2 placebo")]
    out["variants"] = {}
    plac = []
    for var, crit in plan:
        jv = load(f"{tid}_{var}")
        m = jv["concatenated"]["model"]
        x = fl(m["sharpe"])
        if crit == "T3":
            fx = x <= 0 or x < 0.5 * S or x > 1.10 * S
        elif crit == "T4":
            fx = x <= 0
        elif crit.startswith("C1 règle"):
            fx = x >= 0.75 * S
        elif crit == "C2 placebo":
            plac.append(x)
            fx = None
        else:
            fx = x <= 0 or x < 0.5 * S
        eff = jv["effective_configs"][0]
        p(f"  {crit:<28} {var:<18}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) trades {m['trades']} expo "
          f"{m['exposure']:.3f} DD {m['max_drawdown']:.4f} rdt {m['total_return']:+.4f} folds>0 "
          f"{jv['folds_sharpe_positive']}/9" + ("" if fx is None else f" -> {flag(fx)}")
          + f" ; label {json.dumps(eff['label'], separators=(',', ':'))} ; {eff['model']} ; {len(eff['features'])} features")
        out["variants"][var] = {"sharpe": x, "trades": m["trades"], "faille": fx}
    if not has_dow:
        p("  C2 sans jour de semaine : sans objet, le candidat n'utilise aucune feature calendaire (jeu f4) ; "
          "le Sharpe sans jour de semaine est celui du candidat lui-même")
    pm = float(np.mean(plac))
    p(f"  C2 placebo (jour de semaine aléatoire par jour civil AJOUTÉ / substitué) : moyenne {pm:+.4f} "
      f"({pm / S * 100:.0f} % de S) -> {flag(pm < 0.5 * S)} (critère criteres_E24 ; pour un candidat sans calendrier, "
      f"mesure surtout l'effet d'une feature de bruit)")

    # C1e décomposition de l'AUC OOS
    pr = pd.read_parquet(RES / f"adv_{tid}" / "proba_4h.parquet", engine="pyarrow")
    df4 = agg4h.aggregate_4h(df1)
    F = agg4h.compute_features_4h(df4)
    assert pr.index.equals(df4.index)
    oos = (pr.index >= harness.CONCAT_START) & (pr.index <= harness.CONCAT_END)
    p("\n=== C1e Décomposition de l'AUC OOS (bougies 4 h du test, probas du modèle) ===")
    out["auc_decomp"] = {}
    for c in j["effective_configs"]:
        lab = c["label"]
        if lab["kind"] != "tb" or f"y_{c['id']}" not in pr.columns:
            continue
        u, d = agg4h.tb_barriers(df4, F, lab)
        j_up, j_dn = agg4h.tb_first_hits(df4, int(lab["H"]), u, d)
        y = pr[f"y_{c['id']}"].to_numpy()
        hit = (~np.isinf(j_up) | ~np.isinf(j_dn)).astype(float)
        dirn = (j_up < j_dn).astype(float)
        ok = oos & ~np.isnan(pr["p"].to_numpy()) & ~np.isnan(y) if not lab.get("expire_nan") else \
            oos & ~np.isnan(pr["p"].to_numpy()) & ~np.isnan(compute_fwd(df4, lab["H"]))
        pp = pr["p"].to_numpy()
        res_ = {}
        if not lab.get("expire_nan"):
            res_["auc_label"] = float(roc_auc_score(y[ok], pp[ok]))
        res_["auc_hit"] = float(roc_auc_score(hit[ok], pp[ok]))
        oh = ok & (hit == 1)
        res_["auc_dir_given_hit"] = float(roc_auc_score(dirn[oh], pp[oh]))
        res_["n"], res_["hit_rate"], res_["n_hit"] = int(ok.sum()), float(hit[ok].mean()), int(oh.sum())
        vv = F["vol_42"].to_numpy()
        okv = ok & ~np.isnan(vv)
        res_["spearman_p_minus_vol42"] = float(pd.Series(pp[okv]).corr(pd.Series(-vv[okv]), method="spearman"))
        p(f"  {c['id']} : " + " ; ".join(f"{k} {v_:.4f}" if isinstance(v_, float) else f"{k} {v_}" for k, v_ in res_.items()))
        out["auc_decomp"][c["id"]] = res_

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
    _ = kinds


def compute_fwd(df4, H):
    from bot.labels import compute_labels
    return compute_labels(df4, horizon=int(H))["fwd_ret"].to_numpy()


if __name__ == "__main__":
    sys.exit(main())
