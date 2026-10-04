r"""Pièces du brief P3-B7 (n'est pas un essai).

Usage :
    .venv\Scripts\python experiments\b7_report.py --pairs              tableaux 9 folds + concaténé de E40-E47, paires
    .venv\Scripts\python experiments\b7_report.py --id E41 [--full]    tests du veto (+ batterie du candidat)

--pairs : pour chaque essai, modèle / baseline / B&H par fold et concaténé, lignes de train par fold, IS / OOS ;
  contrôle d'égalité baseline / B&H (baseline_equality des JSON) ; écart « f4 + fu » - « f4 seul » par fold et
  concaténé pour chaque paire. Sorties results/b7_pairs.txt / .json.
--id : 1) experiments/b5_report.py IMPORTÉ et exécuté tel quel (T1 labels mélangés 10 seeds, analyse_e10 importé :
  T2 grille 1 h et B&H même exposition, T3/T4 figés, T7 DSR avec N = lignes du registre ; --full : batterie de
  P3-B5) ; 2) T2 sur la GRILLE 4 h, 1000 tirages, non stratifié (seed 4747) et stratifié par fold (seed 4748), avec
  les fonctions place / seg_lengths de tests/adversarial/analyse_e36.py importées (non modifié) : segments de mêmes
  durées (en bougies 4 h) placés au hasard, mêmes trades et même exposition ; 3) --full : sensibilité des barrières
  x0,8 / x1,2 (variantes tbscale re-tunées), placebo de funding (fplac1..3), essai apparié « sans funding ».
  Veto T2 : FAILLE si l'un des percentiles (1 h, 1 h stratifié, 4 h, 4 h stratifié) est <= 95.
  Sorties : results/<ID>_veto.txt / .json (complétés).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
RES = EXP / "results"
sys.path.insert(0, str(EXP))
IDS = [f"E{i}" for i in range(40, 48)]
PAIRS = [("E40", "E41", "TB fixe H=30 (E36)"), ("E42", "E43", "net H=30 (E31)"), ("E44", "E45", "TB vol H=6 (E32)")]


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def fl(x):
    return float(x) if not isinstance(x, str) else float("nan")


def flag(b):
    return "FAILLE" if b else "RÉSISTE"


def pairs() -> int:
    import harness
    log = harness.Tee()
    out = {}
    for tid in IDS:
        j = load(tid)
        out[tid] = j
        cfg = j["config"]
        log(f"\n=== {tid} : features {cfg.get('features', 'union ' + ','.join(j['union_ids']))} ; label "
            f"{json.dumps(cfg.get('label', 'par fold'), ensure_ascii=False)} ; commit {j['commit'][:7]} ; code_dirty "
            f"{j['code_dirty']} ; index max lu {j['index_max_read']} ===")
        eq = j["baseline_equality"]
        log(f"contrôle baseline / B&H vs results/baseline_metrics.json : écart max {max(abs(fl(v)) for v in eq.values()):.2e}"
            f" ({len(eq)} valeurs)")
        log("fold | test début | n_train | train_start | modèle Sharpe / trades / DD / rdt | baseline Sharpe | B&H Sharpe "
            "| Sharpe IS | Sharpe val | AUC train | AUC val | AUC test | config")
        for r in j["folds"]:
            m, b, h = r["model"], r["baseline"], r["buy_and_hold"]
            log(f"{r['fold']:>4} | {r['test_start'][:10]} | {r['n_train']:>7} | {r['train_start'][:16]} | "
                f"{fl(m['sharpe']):+.3f} / {m['trades']:>3} / {fl(m['max_drawdown']):.4f} / {fl(m['total_return']):+.4f} | "
                f"{fl(b['sharpe']):+.3f} | {fl(h['sharpe']):+.3f} | {fl(r['sharpe_in_sample']):+.3f} | "
                f"{fl(r['best_val_sharpe']):+.3f} | {fl(r['auc_train']):.4f} | {fl(r['auc_val']):.4f} | "
                f"{fl(r['auc_test']):.4f} | {r['chosen_config']}")
        c = j["concatenated"]
        log(f"concat 2021-01 -> 2025-09 : modèle Sharpe {fl(c['model']['sharpe']):+.4f} trades {c['model']['trades']} "
            f"DD {fl(c['model']['max_drawdown']):.4f} rdt {fl(c['model']['total_return']):+.4f} expo "
            f"{fl(c['model']['exposure']):.4f} ; baseline {fl(c['baseline']['sharpe']):+.4f} ; B&H "
            f"{fl(c['buy_and_hold']['sharpe']):+.4f} ; folds > 0 : {j['folds_sharpe_positive']}/9")
        imp = j["importance_norm_mean"]
        log("importances moyennes : " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])
                                                 if v > 0))
    log("\n=== Paires : écart « f4 + fu » - « f4 seul » (mêmes lignes, mêmes hp, même mapping) ===")
    pj = {}
    for a, b, lab in PAIRS:
        ja, jb = out[a], out[b]
        same = [ra["n_train"] == rb["n_train"] and ra["train_start"] == rb["train_start"]
                for ra, rb in zip(ja["folds"], jb["folds"])]
        d = [fl(rb["model"]["sharpe"]) - fl(ra["model"]["sharpe"]) for ra, rb in zip(ja["folds"], jb["folds"])]
        dc = fl(jb["concatenated"]["model"]["sharpe"]) - fl(ja["concatenated"]["model"]["sharpe"])
        log(f"{lab} : {b} - {a} par fold {';'.join(f'{x:+.3f}' for x in d)} ; concaténé {dc:+.4f} ; folds où f4+fu > f4 : "
            f"{sum(x > 0 for x in d)}/9 ; mêmes lignes de train (n_train et début) : {all(same)}")
        pj[f"{b}-{a}"] = {"per_fold": d, "concat": dc, "same_rows": all(same)}
    with open(RES / "b7_pairs.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(pj, fh, indent=1)
        fh.write("\n")
    with open(RES / "b7_pairs.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    return 0


def t2_4h(tid: str, S: float, p, draws: int = 1000) -> dict:
    sys.path.insert(0, str(ROOT / "tests" / "adversarial"))
    from analyse_e10 import FOLDS, bt_from_h, ts  # noqa: E402
    from analyse_e36 import place, seg_lengths  # noqa: E402

    import harness
    from bot.backtest import run_backtest
    from bot.metrics import sharpe_daily
    log = harness.Tee()
    df = harness.load_dev(log)
    sig = pd.read_parquet(RES / f"adv_{tid}" / "signal_1h.parquet", engine="pyarrow")["signal"]
    A, B = harness.CONCAT_START, harness.CONCAT_END
    res = run_backtest(sig, df["open"], start=A, end=B, exec_delay=1, cost_per_side=0.0015)
    S0 = sharpe_daily(res.equity)
    assert abs(S0 - S) < 1e-12
    a_i = int(df.index.searchsorted(A))
    h0 = res.position.to_numpy()
    n = len(h0)
    assert n % 4 == 0 and A.hour == 0
    blk = h0.reshape(-1, 4)
    nonconst = int((blk.min(axis=1) != blk.max(axis=1)).sum())
    h4 = (blk.mean(axis=1) >= 0.5).astype(float)
    L4 = seg_lengths(h4)
    nb = len(h4)
    chk = sharpe_daily(bt_from_h(np.repeat(h4, 4), df, a_i).equity)
    p(f"\n=== T2 grille 4 h : {nb} blocs 4 h OOS, {len(L4)} segments, blocs exposés {int(L4.sum())} (expo "
      f"{L4.sum() / nb:.4f}), blocs à position non constante {nonconst} ; position 4 h reconstruite rebacktestée : "
      f"Sharpe {chk:+.4f} (S {S:+.4f}) ===")
    rng = np.random.default_rng(4747)
    s4 = np.array([sharpe_daily(bt_from_h(np.repeat(place(rng, nb, L4), 4), df, a_i).equity) for _ in range(draws)])
    pct4 = float(np.mean(s4 < S) * 100)
    p(f"  non stratifié (seed 4747, {draws} tirages) : moyenne {s4.mean():+.4f} p50 {np.percentile(s4, 50):+.4f} p95 "
      f"{np.percentile(s4, 95):+.4f} max {s4.max():+.4f} ; {tid} percentile {pct4:.1f} -> {flag(pct4 <= 95)}")
    idx4 = res.position.index[::4]
    plan = []
    for a_s, b_s in FOLDS:
        mk = (idx4 >= ts(a_s)) & (idx4 <= ts(b_s, True))
        plan.append((int(mk.sum()), seg_lengths(h4[mk])))
    rng = np.random.default_rng(4748)
    s5 = np.array([sharpe_daily(bt_from_h(np.repeat(np.concatenate([place(rng, a, b) for a, b in plan]), 4), df,
                                          a_i).equity) for _ in range(draws)])
    pct5 = float(np.mean(s5 < S) * 100)
    p(f"  stratifié par fold (seed 4748, {draws} tirages ; segments par fold {[len(b) for _, b in plan]}) : moyenne "
      f"{s5.mean():+.4f} p50 {np.percentile(s5, 50):+.4f} p95 {np.percentile(s5, 95):+.4f} max {s5.max():+.4f} ; "
      f"{tid} percentile {pct5:.1f} -> {flag(pct5 <= 95)}")
    return {"percentile_4h": pct4, "p95_4h": float(np.percentile(s4, 95)), "percentile_4h_strat": pct5,
            "p95_4h_strat": float(np.percentile(s5, 95)), "segments_4h": int(len(L4)), "check_sharpe": chk}


def extra_full(tid: str, j: dict, S: float, p, out: dict) -> None:
    p("\n=== Batterie P3-B7 (candidat) : barrières x0,8 / x1,2, placebo de funding, essai apparié sans funding ===")
    cfg = j["config"]
    labs = [c["label"] for c in j["effective_configs"]]
    if any(lb["kind"] == "tb" for lb in labs):
        for var in ("tbscale0.8", "tbscale1.2"):
            jv = load(f"{tid}_{var}")
            m = jv["concatenated"]["model"]
            x = fl(m["sharpe"])
            fx = x <= 0 or x < 0.5 * S
            p(f"  barrières {var:<11}: Sharpe {x:+.4f} ({x / S * 100:.0f} % de S) trades {m['trades']} folds>0 "
              f"{jv['folds_sharpe_positive']}/9 ; labels {[c['label'] for c in jv['effective_configs']]} -> {flag(fx)}")
            out.setdefault("b7", {})[var] = {"sharpe": x, "trades": m["trades"], "faille": fx}
    else:
        p("  barrières x0,8 / x1,2 : sans objet (aucun label triple barrière)")
    uses_fu = any(any(f.startswith("fr_") for f in c["features"]) for c in j["effective_configs"])
    if uses_fu:
        v = []
        for s in (1, 2, 3):
            jv = load(f"{tid}_fplac{s}")
            m = jv["concatenated"]["model"]
            v.append(fl(m["sharpe"]))
            p(f"  placebo funding fplac{s} (taux permutés par blocs de 30 j) : Sharpe {v[-1]:+.4f} trades {m['trades']} "
              f"folds>0 {jv['folds_sharpe_positive']}/9 ; importances funding "
              + ", ".join(f"{k} {x:.3f}" for k, x in jv["importance_norm_mean"].items() if k.startswith("fr_") and x > 0))
        pm = float(np.mean(v))
        fx = pm >= 0.75 * S
        p(f"  placebo funding : moyenne {pm:+.4f} ({pm / S * 100:.0f} % de S) ; max {max(v):+.4f} -> "
          f"{'FAILLE (le placebo garde >= 75 % de S : le gain ne vient pas de l information du funding)' if fx else 'RÉSISTE'}")
        out.setdefault("b7", {})["placebo"] = {"sharpes": v, "mean": pm, "faille": fx}
    else:
        p("  placebo funding : sans objet (le candidat n'utilise aucune feature de funding)")
    pair = cfg.get("pair")
    if pair:
        jp = load(pair)
        x = fl(jp["concatenated"]["model"]["sharpe"])
        d = [fl(r["model"]["sharpe"]) - fl(q["model"]["sharpe"]) for r, q in zip(j["folds"], jp["folds"])]
        p(f"  essai apparié {pair} (même config, features {jp['config']['features']}) : Sharpe {x:+.4f} ; {tid} - {pair} "
          f"= {S - x:+.4f} ; par fold {';'.join(f'{y:+.3f}' for y in d)}")
        out.setdefault("b7", {})["pair"] = {"id": pair, "sharpe": x, "delta": S - x, "per_fold": d}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--id")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.pairs:
        return pairs()
    tid = args.id
    import b5_report
    argv = sys.argv
    sys.argv = ["b5_report.py", "--id", tid] + (["--full"] if args.full else [])
    b5_report.main()
    sys.argv = argv
    vj = json.loads((RES / f"{tid}_veto.json").read_text(encoding="utf-8"))
    lines = (RES / f"{tid}_veto.txt").read_text(encoding="utf-8").rstrip("\n").splitlines()

    def p(s=""):
        print(s)
        lines.append(s)

    j = load(tid)
    S = fl(j["concatenated"]["model"]["sharpe"])
    vj["T2_4h"] = t2_4h(tid, S, p)
    pcts = {"1h": vj["T2"]["percentile"], "1h stratifié": vj["T2"]["strat_percentile"],
            "4h": vj["T2_4h"]["percentile_4h"], "4h stratifié": vj["T2_4h"]["percentile_4h_strat"]}
    f2 = any(v <= 95 for v in pcts.values())
    vj["veto"]["T2"] = f2
    p(f"\nT2 retenu (FAILLE si un percentile <= 95) : " + " ; ".join(f"{k} {v:.1f}" for k, v in pcts.items())
      + f" -> {flag(f2)}")
    if args.full:
        extra_full(tid, j, S, p, vj)
    p(f"\nVETO FINAL {tid} : T1 {flag(vj['veto']['T1'])} ; T2 {flag(vj['veto']['T2'])} ; T7 {flag(vj['veto']['T7'])}"
      f" (DSR {vj['T7']['dsr']:.6f}, N {vj['T7']['N']})")
    with open(RES / f"{tid}_veto.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(vj, fh, indent=1, ensure_ascii=False, default=float)
        fh.write("\n")
    with open(RES / f"{tid}_veto.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
