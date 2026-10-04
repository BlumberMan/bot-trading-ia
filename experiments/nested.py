"""E17 : procédure emboîtée sur l'union des espaces E01-E16 (brief P3-B3).

Usage :
    .venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json [--registry]
    .venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json --variant shuf3
    .venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json --variant delay2|featlag1|drop_<feat>
    .venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json --out PATH   (relance de reproductibilité)
    .venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json --selftest   (données SYNTHÉTIQUES)

Procédure (règles détaillées dans experiments/configs/E17.json) : pour chaque fold k,
1. fenêtre de validation COMMUNE à toutes les configurations : train de référence
   split_fold(df, make_folds(24)[k], horizon=24), lignes propres sur l'union des features
   et le label H=24, val_start = ligne propre floor(0,8 n), val_end = train_end (H=24) ;
2. pour chaque configuration c de E01..E16 (fichiers commités, vérifiés par hash git) :
   train final de c = split_fold(df, make_folds(H_c)[k], horizon=H_c), lignes propres ;
   train interne = lignes t < val_start, purgées par purge_mask(val_start, val_end, horizon=24) ;
   chaque (hp, mapping) de c est scoré par le Sharpe net du backtest de la fenêtre commune
   (harness.window_sharpe : exec_delay=1, 0,0015/côté, x1) ;
3. choix = premier maximum dans l'ordre (c, hp, mapping) ; réentraînement de la configuration
   choisie sur tout son train final ; proba du test comme harness.py.
Évaluation identique à harness.run_trial (un backtest continu, baseline / B&H contrôlés).
Les fonctions du harnais (modèles, mapping, backtest de validation, baseline) sont importées,
pas recopiées ; les variantes adversariales modifient les mêmes attributs du harnais que
tests/adversarial/run_variant.py (EXEC_DELAY, build_matrix décalé, feature retirée).
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB (OpenMP) mono-thread : déterminisme

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from bot.data import sha256_file  # noqa: E402
from bot.features import FEATURES  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.metrics import daily_returns, summarize  # noqa: E402
from bot.split import DEFAULT_DATASET, DEV_END, DEV_START, H1, make_folds, purge_mask, split_fold  # noqa: E402
from extra_features import EXTRA_FEATURES  # noqa: E402

ROOT = harness.ROOT
RESULTS = harness.RESULTS
REGISTRE = harness.REGISTRE
H_COMMON = 24


# ------------------------------------------------------------------ union des espaces

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_union(cfg17: dict, log, check_git: bool = True) -> list[dict]:
    src = cfg17["union"]["source_commit"]
    out = []
    for i, ent in enumerate(cfg17["union"]["configs"]):
        p = ROOT / ent["path"]
        local = git_blob_sha1(p)
        ok = local == ent["git_blob"]
        at_src = harness.git("rev-parse", f"{src}:{ent['path']}") if check_git else ent["git_blob"]
        ok2 = at_src == ent["git_blob"]
        if not (ok and ok2):
            raise RuntimeError(f"{ent['id']} : hash {local} / {at_src} != déclaré {ent['git_blob']}")
        c = json.loads(p.read_text(encoding="utf-8"))
        assert c["id"] == ent["id"]
        out.append({"ci": i, "id": c["id"], "model": c["model"], "H": int(c["horizon"]),
                    "features": harness.resolve_features(c["features"]),
                    "hps": harness.hp_grid(c), "maps": harness.mapping_grid(c), "seed": int(c["seed"])})
    tot = sum(len(c["hps"]) * len(c["maps"]) for c in out)
    log(f"[union] {len(out)} configurations vérifiées (blob git local = blob à {src} = déclaré) ; "
        f"{tot} couples (hp, mapping) au total ; H = {sorted({c['H'] for c in out})}")
    for c in out:
        log(f"  {c['id']} : {c['model']} H={c['H']} {len(c['features'])} features, "
            f"{len(c['hps'])} hp x {len(c['maps'])} mappings")
    return out


# ------------------------------------------------------------------ fold

def config_train(df, X, y, c, k, shuffle_seed):
    fH = make_folds(horizon=c["H"])[k - 1]
    assert fH.k == k
    tr, _ = split_fold(df, fH, horizon=c["H"])
    Xa = X.loc[tr.index, c["features"]]
    ya = y[c["H"]].loc[tr.index]
    clean = (~Xa.isna().any(axis=1)) & (~ya.isna())
    Xtr = Xa.loc[clean]
    ytr = ya.loc[clean].to_numpy().copy()
    if shuffle_seed is not None:
        ytr = np.random.default_rng([shuffle_seed, k, c["ci"]]).permutation(ytr)
    return fH, Xtr, ytr


def run_fold(f24, df, X, y, union, seed, shuffle_seed, log) -> tuple[dict, object]:
    k = f24.k
    tr24, _ = split_fold(df, f24, horizon=H_COMMON)
    all_feats = list(X.columns)
    Xc = X.loc[tr24.index, all_feats]
    yc = y[H_COMMON].loc[tr24.index]
    cl = (~Xc.isna().any(axis=1)) & (~yc.isna())
    idx_c = tr24.index[cl.to_numpy()]
    n = len(idx_c)
    v0 = int(math.floor((1.0 - harness.VAL_FRAC) * n))
    val_start, val_end = idx_c[v0], f24.train_end
    vmask = (df.index >= val_start) & (df.index <= val_end)
    vidx = df.index[vmask]
    open_v = df["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= f24.test_start - H1

    per_cfg = []
    best = None
    for c in union:
        _, Xtr, ytr = config_train(df, X, y, c, k, shuffle_seed)
        t = Xtr.index
        inner = np.asarray(t < val_start) & purge_mask(t, val_start, val_end, horizon=H_COMMON)
        assert not np.any(np.asarray(t[inner] >= val_start))
        vrows = np.asarray((t >= val_start) & (t <= val_end))
        Xin, yin = Xtr.iloc[inner], ytr[inner]
        Xv = X.loc[vidx, c["features"]]
        cbest = None
        for hi, hp in enumerate(c["hps"]):
            mdl = harness.make_model(c["model"], hp, seed).fit(Xin.to_numpy(), yin)
            pv = harness.predict_proba(mdl, Xv)
            auc_v = harness.auc(ytr[vrows], harness.predict_proba(mdl, Xtr.iloc[vrows]))
            for mi, mp in enumerate(c["maps"]):
                s, ntr = harness.window_sharpe(pv, vidx, mp, open_v, val_start, val_end)
                score = s if (isinstance(s, float) and math.isfinite(s)) else -math.inf
                if cbest is None or score > cbest["score"]:
                    cbest = {"score": score, "hp_i": hi, "map_i": mi, "val_trades": ntr, "auc_val": auc_v}
        rec = {"config": c["id"], "best_val_sharpe": cbest["score"], "hp_i": cbest["hp_i"],
               "map_i": cbest["map_i"], "val_trades": cbest["val_trades"], "auc_val": cbest["auc_val"],
               "n_inner_train": int(inner.sum()), "n_val_rows": int(vrows.sum())}
        per_cfg.append(rec)
        if best is None or cbest["score"] > best["score"]:
            best = {**cbest, "cfg": c}
    c = best["cfg"]
    hp, mp = c["hps"][best["hp_i"]], c["maps"][best["map_i"]]
    fH, Xtr, ytr = config_train(df, X, y, c, k, shuffle_seed)
    model = harness.make_model(c["model"], hp, seed).fit(Xtr.to_numpy(), ytr)

    auc_tr = harness.auc(ytr, harness.predict_proba(model, Xtr))
    t_first = Xtr.index[0]
    is_idx = df.index[(df.index >= t_first) & (df.index <= fH.train_end)]
    sh_is, ntr_is = harness.window_sharpe(harness.predict_proba(model, X.loc[is_idx, c["features"]]), is_idx, mp,
                                          df["open"].loc[t_first: fH.train_end + H1], t_first, fH.train_end)
    # importances (règle E17.json)
    imp = {fn: 0.0 for fn in all_feats}
    if c["model"] == "lgbm":
        g = model.booster_.feature_importance(importance_type="gain").astype(float)
        imp_kind = "gain"
    else:
        from sklearn.inspection import permutation_importance
        vr = np.asarray((Xtr.index >= val_start) & (Xtr.index <= val_end))
        pi = permutation_importance(model, Xtr.iloc[vr].to_numpy(), ytr[vr], scoring="roc_auc",
                                    n_repeats=3, random_state=seed, n_jobs=1)
        g = np.clip(pi.importances_mean, 0.0, None)
        imp_kind = "permutation_auc"
    tot = g.sum()
    for fn, v in zip(c["features"], g):
        imp[fn] = float(v / tot) if tot > 0 else 0.0
    top = sorted(per_cfg, key=lambda r: -r["best_val_sharpe"])[:5]
    log(f"  fold {k}: validation commune {val_start} -> {val_end} ({n - v0} lignes propres H=24) ; "
        f"CHOIX {c['id']} ({c['model']}, H={c['H']}) hp#{best['hp_i']} {hp} map {mp} : Sharpe val "
        f"{best['score']:.4f} ({best['val_trades']} trades), AUC val {best['auc_val']:.4f} ; "
        f"train final {Xtr.index[0]} -> {fH.train_end} ({len(Xtr)} lignes), AUC train {auc_tr:.4f}, "
        f"Sharpe IS {sh_is:.3f}")
    log("    top 5 configurations (meilleur score de validation) : "
        + ", ".join(f"{r['config']} {r['best_val_sharpe']:.4f}" for r in top))
    r = {"fold": k, "val_start": val_start, "val_end": val_end, "n_val_common": n - v0,
         "chosen_config": c["id"], "chosen_model": c["model"], "chosen_H": c["H"],
         "chosen_features": c["features"], "best_hp": hp, "best_map": mp,
         "best_val_sharpe": best["score"], "best_val_trades": best["val_trades"], "auc_val": best["auc_val"],
         "train_start": Xtr.index[0], "train_end": fH.train_end, "n_train": len(Xtr),
         "test_start": f24.test_start, "test_end": f24.test_end,
         "auc_train": auc_tr, "sharpe_in_sample": sh_is, "trades_in_sample": ntr_is,
         "importance_kind": imp_kind, "importance_norm": imp, "per_config_best": per_cfg}
    return r, model


# ------------------------------------------------------------------ essai

def run_nested(cfg17: dict, union: list[dict], shuffle_seed, log, df=None, skip_baseline=False) -> dict:
    seed = int(cfg17["seed"])
    if df is None:
        df = harness.load_dev(log)
    assert df.index.max() <= harness.MAX_INDEX
    folds = make_folds(horizon=H_COMMON)
    log(f"essai {cfg17['id']} : procédure emboîtée, union {len(union)} configurations, seed {seed}"
        + (f", LABELS MÉLANGÉS seed {shuffle_seed}" if shuffle_seed is not None else "")
        + f" ; exec_delay {harness.EXEC_DELAY}, coût {harness.COST_PER_SIDE} x{harness.COST_MULTIPLIER}")
    if skip_baseline:
        base_rows = [{"baseline": None, "buy_and_hold": None} for _ in folds]
        base_concat, base_diffs = {"baseline": None, "buy_and_hold": None}, {}
    else:
        base_rows, base_concat, base_diffs = harness.baseline_block(df, make_folds(), log)
    all_feats = list(FEATURES) + list(EXTRA_FEATURES)
    X = harness.build_matrix(df, all_feats)
    y = {H: compute_labels(df, horizon=H)["label"] for H in sorted({c["H"] for c in union} | {H_COMMON})}
    assert X.index.max() <= harness.MAX_INDEX and all(v.index.max() <= harness.MAX_INDEX for v in y.values())

    n = len(df)
    p_all = np.full(n, np.nan)
    enter = np.full(n, np.inf)
    exit_ = np.full(n, -np.inf)
    mh = np.zeros(n, dtype=np.int64)
    fold_out = []
    for f in folds:
        r, model = run_fold(f, df, X, y, union, seed, shuffle_seed, log)
        a = f.test_start - (2 if f.k == 1 else 1) * H1
        b = f.test_end if f.k == len(folds) else f.test_end - H1
        cm = (df.index >= a) & (df.index <= b)
        p_all[cm] = harness.predict_proba(model, X.loc[cm, r["chosen_features"]])
        enter[cm], exit_[cm], mh[cm] = r["best_map"]["enter"], r["best_map"]["exit"], r["best_map"]["min_hold"]
        tm = (df.index >= f.test_start) & (df.index <= f.test_end)
        r["auc_test"] = harness.auc(y[r["chosen_H"]].loc[tm].to_numpy(), p_all[tm])
        r["signal_cover"] = [a, b]
        fold_out.append(r)

    sig = pd.Series(harness.map_positions(p_all, enter, exit_, mh), index=df.index, name="model_signal")
    assert sig.loc[: harness.CONCAT_START - 3 * H1].isna().all()
    bt = dict(exec_delay=harness.EXEC_DELAY, cost_per_side=harness.COST_PER_SIDE,
              cost_multiplier=harness.COST_MULTIPLIER)
    for f, r in zip(folds, fold_out):
        r["model"] = summarize(harness.run_backtest(sig, df["open"], start=f.test_start, end=f.test_end, **bt))
    res_c = harness.run_backtest(sig, df["open"], start=harness.CONCAT_START, end=harness.CONCAT_END, **bt)
    m_c = summarize(res_c)
    dr = daily_returns(res_c.equity).to_numpy()
    from scipy.stats import kurtosis, skew
    daily = {"n_days": int(len(dr)), "mean": float(dr.mean()), "std": float(dr.std(ddof=1)),
             "sr_daily": float(dr.mean() / dr.std(ddof=1)), "skew": float(skew(dr, bias=False)),
             "kurtosis": float(kurtosis(dr, fisher=False, bias=False))}

    log()
    log("=== Choix internes par fold (validation commune, Sharpe net) ===")
    for r in fold_out:
        log(f"fold {r['fold']} : {r['chosen_config']} {r['chosen_model']} H={r['chosen_H']} "
            f"hp {json.dumps(r['best_hp'])} map {json.dumps(r['best_map'])} score val {r['best_val_sharpe']:.4f}")
    log()
    log("=== Par fold (test) : modèle / baseline SMA168 / buy & hold, mêmes folds, mêmes coûts ===")
    log(f"{'fold':<16} | {'stratégie':<9} | " + harness.metrics_header())
    for f, r, b in zip(folds, fold_out, base_rows):
        log(f"{f'{f.k} {f.test_start.date()}':<16} | {'modèle':<9} | " + harness.fmt_metrics(r["model"]))
        if b["baseline"] is not None:
            log(f"{'':<16} | {'baseline':<9} | " + harness.fmt_metrics(b["baseline"]))
            log(f"{'':<16} | {'B&H':<9} | " + harness.fmt_metrics(b["buy_and_hold"]))
    log(f"{'concat 2021-25':<16} | {'modèle':<9} | " + harness.fmt_metrics(m_c))
    if base_concat["baseline"] is not None:
        log(f"{'':<16} | {'baseline':<9} | " + harness.fmt_metrics(base_concat["baseline"]))
        log(f"{'':<16} | {'B&H':<9} | " + harness.fmt_metrics(base_concat["buy_and_hold"]))
    log()
    log("=== In-sample vs OOS par fold ===")
    log("fold | config | Sharpe IS (train) | Sharpe val int. | Sharpe OOS | AUC train | AUC val | AUC test")
    for r in fold_out:
        log(f"{r['fold']:>4} | {r['chosen_config']:>6} | {r['sharpe_in_sample']:>17.3f} | "
            f"{r['best_val_sharpe']:>15.3f} | {r['model']['sharpe']:>10.3f} | {r['auc_train']:>9.4f} | "
            f"{r['auc_val']:>7.4f} | {r['auc_test']:>8.4f}")
    npos = sum(1 for r in fold_out if isinstance(r["model"]["sharpe"], float) and r["model"]["sharpe"] > 0)
    d_base = d_bh = None
    if base_concat["baseline"] is not None:
        d_base = m_c["sharpe"] - base_concat["baseline"]["sharpe"]
        d_bh = m_c["sharpe"] - base_concat["buy_and_hold"]["sharpe"]
        log()
        log(f"Sharpe OOS concaténé modèle {m_c['sharpe']:.4f} ; baseline {base_concat['baseline']['sharpe']:.4f} "
            f"(écart {d_base:+.4f}) ; B&H {base_concat['buy_and_hold']['sharpe']:.4f} (écart {d_bh:+.4f}) ; "
            f"trades {m_c['trades']} ; folds à Sharpe > 0 : {npos}/9")
    else:
        log(f"Sharpe concat {m_c['sharpe']:.4f} trades {m_c['trades']} folds>0 {npos}/9")
    log(f"rendements journaliers OOS : {daily}")
    last_open = max(res_c.end_mark_time, *[pd.Timestamp(r["model"]["end_mark_time"]) for r in fold_out])
    log(f"[garde] dernier open lu par les backtests = {last_open} <= {harness.MAX_INDEX} : "
        f"{last_open <= harness.MAX_INDEX}")
    assert last_open <= harness.MAX_INDEX
    imp_mean = {fn: float(np.mean([r["importance_norm"][fn] for r in fold_out])) for fn in all_feats}
    top_feat = max(all_feats, key=lambda fn: imp_mean[fn])
    log("importances normalisées moyennes : " + ", ".join(
        f"{fn} {imp_mean[fn]:.4f}" for fn in sorted(all_feats, key=lambda q: -imp_mean[q])))
    log(f"feature la plus importante (règle E17.json) : {top_feat}")
    return {
        "id": cfg17["id"], "config": cfg17, "seed": seed, "shuffle_seed": shuffle_seed,
        "dataset": DEFAULT_DATASET.name,
        "dataset_sha256": None if skip_baseline else sha256_file(DEFAULT_DATASET),
        "index_max_read": df.index.max(), "cost_per_side": harness.COST_PER_SIDE,
        "exec_delay": harness.EXEC_DELAY, "cost_multiplier": harness.COST_MULTIPLIER,
        "union_ids": [c["id"] for c in union],
        "grid_size_total": sum(len(c["hps"]) * len(c["maps"]) for c in union),
        "baseline_equality": base_diffs,
        "folds": [{**r, "baseline": b["baseline"], "buy_and_hold": b["buy_and_hold"]}
                  for r, b in zip(fold_out, base_rows)],
        "concatenated": {"model": m_c, "baseline": base_concat["baseline"],
                         "buy_and_hold": base_concat["buy_and_hold"]},
        "delta_sharpe_vs_baseline": d_base, "delta_sharpe_vs_bh": d_bh,
        "folds_sharpe_positive": npos, "daily_oos": daily,
        "importance_norm_mean": imp_mean, "most_important_feature": top_feat,
        "_signal": sig,
    }


def registry_line(out: dict, cmd: str, date: str) -> str:
    fs = ";".join(f"{r['model']['sharpe']:.3f}" for r in out["folds"])
    ft = ";".join(str(r["model"]["trades"]) for r in out["folds"])
    ch = ";".join(r["chosen_config"] for r in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = ("éligible (> baseline ; candidat désigné dans le rapport P3-R3)" if eligible
              else "abandonné (<= baseline)")
    return (f"| E17 | {date} | {out['commit'][:7]} | seed={out['seed']} | emboîté (union lgbm/hgb de E01-E16) | "
            f"features=celles de la config choisie par fold | H=par fold ({ch}) | grille hp et mapping = union des "
            f"16 configs E01-E16 (blobs git vérifiés, experiments/configs/E17.json) ; validation commune H=24 | "
            f"grille interne {out['grid_size_total']} couples par fold | folds 1-9 (make_folds H=24 pour la "
            f"validation, H choisi pour le train final) | Sharpe/fold {fs} | trades/fold {ft} | Sharpe concat "
            f"{m['sharpe']:.4f} | trades concat {m['trades']} | écart baseline {out['delta_sharpe_vs_baseline']:+.4f} | "
            f"écart B&H {out['delta_sharpe_vs_bh']:+.4f} | {statut} | `{cmd}` |")


# ------------------------------------------------------------------ données synthétiques (auto-test)

def synthetic_df(seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(DEV_START, DEV_END, freq="h")
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.01, len(idx))
    c = 10000 * np.exp(np.cumsum(lr))
    o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, len(idx))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, len(idx))))
    v = np.exp(rng.normal(5, 1, len(idx)))
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)
    df.iloc[30000:30003] = np.nan  # trou
    return df


def apply_variant(var: str, union: list[dict]) -> None:
    """Mêmes perturbations que tests/adversarial/run_variant.py, appliquées à la procédure emboîtée."""
    if var in ("base",) or var.startswith("shuf"):
        return
    if var == "delay2":
        harness.EXEC_DELAY = 2
    elif var == "featlag1":
        orig = harness.build_matrix

        def lagged(df, feats):
            return orig(df, feats).shift(1)
        harness.build_matrix = lagged
    elif var in ("cost2", "cost3"):
        harness.COST_MULTIPLIER = float(var[4:])
    elif var.startswith("drop_"):
        feat = var[5:]
        hit = 0
        for c in union:
            if feat in c["features"]:
                c["features"] = [x for x in c["features"] if x != feat]
                hit += 1
        assert hit > 0, f"feature {feat} absente de l'union"
    else:
        raise SystemExit(f"variante inconnue {var}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--registry", action="store_true")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-configs", default="E01,E05,E07,E08,E11")
    args = ap.parse_args()
    log = harness.Tee()
    commit = harness.git("rev-parse", "HEAD")
    dirty = harness.git("status", "--porcelain", "--", "src", "experiments/harness.py", "experiments/nested.py",
                        "experiments/extra_features.py", "experiments/configs") != ""
    log(f"commit {commit} ; code modifié non commité (src, harnais, nested, configs) : {dirty}")
    cfg17 = json.loads(args.config.read_text(encoding="utf-8"))
    union = load_union(cfg17, log)
    var = args.variant
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    apply_variant(var, union)
    log(f"variante : {var}")

    if args.selftest:
        keep = args.selftest_configs.split(",")
        union = [c for c in union if c["id"] in keep]
        log(f"AUTO-TEST sur données SYNTHÉTIQUES (aucune lecture du dataset), configs {keep}")
        out = run_nested(cfg17, union, shuffle_seed, log, df=synthetic_df(), skip_baseline=True)
        assert out["concatenated"]["model"]["trades"] >= 0
        log("auto-test : OK")
        return 0

    out = run_nested(cfg17, union, shuffle_seed, log)
    sig = out.pop("_signal")
    out = {"commit": commit, "code_dirty": dirty, "variant": var, **out}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = args.out or (RESULTS / ("E17.json" if var == "base" else f"E17_{var}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    log(f"écrit : {path}")
    if var == "base" and args.out is None:
        # signal complet au format attendu par tests/adversarial/analyse_e10.py (nom de fichier imposé)
        d = RESULTS / "adv_E17"
        d.mkdir(parents=True, exist_ok=True)
        sig.to_frame("signal").to_csv(d / "E10_adv_base_signal_full.csv")
        log(f"signal complet : {d / 'E10_adv_base_signal_full.csv'}")
    if args.registry:
        if dirty:
            raise RuntimeError("code non commité : ligne de registre refusée")
        if var != "base" or args.out is not None:
            raise RuntimeError("seule l'exécution de base de E17 est un essai du registre")
        cmd = ".venv\\Scripts\\python experiments\\nested.py experiments\\configs\\E17.json --registry"
        line = registry_line(out, cmd, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
        with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        log(f"registre : ligne ajoutée ({len(REGISTRE.read_text(encoding='utf-8').splitlines())} lignes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
