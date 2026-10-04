"""Brief P3-B4 : essais E18-E28 (labels nets de coûts / triple barrière), procédure emboîtée E28.

Usage :
    .venv\\Scripts\\python experiments\\b4.py experiments\\configs\\E18.json                 (essai, écrit results/E18.json)
    .venv\\Scripts\\python experiments\\b4.py experiments\\configs\\E18.json --variant shuf3  (labels mélangés, hors registre)
    .venv\\Scripts\\python experiments\\b4.py CONFIG --variant delay2|featlag1|cost2|cost3|drop_<feat>
    .venv\\Scripts\\python experiments\\b4.py CONFIG --out PATH                              (relance de reproductibilité)
    .venv\\Scripts\\python experiments\\b4.py --register E18                                  (ajoute LA ligne de E18 au registre)
    .venv\\Scripts\\python experiments\\b4.py --selftest                                      (données SYNTHÉTIQUES uniquement)

Les essais tournent en parallèle SANS toucher au registre ; les lignes sont ajoutées ensuite,
une par une, par un seul processus (--register), à partir du JSON de l'essai (refus si le code
du JSON n'était pas commité ou si l'ID est déjà au registre).

Procédure « single » (E18-E27), pour chaque fold k de make_folds(horizon=H) :
1. train = split_fold(df, fold, horizon=H) (purge H+1) ; lignes propres = features de la config
   et label non NaN ;
2. élagage éventuel (config "prune") : sur les lignes propres du train du fold (features seules,
   sans label, sans test), Spearman ; parcours des features dans l'ordre de la liste, une feature
   est retirée si |rho| > seuil avec une feature déjà gardée ;
3. validation interne = derniers 20 % des lignes propres, train interne = lignes antérieures
   purgées (purge_mask, horizon=H) ; pour chaque hp : fit, probas sur la fenêtre de validation ;
   mapping par QUANTILES : entrée si p > Q(q_entrée), sortie si p <= Q(q_entrée - écart) et détention
   >= m x H bougies, Q = quantiles des probas du modèle sur son train interne ; score = Sharpe net du backtest de la fenêtre (harness.window_sharpe :
   exec_delay=1, 0,0015/côté, x1), NaN -> -inf ; premier maximum dans l'ordre (hp, mapping) ;
4. réentraînement sur tout le train propre ; seuils = mêmes quantiles des probas du modèle final
   sur tout ce train ; probas sur
   t in [test_start - 1h, test_end - 1h] (fold 1 depuis test_start - 2h, fold 9 jusqu'à test_end).
Procédure « nested » (E28) : comme experiments/nested.py (validation commune H=24, purge du plus
grand H, choix sur la seule validation interne parmi l'union E18-E27, blobs git vérifiés), avec
pour chaque configuration son label, son élagage et son mapping par quantiles.
Évaluation : identique à harness.run_trial (probas concaténées, mapping continu, UN backtest,
métriques par fold / concaténées, baseline et B&H recalculés et contrôlés à 1e-9).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
import labels_b4  # noqa: E402
from bot.data import sha256_file  # noqa: E402
from bot.features import FEATURES  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.metrics import daily_returns, summarize  # noqa: E402
from bot.split import DEFAULT_DATASET, DEV_END, DEV_START, H1, make_folds, purge_mask, split_fold  # noqa: E402
from extra_features import EXTRA_FEATURES  # noqa: E402

ROOT = harness.ROOT
RESULTS = harness.RESULTS
REGISTRE = harness.REGISTRE
CONFIGS = ROOT / "experiments" / "configs"
H_COMMON = 24
ALL_FEATS = list(FEATURES) + list(EXTRA_FEATURES)
CODE_PATHS = ("src", "experiments/harness.py", "experiments/extra_features.py", "experiments/b4.py",
              "experiments/labels_b4.py", "experiments/configs")


# ------------------------------------------------------------------ configurations

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")  # core.autocrlf=true : blob git = contenu LF
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def prep(cfg: dict, ci: int = 0) -> dict:
    H = int(cfg["label"]["H"])
    m = cfg["mapping"]
    maps = [{"q_enter": float(q), "q_gap": float(g), "mh_mult": int(k)}
            for q, g, k in itertools.product(m["enter_quantile"], m["exit_quantile_gap"], m["min_hold_mult"])]
    return {"ci": ci, "id": cfg["id"], "model": cfg["model"], "H": H, "label": cfg["label"],
            "features": harness.resolve_features(cfg["features"]), "prune": cfg.get("prune"),
            "hps": harness.hp_grid(cfg), "maps": maps, "seed": int(cfg["seed"])}


def load_union(cfgN: dict, log, check_git: bool = True) -> list[dict]:
    out = []
    for i, ent in enumerate(cfgN["union"]):
        p = ROOT / ent["path"]
        local = git_blob_sha1(p)
        at_head = harness.git("rev-parse", f"HEAD:{ent['path']}") if check_git else ent["git_blob"]
        if not (local == ent["git_blob"] == at_head):
            raise RuntimeError(f"{ent['id']} : blob local {local} / HEAD {at_head} != déclaré {ent['git_blob']}")
        c = json.loads(p.read_text(encoding="utf-8"))
        assert c["id"] == ent["id"] and c.get("procedure", "single") == "single"
        out.append(prep(c, i))
    log(f"[union] {len(out)} configurations vérifiées (blob local = blob HEAD = déclaré) : "
        + ", ".join(f"{c['id']}({c['label']['kind']} H={c['H']}, {len(c['hps'])}x{len(c['maps'])})" for c in out))
    return out


def abs_map(mp: dict, p_fit: np.ndarray, H: int) -> dict:
    """Seuils = quantiles des probas du modèle sur SES lignes d'ajustement (train interne pour la
    validation, train complet pour le modèle final) : entrée = Q(q_enter), sortie = Q(q_enter - q_gap)."""
    p = p_fit[~np.isnan(p_fit)]
    return {"enter": float(np.quantile(p, mp["q_enter"])), "exit": float(np.quantile(p, mp["q_enter"] - mp["q_gap"])),
            "min_hold": mp["mh_mult"] * H}


def prune_features(Xtr: pd.DataFrame, feats: list[str], prune: dict | None) -> tuple[list[str], list]:
    if not prune:
        return list(feats), []
    corr = Xtr[feats].corr(method=prune["method"]).abs()
    kept, dropped = [], []
    for f in feats:
        hit = [(g, float(corr.loc[f, g])) for g in kept if corr.loc[f, g] > prune["thr"]]
        if hit:
            dropped.append((f, hit[0][0], hit[0][1]))
        else:
            kept.append(f)
    return kept, dropped


# ------------------------------------------------------------------ briques

def config_train(df, X, Y, c, k, shuffle_seed):
    fH = make_folds(horizon=c["H"])[k - 1]
    assert fH.k == k
    tr, _ = split_fold(df, fH, horizon=c["H"])
    Xa = X.loc[tr.index, c["features"]]
    ya = Y[c["id"]].loc[tr.index]
    clean = (~Xa.isna().any(axis=1)) & (~ya.isna())
    Xtr = Xa.loc[clean]
    ytr = ya.loc[clean].to_numpy().copy()
    if shuffle_seed is not None:
        ytr = np.random.default_rng([shuffle_seed, k, c["ci"]]).permutation(ytr)
    feats, dropped = prune_features(Xtr, c["features"], c["prune"])
    return fH, Xtr[feats], ytr, feats, dropped


def score_grid(c, seed, Xin, yin, Xv, vidx, open_v, val_start, val_end, Xvr, yvr):
    prior = float(np.mean(yin))
    best = None
    for hi, hp in enumerate(c["hps"]):
        mdl = harness.make_model(c["model"], hp, seed).fit(Xin.to_numpy(), yin)
        p_fit = harness.predict_proba(mdl, Xin)
        pv = harness.predict_proba(mdl, Xv)
        auc_v = harness.auc(yvr, harness.predict_proba(mdl, Xvr))
        for mi, mp in enumerate(c["maps"]):
            s, ntr = harness.window_sharpe(pv, vidx, abs_map(mp, p_fit, c["H"]), open_v, val_start, val_end)
            score = s if (isinstance(s, float) and math.isfinite(s)) else -math.inf
            if best is None or score > best["score"]:
                best = {"score": score, "hp_i": hi, "map_i": mi, "val_trades": ntr, "auc_val": auc_v,
                        "prior_inner": prior}
    return best


def final_fit(df, X, c, k, fH, Xtr, ytr, feats, best, seed):
    hp, mp_rel = c["hps"][best["hp_i"]], c["maps"][best["map_i"]]
    model = harness.make_model(c["model"], hp, seed).fit(Xtr.to_numpy(), ytr)
    prior = float(np.mean(ytr))
    p_fit = harness.predict_proba(model, Xtr)
    mp = abs_map(mp_rel, p_fit, c["H"])
    auc_tr = harness.auc(ytr, p_fit)
    t_first = Xtr.index[0]
    is_idx = df.index[(df.index >= t_first) & (df.index <= fH.train_end)]
    sh_is, ntr_is = harness.window_sharpe(harness.predict_proba(model, X.loc[is_idx, feats]), is_idx, mp,
                                          df["open"].loc[t_first: fH.train_end + H1], t_first, fH.train_end)
    g = model.booster_.feature_importance(importance_type="gain").astype(float)
    tot = g.sum()
    imp = {fn: 0.0 for fn in ALL_FEATS}
    for fn, v in zip(feats, g):
        imp[fn] = float(v / tot) if tot > 0 else 0.0
    return model, {"best_hp": hp, "best_map_rel": mp_rel, "best_map": mp, "prior_train": prior,
                   "auc_train": auc_tr, "sharpe_in_sample": sh_is, "trades_in_sample": ntr_is,
                   "importance_norm": imp}


def run_fold_single(f, df, X, Y, c, seed, shuffle_seed, log):
    k = f.k
    fH, Xtr, ytr, feats, dropped = config_train(df, X, Y, c, k, shuffle_seed)
    assert fH.train_end == f.train_end
    n = len(Xtr)
    v0 = int(math.floor((1.0 - harness.VAL_FRAC) * n))
    val_start, val_end = Xtr.index[v0], f.train_end
    inner = (np.arange(n) < v0) & purge_mask(Xtr.index, val_start, val_end, horizon=c["H"])
    vidx = df.index[(df.index >= val_start) & (df.index <= val_end)]
    open_v = df["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= f.test_start - H1
    best = score_grid(c, seed, Xtr.iloc[inner], ytr[inner], X.loc[vidx, feats], vidx, open_v, val_start,
                      val_end, Xtr.iloc[v0:], ytr[v0:])
    model, fin = final_fit(df, X, c, k, fH, Xtr, ytr, feats, best, seed)
    log(f"  fold {k}: train {Xtr.index[0]} -> {f.train_end} ({n} lignes propres, taux y=1 {np.mean(ytr):.4f}), "
        f"train interne {int(inner.sum())}, validation {val_start} -> {val_end} ({n - v0} lignes) ; "
        f"features {len(feats)}/{len(c['features'])}" + (f" (élaguées {dropped})" if dropped else "")
        + f" ; choix hp#{best['hp_i']} {fin['best_hp']} map {fin['best_map_rel']} -> {fin['best_map']} : "
        f"Sharpe val {best['score']:.3f} ({best['val_trades']} trades), AUC val {best['auc_val']:.4f}, "
        f"AUC train {fin['auc_train']:.4f}, Sharpe IS {fin['sharpe_in_sample']:.3f}")
    r = {"fold": k, "chosen_config": c["id"], "chosen_H": c["H"], "chosen_features": feats, "pruned": dropped,
         "train_start": Xtr.index[0], "train_end": f.train_end, "n_train": n, "n_inner_train": int(inner.sum()),
         "val_start": val_start, "val_end": val_end, "n_val": n - v0, "test_start": f.test_start,
         "test_end": f.test_end, "best_val_sharpe": best["score"], "best_val_trades": best["val_trades"],
         "auc_val": best["auc_val"], "prior_inner": best["prior_inner"], **fin}
    return r, model


def run_fold_nested(f24, df, X, Y, union, seed, shuffle_seed, log):
    k = f24.k
    tr24, _ = split_fold(df, f24, horizon=H_COMMON)
    Xc = X.loc[tr24.index, ALL_FEATS]
    yc = Y["__common__"].loc[tr24.index]
    cl = (~Xc.isna().any(axis=1)) & (~yc.isna())
    idx_c = tr24.index[cl.to_numpy()]
    n = len(idx_c)
    v0 = int(math.floor((1.0 - harness.VAL_FRAC) * n))
    val_start, val_end = idx_c[v0], f24.train_end
    vidx = df.index[(df.index >= val_start) & (df.index <= val_end)]
    open_v = df["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= f24.test_start - H1
    per_cfg, best, cache = [], None, {}
    for c in union:
        fH, Xtr, ytr, feats, dropped = config_train(df, X, Y, c, k, shuffle_seed)
        cache[c["id"]] = (fH, Xtr, ytr, feats, dropped)
        t = Xtr.index
        inner = np.asarray(t < val_start) & purge_mask(t, val_start, val_end, horizon=H_COMMON)
        assert not np.any(np.asarray(t[inner] + (1 + c["H"]) * H1 >= val_start))
        vr = np.asarray((t >= val_start) & (t <= val_end))
        cb = score_grid(c, seed, Xtr.iloc[inner], ytr[inner], X.loc[vidx, feats], vidx, open_v, val_start,
                        val_end, Xtr.iloc[vr], ytr[vr])
        per_cfg.append({"config": c["id"], "best_val_sharpe": cb["score"], "hp_i": cb["hp_i"], "map_i": cb["map_i"],
                        "val_trades": cb["val_trades"], "auc_val": cb["auc_val"], "n_inner_train": int(inner.sum()),
                        "n_val_rows": int(vr.sum())})
        if best is None or cb["score"] > best["score"]:
            best = {**cb, "cfg": c}
    c = best["cfg"]
    fH, Xtr, ytr, feats, dropped = cache[c["id"]]
    model, fin = final_fit(df, X, c, k, fH, Xtr, ytr, feats, best, seed)
    top = sorted(per_cfg, key=lambda r: -r["best_val_sharpe"])[:5]
    log(f"  fold {k}: validation commune {val_start} -> {val_end} ({n - v0} lignes propres H=24) ; CHOIX {c['id']} "
        f"(H={c['H']}) hp#{best['hp_i']} {fin['best_hp']} map {fin['best_map_rel']} -> {fin['best_map']} : Sharpe "
        f"val {best['score']:.4f} ({best['val_trades']} trades), AUC val {best['auc_val']:.4f} ; train final "
        f"{Xtr.index[0]} -> {fH.train_end} ({len(Xtr)} lignes, taux y=1 {np.mean(ytr):.4f}), AUC train "
        f"{fin['auc_train']:.4f}, Sharpe IS {fin['sharpe_in_sample']:.3f}")
    log("    top 5 (score de validation) : " + ", ".join(f"{r['config']} {r['best_val_sharpe']:.4f}" for r in top))
    r = {"fold": k, "chosen_config": c["id"], "chosen_H": c["H"], "chosen_features": feats, "pruned": dropped,
         "train_start": Xtr.index[0], "train_end": fH.train_end, "n_train": len(Xtr), "val_start": val_start,
         "val_end": val_end, "n_val_common": n - v0, "test_start": f24.test_start, "test_end": f24.test_end,
         "best_val_sharpe": best["score"], "best_val_trades": best["val_trades"], "auc_val": best["auc_val"],
         "prior_inner": best["prior_inner"], "per_config_best": per_cfg, **fin}
    return r, model


# ------------------------------------------------------------------ essai

def run(cfg: dict, configs: list[dict], nested: bool, shuffle_seed, log, df=None, skip_baseline=False) -> dict:
    seed = int(cfg["seed"])
    if df is None:
        df = harness.load_dev(log)
    assert df.index.max() <= harness.MAX_INDEX
    log(f"essai {cfg['id']} : {'procédure emboîtée sur ' + ','.join(c['id'] for c in configs) if nested else 'single'}"
        f", seed {seed}" + (f", LABELS MÉLANGÉS seed {shuffle_seed}" if shuffle_seed is not None else "")
        + f" ; exec_delay {harness.EXEC_DELAY}, coût {harness.COST_PER_SIDE} x{harness.COST_MULTIPLIER}")
    if not nested:
        c = configs[0]
        log(f"  label {json.dumps(c['label'])} ; features {len(c['features'])} ; élagage {c['prune']} ; "
            f"grille {len(c['hps'])} hp x {len(c['maps'])} mappings = {len(c['hps']) * len(c['maps'])}")
    if skip_baseline:
        base_rows = [{"baseline": None, "buy_and_hold": None} for _ in range(9)]
        base_concat, base_diffs = {"baseline": None, "buy_and_hold": None}, {}
    else:
        base_rows, base_concat, base_diffs = harness.baseline_block(df, make_folds(), log)
    X = harness.build_matrix(df, ALL_FEATS)
    Y = {c["id"]: labels_b4.make_label(df, c["label"]) for c in configs}
    Y["__common__"] = compute_labels(df, horizon=H_COMMON)["fwd_ret"]
    assert X.index.max() <= harness.MAX_INDEX and all(v.index.max() <= harness.MAX_INDEX for v in Y.values())
    folds = make_folds(horizon=H_COMMON if nested else configs[0]["H"])

    n = len(df)
    p_all = np.full(n, np.nan)
    enter, exit_, mh = np.full(n, np.inf), np.full(n, -np.inf), np.zeros(n, dtype=np.int64)
    fold_out = []
    for f in folds:
        if nested:
            r, model = run_fold_nested(f, df, X, Y, configs, seed, shuffle_seed, log)
        else:
            r, model = run_fold_single(f, df, X, Y, configs[0], seed, shuffle_seed, log)
        a = f.test_start - (2 if f.k == 1 else 1) * H1
        b = f.test_end if f.k == len(folds) else f.test_end - H1
        cm = (df.index >= a) & (df.index <= b)
        p_all[cm] = harness.predict_proba(model, X.loc[cm, r["chosen_features"]])
        enter[cm], exit_[cm], mh[cm] = r["best_map"]["enter"], r["best_map"]["exit"], r["best_map"]["min_hold"]
        tm = (df.index >= f.test_start) & (df.index <= f.test_end)
        yt = Y[r["chosen_config"]].loc[tm].to_numpy()
        r["auc_test"] = harness.auc(yt, p_all[tm])
        r["label_rate_test"] = float(np.nanmean(yt))
        r["label_rate_train"] = r["prior_train"] if shuffle_seed is None else None
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
    log("=== Dates des folds et taux de label = 1 (train propre du fold / test) ===")
    for r in fold_out:
        log(f"fold {r['fold']} : train {r['train_start']} -> {r['train_end']} | validation {r['val_start']} -> "
            f"{r['val_end']} | test {r['test_start']} -> {r['test_end']} | config {r['chosen_config']} H={r['chosen_H']} | "
            f"taux y=1 train {r['prior_train']:.4f} test {r['label_rate_test']:.4f}")
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
        log(f"{r['fold']:>4} | {r['chosen_config']:>6} | {r['sharpe_in_sample']:>17.3f} | {r['best_val_sharpe']:>15.3f} | "
            f"{r['model']['sharpe']:>10.3f} | {r['auc_train']:>9.4f} | {r['auc_val']:>7.4f} | {r['auc_test']:>8.4f}")
    npos = sum(1 for r in fold_out if isinstance(r["model"]["sharpe"], float) and r["model"]["sharpe"] > 0)
    d_base = d_bh = None
    if base_concat["baseline"] is not None:
        d_base = m_c["sharpe"] - base_concat["baseline"]["sharpe"]
        d_bh = m_c["sharpe"] - base_concat["buy_and_hold"]["sharpe"]
        log()
        log(f"Sharpe OOS concaténé modèle {m_c['sharpe']:.4f} ; baseline {base_concat['baseline']['sharpe']:.4f} "
            f"(écart {d_base:+.4f}) ; B&H {base_concat['buy_and_hold']['sharpe']:.4f} (écart {d_bh:+.4f}) ; "
            f"trades {m_c['trades']} ; expo {m_c['exposure']:.4f} ; folds à Sharpe > 0 : {npos}/9")
    else:
        log(f"Sharpe concat {m_c['sharpe']:.4f} trades {m_c['trades']} folds>0 {npos}/9")
    log(f"rendements journaliers OOS : {daily}")
    last_open = max(res_c.end_mark_time, *[pd.Timestamp(r["model"]["end_mark_time"]) for r in fold_out])
    log(f"[garde] dernier open lu par les backtests = {last_open} <= {harness.MAX_INDEX} : {last_open <= harness.MAX_INDEX}")
    assert last_open <= harness.MAX_INDEX
    imp_mean = {fn: float(np.mean([r["importance_norm"][fn] for r in fold_out])) for fn in ALL_FEATS}
    top_feat = max(ALL_FEATS, key=lambda fn: imp_mean[fn])
    log("importances (gain normalisé) moyennes : " + ", ".join(
        f"{fn} {imp_mean[fn]:.4f}" for fn in sorted(ALL_FEATS, key=lambda q: -imp_mean[q]) if imp_mean[fn] > 0))
    log(f"feature la plus importante : {top_feat}")
    return {
        "id": cfg["id"], "config": cfg, "seed": seed, "shuffle_seed": shuffle_seed,
        "dataset": DEFAULT_DATASET.name, "dataset_sha256": None if skip_baseline else sha256_file(DEFAULT_DATASET),
        "index_max_read": df.index.max(), "cost_per_side": harness.COST_PER_SIDE,
        "exec_delay": harness.EXEC_DELAY, "cost_multiplier": harness.COST_MULTIPLIER,
        "union_ids": [c["id"] for c in configs],
        "grid_size_total": sum(len(c["hps"]) * len(c["maps"]) for c in configs),
        "baseline_equality": base_diffs,
        "folds": [{**r, "baseline": b["baseline"], "buy_and_hold": b["buy_and_hold"]} for r, b in zip(fold_out, base_rows)],
        "concatenated": {"model": m_c, "baseline": base_concat["baseline"], "buy_and_hold": base_concat["buy_and_hold"]},
        "delta_sharpe_vs_baseline": d_base, "delta_sharpe_vs_bh": d_bh,
        "folds_sharpe_positive": npos, "daily_oos": daily,
        "importance_norm_mean": imp_mean, "most_important_feature": top_feat,
        "_signal": sig,
    }


# ------------------------------------------------------------------ variantes adversariales

def apply_variant(var: str, configs: list[dict]) -> None:
    if var == "base" or var.startswith("shuf"):
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
        for c in configs:
            if feat in c["features"]:
                c["features"] = [x for x in c["features"] if x != feat]
                hit += 1
        assert hit > 0, f"feature {feat} absente"
    else:
        raise SystemExit(f"variante inconnue {var}")


# ------------------------------------------------------------------ registre

def registry_line(out: dict, date: str) -> str:
    cfg = out["config"]
    fs = ";".join(f"{r['model']['sharpe']:.3f}" for r in out["folds"])
    ft = ";".join(str(r["model"]["trades"]) for r in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = ("éligible (> baseline ; candidat désigné dans le rapport P3-R4)" if eligible
              else "abandonné (<= baseline)")
    run_cmd = f".venv\\Scripts\\python experiments\\b4.py experiments\\configs\\{cfg['id']}.json"
    reg_cmd = f".venv\\Scripts\\python experiments\\b4.py --register {cfg['id']}"
    if cfg.get("procedure") == "nested":
        ch = ";".join(r["chosen_config"] for r in out["folds"])
        desc = (f"emboîté lgbm (union {','.join(out['union_ids'])}) | features=celles de la config choisie par fold | "
                f"label/H=par fold ({ch}) | grille hp et mapping = union des configs (blobs git vérifiés, "
                f"experiments/configs/{cfg['id']}.json) ; validation commune H=24 | grille interne "
                f"{out['grid_size_total']} couples par fold | folds 1-9 (make_folds H=24 pour la validation, H choisi "
                f"pour le train final)")
    else:
        lab = cfg["label"]
        ltxt = (f"net de coûts H={lab['H']}" if lab["kind"] == "net" else
                f"triple barrière H={lab['H']} haut={json.dumps(lab['up'], separators=(',', ':'))} "
                f"bas={json.dumps(lab['down'], separators=(',', ':'))}")
        feats = cfg["features"] if isinstance(cfg["features"], str) else ",".join(cfg["features"])
        if cfg.get("prune"):
            feats += f" élagage {cfg['prune']['method']} |rho|>{cfg['prune']['thr']}"
        g = len(harness.hp_grid(cfg))
        mp = json.dumps(cfg["mapping"], separators=(",", ":"))
        desc = (f"{cfg['model']} | features={feats} | label {ltxt} | grille hp="
                f"{json.dumps(cfg['grid'], separators=(',', ':'))} mapping quantiles={mp} | grille interne "
                f"{g}x{out['grid_size_total'] // g}={out['grid_size_total']} | folds 1-9 (make_folds H={lab['H']})")
    return (f"| {cfg['id']} | {date} | {out['commit'][:7]} | seed={out['seed']} | {desc} | Sharpe/fold {fs} | "
            f"trades/fold {ft} | Sharpe concat {m['sharpe']:.4f} | trades concat {m['trades']} | "
            f"écart baseline {out['delta_sharpe_vs_baseline']:+.4f} | écart B&H {out['delta_sharpe_vs_bh']:+.4f} | "
            f"{statut} | `{run_cmd}` puis `{reg_cmd}` |")


def register(tid: str) -> int:
    out = json.loads((RESULTS / f"{tid}.json").read_text(encoding="utf-8"))
    if out["code_dirty"] or out["variant"] != "base" or out["shuffle_seed"] is not None:
        raise RuntimeError("JSON non éligible au registre (code non commité ou variante)")
    lines = REGISTRE.read_text(encoding="utf-8").splitlines()
    if any(ln.split("|")[1].strip() == tid for ln in lines):
        raise RuntimeError(f"{tid} déjà au registre")
    line = registry_line(out, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(f"registre : ligne {tid} ajoutée ({len(lines) + 1} lignes)")
    return 0


# ------------------------------------------------------------------ autotest synthétique

def selftest(log) -> int:
    from nested import synthetic_df  # données synthétiques de P3-B3 (grille horaire dev complète, un trou)
    df = synthetic_df()
    log("AUTO-TEST sur données SYNTHÉTIQUES (aucune lecture du dataset)")
    specs = []
    cfgs = []
    for i in range(18, 28):
        cfg = json.loads((CONFIGS / f"E{i}.json").read_text(encoding="utf-8"))
        cfgs.append(cfg)
        if cfg["label"] not in specs:
            specs.append(cfg["label"])
    # anti-fuite sur une version à volatilité réduite (x0,3 en log) pour que les barrières fixes
    # ne soient pas toujours touchées avant H (sinon la sensibilité à t+1+H est nulle par construction)
    dl = df.iloc[:6000].copy()
    for col in ("open", "high", "low", "close"):
        dl[col] = 10000.0 * np.exp(0.3 * np.log(dl[col] / 10000.0))
    labels_b4.selftest(dl, specs, log)
    # exemple numérique triple barrière à la main
    idx = pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC")
    o = np.full(400, 100.0)
    o[301:306] = [100.0, 101.0, 102.5, 99.0, 98.0]  # entrée t+1=301 à 100, +2,5 % à j=2
    d = pd.DataFrame({"open": o, "high": o * 1.001, "low": o * 0.999, "close": o, "volume": 1.0}, index=idx)
    yt = labels_b4.triple_barrier_labels(d, 4, {"type": "fixed", "r": 0.02}, {"type": "fixed", "r": 0.02})
    yn = labels_b4.net_cost_labels(d, 4)
    log(f"[exemple] TB fixe 2 %/2 % H=4, t=300 : {yt.iloc[300]} (attendu 1, haut touché à j=2) ; "
        f"t=301 : {yt.iloc[301]} (attendu 0 : r=+1,5 %, -2 %, -3 % : bas touché à j=3) ; "
        f"net H=4 t=300 : {yn.iloc[300]} (r = log(98/100) < C -> 0)")
    assert yt.iloc[300] == 1.0 and yt.iloc[301] == 0.0 and yn.iloc[300] == 0.0
    # élagage
    Xs = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) ** 2, "c": np.sin(np.arange(100.0))})
    kept, dropped = prune_features(Xs, ["a", "b", "c"], {"method": "spearman", "thr": 0.9})
    log(f"[élagage] a, b=a² (rho 1), c : gardées {kept}, retirées {dropped}")
    assert kept == ["a", "c"]
    # essais réduits (grille tronquée) sur données synthétiques : single puis emboîté
    small = []
    for cfg in cfgs:
        c = prep(cfg, len(small))
        c["hps"], c["maps"] = c["hps"][:1], c["maps"][:2]
        small.append(c)
    for c in (small[0], small[6]):
        out = run({"id": "selftest", "seed": 42}, [c], False, None, log, df=df, skip_baseline=True)
        assert out["concatenated"]["model"]["trades"] >= 0
    out = run({"id": "selftest", "seed": 42}, small[:3] + small[6:7], True, 1, log, df=df, skip_baseline=True)
    log("auto-test : OK")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--register", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    log = harness.Tee()
    if args.register:
        return register(args.register)
    commit = harness.git("rev-parse", "HEAD")
    dirty = harness.git("status", "--porcelain", "--", *CODE_PATHS) != ""
    log(f"commit {commit} ; code modifié non commité ({', '.join(CODE_PATHS)}) : {dirty}")
    if args.selftest:
        return selftest(log)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    nested = cfg.get("procedure") == "nested"
    configs = load_union(cfg, log) if nested else [prep(cfg, 0)]
    var = args.variant
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    apply_variant(var, configs)
    log(f"variante : {var}")
    out = run(cfg, configs, nested, shuffle_seed, log)
    sig = out.pop("_signal")
    out = {"commit": commit, "code_dirty": dirty, "variant": var, **out}
    path = args.out or (RESULTS / (f"{cfg['id']}.json" if var == "base" else f"{cfg['id']}_{var}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    log(f"écrit : {path}")
    if var == "base" and args.out is None:
        d = RESULTS / f"adv_{cfg['id']}"
        d.mkdir(parents=True, exist_ok=True)
        sig.to_frame("signal").to_csv(d / "E10_adv_base_signal_full.csv")  # nom attendu par analyse_e10.py
        log(f"signal complet : {d / 'E10_adv_base_signal_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
