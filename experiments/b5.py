"""Brief P3-B5 : essais E29-E39 sur bougies 4 h (agrégées depuis les bougies 1 h), procédure emboîtée E39.

Usage :
    .venv\\Scripts\\python experiments\\b5.py experiments\\configs\\E29.json                 (essai -> results/E29.json)
    .venv\\Scripts\\python experiments\\b5.py experiments\\configs\\E29.json --variant shuf3  (labels mélangés, hors registre)
    .venv\\Scripts\\python experiments\\b5.py CONFIG --variant delay2|featlag1|cost2|cost3|drop_<f>|tbfixed|tbdir|
                                                       rule_<f>|nodow|dowplac<S>          (contrôles, hors registre)
    .venv\\Scripts\\python experiments\\b5.py CONFIG --out PATH                              (relance de reproductibilité)
    .venv\\Scripts\\python experiments\\b5.py --register E29                                  (ajoute LA ligne de E29)
    .venv\\Scripts\\python experiments\\b5.py --selftest                                      (données SYNTHÉTIQUES)

Données : harness.load_dev() = bot.split.load_dataset() par défaut + garde « index max <= 2025-09-30 23:00 UTC ».
Bougies 4 h, features, labels, projection : experiments/agg4h.py (docstring).

Procédure « single », pour chaque fold k de bot.split.make_folds(horizon=4H+3) (H en bougies 4 h ; mêmes
dates de test que tous les briefs ; purge = H+1 bougies 4 h, voir agg4h.horizon_hours) :
1. train = bot.split.split_fold(df4, fold, horizon=4H+3) sur la grille 4 h ; lignes propres = features de la
   config et label non NaN ;
2. validation interne = derniers 20 % des lignes propres (val_start = ligne floor(0,8 n)), val_end =
   fold.train_end ; train interne = lignes antérieures purgées (bot.split.purge_mask, horizon=4H+3) ;
3. pour chaque hp : fit sur le train interne ; probas sur les bougies 4 h de [val_start, val_end] ; mapping
   par QUANTILES des probas du modèle sur son train interne (b4.abs_map : entrée si p > Q(q), sortie si
   p <= Q(q - écart) et détention >= m x H bougies 4 h) ; positions 4 h projetées sur la grille 1 h
   (agg4h.project_to_1h) ; score = Sharpe journalier (bot.metrics.sharpe_daily) du backtest bot.backtest
   (exec_delay=1 h, soit 1 bougie 4 h après la clôture ; 0,0015/côté ; x1) de la fenêtre 1 h
   [val_start, val_end], démarrage à plat ; NaN -> -inf ; premier maximum dans l'ordre (hp, mapping) ;
4. réentraînement sur tout le train propre du fold ; seuils = mêmes quantiles des probas du modèle final
   sur ce train ; probas sur les bougies 4 h T in [test_start - 4h, test_end - 7h] (décision tenue à
   partir de T+4h dans le test ; fold 1 depuis test_start - 8h pour hériter la position avant
   2021-01-01 comme la baseline ; fold 9 jusqu'à test_end - 3h).
Procédure « nested » (E39) : validation commune définie avec H=30 (horizon 123 h, le plus grand H) :
train de référence = split_fold(df4, make_folds(horizon=123)[k], horizon=123), lignes propres = features f4
et fwd_ret H=30 non NaN, val_start = ligne floor(0,8 n), val_end = train_end (H=30) ; train interne de chaque
configuration c = ses lignes propres t < val_start et purge_mask(val_start, val_end, horizon=123) ; choix =
premier maximum du score dans l'ordre (configuration E29..E38, hp, mapping) ; réentraînement de la
configuration choisie sur tout son train du fold (son H). Union vérifiée par blob git.
Évaluation : probas 4 h des 9 folds concaténées, mapping continu (état conservé entre folds, seuils du fold
propriétaire), projection 1 h, UN backtest continu bot.backtest ; métriques par fold sur [test_start,
test_end] et concaténées 2021-01-01 -> 2025-09-30 23:00 (bot.metrics.summarize, Sharpe journalier x sqrt(365)) ;
baseline SMA168 et B&H recalculés sur les mêmes folds 1 h (harness.baseline_block, contrôle <= 1e-9 avec
results/baseline_metrics.json).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg4h  # noqa: E402
import harness  # noqa: E402
from b4 import abs_map, git_blob_sha1  # noqa: E402
from bot.backtest import run_backtest  # noqa: E402
from bot.data import sha256_file  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.metrics import daily_returns, sharpe_daily, summarize  # noqa: E402
from bot.split import DEFAULT_DATASET, H1, make_folds, purge_mask, split_fold  # noqa: E402

ROOT = harness.ROOT
RESULTS = harness.RESULTS
REGISTRE = harness.REGISTRE
CONFIGS = ROOT / "experiments" / "configs"
H_COMMON = 30  # bougies 4 h
CODE_PATHS = ("src", "experiments/harness.py", "experiments/b4.py", "experiments/agg4h.py", "experiments/b5.py",
              "experiments/configs")
COST_PER_SIDE = harness.COST_PER_SIDE  # 0,0015
EXEC_DELAY_H = 1  # heures sur la grille 1 h projetée = exécution à l'open de la bougie 4 h suivante
COST_MULTIPLIER = 1.0


# ------------------------------------------------------------------ configurations

def resolve_features(spec) -> list[str]:
    if spec == "f4":
        return list(agg4h.F4)
    if spec == "f4cal":
        return list(agg4h.ALL4)
    return list(spec)


def prep(cfg: dict, ci: int = 0) -> dict:
    H = int(cfg["label"]["H"])
    m = cfg["mapping"]
    maps = [{"q_enter": float(q), "q_gap": float(g), "mh_mult": int(k)}
            for q, g, k in itertools.product(m["enter_quantile"], m["exit_quantile_gap"], m["min_hold_mult"])]
    return {"ci": ci, "id": cfg["id"], "model": cfg["model"], "H": H, "Hh": agg4h.horizon_hours(H),
            "label": copy.deepcopy(cfg["label"]), "features": resolve_features(cfg["features"]),
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


class RuleModel:
    """Règle sans apprentissage : score = signe x valeur de la feature unique (contrôle « rule_<f> »)."""

    def __init__(self, sign: float):
        self.sign = float(sign)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = self.sign * np.asarray(X, dtype=np.float64)[:, 0]
        return np.column_stack([-s, s])


def make_model(kind: str, hp: dict, seed: int):
    if kind == "rule":
        return RuleModel(hp["sign"])
    return harness.make_model(kind, hp, seed)


# ------------------------------------------------------------------ briques

def window_sharpe4(p4: np.ndarray, idx4: pd.DatetimeIndex, mp: dict, open1: pd.Series, start, end):
    n = len(p4)
    sig4 = harness.map_positions(p4, np.full(n, mp["enter"]), np.full(n, mp["exit"]),
                                 np.full(n, mp["min_hold"], dtype=np.int64))
    sig1 = agg4h.project_to_1h(sig4, idx4, open1.index)
    res = run_backtest(sig1, open1, start=start, end=end, exec_delay=EXEC_DELAY_H, cost_per_side=COST_PER_SIDE,
                       cost_multiplier=COST_MULTIPLIER)
    return sharpe_daily(res.equity), int(len(res.trades))


def config_train(df4, X, Y, c, k, shuffle_seed):
    fH = make_folds(horizon=c["Hh"])[k - 1]
    assert fH.k == k
    tr, _ = split_fold(df4, fH, horizon=c["Hh"])
    Xa = X.loc[tr.index, c["features"]]
    ya = Y[c["id"]].loc[tr.index]
    clean = (~Xa.isna().any(axis=1)) & (~ya.isna())
    Xtr = Xa.loc[clean]
    ytr = ya.loc[clean].to_numpy().copy()
    # anti-fuite : label du dernier t du train = open4[t+1+H] < test_start
    assert Xtr.index.max() + 4 * (c["H"] + 1) * H1 < fH.test_start
    if shuffle_seed is not None:
        ytr = np.random.default_rng([shuffle_seed, k, c["ci"]]).permutation(ytr)
    return fH, Xtr, ytr


def score_grid(c, seed, Xin, yin, Xv, vidx4, open_v, val_start, val_end, Xvr, yvr):
    prior = float(np.mean(yin))
    best = None
    for hi, hp in enumerate(c["hps"]):
        mdl = make_model(c["model"], hp, seed).fit(Xin.to_numpy(), yin)
        p_fit = harness.predict_proba(mdl, Xin)
        pv = harness.predict_proba(mdl, Xv)
        auc_v = harness.auc(yvr, harness.predict_proba(mdl, Xvr))
        for mi, mp in enumerate(c["maps"]):
            s, ntr = window_sharpe4(pv, vidx4, abs_map(mp, p_fit, c["H"]), open_v, val_start, val_end)
            score = s if (isinstance(s, float) and math.isfinite(s)) else -math.inf
            if best is None or score > best["score"]:
                best = {"score": score, "hp_i": hi, "map_i": mi, "val_trades": ntr, "auc_val": auc_v,
                        "prior_inner": prior}
    return best


def final_fit(df1, df4, X, c, fH, Xtr, ytr, best, seed, all_feats):
    hp, mp_rel = c["hps"][best["hp_i"]], c["maps"][best["map_i"]]
    model = make_model(c["model"], hp, seed).fit(Xtr.to_numpy(), ytr)
    p_fit = harness.predict_proba(model, Xtr)
    mp = abs_map(mp_rel, p_fit, c["H"])
    auc_tr = harness.auc(ytr, p_fit)
    t_first = Xtr.index[0]
    is_idx = df4.index[(df4.index >= t_first) & (df4.index <= fH.train_end)]
    sh_is, ntr_is = window_sharpe4(harness.predict_proba(model, X.loc[is_idx, c["features"]]), is_idx, mp,
                                   df1["open"].loc[t_first: fH.train_end + H1], t_first, fH.train_end)
    imp = {fn: 0.0 for fn in all_feats}
    if c["model"] == "lgbm":
        g = model.booster_.feature_importance(importance_type="gain").astype(float)
        tot = g.sum()
        for fn, v in zip(c["features"], g):
            imp[fn] = float(v / tot) if tot > 0 else 0.0
    return model, {"best_hp": hp, "best_map_rel": mp_rel, "best_map": mp, "prior_train": float(np.mean(ytr)),
                   "auc_train": auc_tr, "sharpe_in_sample": sh_is, "trades_in_sample": ntr_is,
                   "importance_norm": imp}


def run_fold_single(f, df1, df4, X, Y, c, seed, shuffle_seed, log, all_feats):
    k = f.k
    fH, Xtr, ytr = config_train(df4, X, Y, c, k, shuffle_seed)
    n = len(Xtr)
    v0 = int(math.floor((1.0 - harness.VAL_FRAC) * n))
    val_start, val_end = Xtr.index[v0], fH.train_end
    inner = (np.arange(n) < v0) & purge_mask(Xtr.index, val_start, val_end, horizon=c["Hh"])
    assert not np.any(np.asarray(Xtr.index[inner] + 4 * (c["H"] + 1) * H1 >= val_start))
    vidx4 = df4.index[(df4.index >= val_start) & (df4.index <= val_end)]
    open_v = df1["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= fH.test_start - H1
    best = score_grid(c, seed, Xtr.iloc[inner], ytr[inner], X.loc[vidx4, c["features"]], vidx4, open_v,
                      val_start, val_end, Xtr.iloc[v0:], ytr[v0:])
    model, fin = final_fit(df1, df4, X, c, fH, Xtr, ytr, best, seed, all_feats)
    log(f"  fold {k}: train {Xtr.index[0]} -> {Xtr.index[-1]} (train_end {fH.train_end}, {n} bougies 4 h propres, "
        f"taux y=1 {np.mean(ytr):.4f}), train interne {int(inner.sum())}, validation {val_start} -> {val_end} "
        f"({n - v0} lignes) ; choix hp#{best['hp_i']} {fin['best_hp']} map {fin['best_map_rel']} -> "
        f"{fin['best_map']} : Sharpe val {best['score']:.3f} ({best['val_trades']} trades), AUC val "
        f"{best['auc_val']:.4f}, AUC train {fin['auc_train']:.4f}, Sharpe IS {fin['sharpe_in_sample']:.3f}")
    r = {"fold": k, "chosen_config": c["id"], "chosen_H": c["H"], "chosen_features": c["features"],
         "train_start": Xtr.index[0], "train_last_row": Xtr.index[-1], "train_end": fH.train_end, "n_train": n,
         "n_inner_train": int(inner.sum()), "val_start": val_start, "val_end": val_end, "n_val": n - v0,
         "test_start": f.test_start, "test_end": f.test_end, "best_val_sharpe": best["score"],
         "best_val_trades": best["val_trades"], "auc_val": best["auc_val"], "prior_inner": best["prior_inner"], **fin}
    return r, model


def run_fold_nested(f, df1, df4, X, Y, union, seed, shuffle_seed, log, all_feats):
    k = f.k
    hc = agg4h.horizon_hours(H_COMMON)
    fc = make_folds(horizon=hc)[k - 1]
    trc, _ = split_fold(df4, fc, horizon=hc)
    Xc = X.loc[trc.index, agg4h.F4]
    yc = Y["__common__"].loc[trc.index]
    cl = (~Xc.isna().any(axis=1)) & (~yc.isna())
    idx_c = trc.index[cl.to_numpy()]
    n = len(idx_c)
    v0 = int(math.floor((1.0 - harness.VAL_FRAC) * n))
    val_start, val_end = idx_c[v0], fc.train_end
    vidx4 = df4.index[(df4.index >= val_start) & (df4.index <= val_end)]
    open_v = df1["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= fc.test_start - H1
    per_cfg, best, cache = [], None, {}
    for c in union:
        fH, Xtr, ytr = config_train(df4, X, Y, c, k, shuffle_seed)
        cache[c["id"]] = (fH, Xtr, ytr)
        t = Xtr.index
        inner = np.asarray(t < val_start) & purge_mask(t, val_start, val_end, horizon=hc)
        assert not np.any(np.asarray(t[inner] + 4 * (c["H"] + 1) * H1 >= val_start))
        vr = np.asarray((t >= val_start) & (t <= val_end))
        cb = score_grid(c, seed, Xtr.iloc[inner], ytr[inner], X.loc[vidx4, c["features"]], vidx4, open_v,
                        val_start, val_end, Xtr.iloc[vr], ytr[vr])
        per_cfg.append({"config": c["id"], "best_val_sharpe": cb["score"], "hp_i": cb["hp_i"], "map_i": cb["map_i"],
                        "val_trades": cb["val_trades"], "auc_val": cb["auc_val"], "n_inner_train": int(inner.sum()),
                        "n_val_rows": int(vr.sum())})
        if best is None or cb["score"] > best["score"]:
            best = {**cb, "cfg": c}
    c = best["cfg"]
    fH, Xtr, ytr = cache[c["id"]]
    model, fin = final_fit(df1, df4, X, c, fH, Xtr, ytr, best, seed, all_feats)
    top = sorted(per_cfg, key=lambda r: -r["best_val_sharpe"])[:5]
    log(f"  fold {k}: validation commune {val_start} -> {val_end} ({n - v0} lignes propres H=30) ; CHOIX {c['id']} "
        f"(H={c['H']}) hp#{best['hp_i']} {fin['best_hp']} map {fin['best_map_rel']} -> {fin['best_map']} : Sharpe "
        f"val {best['score']:.4f} ({best['val_trades']} trades), AUC val {best['auc_val']:.4f} ; train final "
        f"{Xtr.index[0]} -> {Xtr.index[-1]} ({len(Xtr)} lignes, taux y=1 {np.mean(ytr):.4f}), AUC train "
        f"{fin['auc_train']:.4f}, Sharpe IS {fin['sharpe_in_sample']:.3f}")
    log("    top 5 (score de validation) : " + ", ".join(f"{r['config']} {r['best_val_sharpe']:.4f}" for r in top))
    r = {"fold": k, "chosen_config": c["id"], "chosen_H": c["H"], "chosen_features": c["features"],
         "train_start": Xtr.index[0], "train_last_row": Xtr.index[-1], "train_end": fH.train_end,
         "n_train": len(Xtr), "val_start": val_start, "val_end": val_end, "n_val_common": n - v0,
         "test_start": f.test_start, "test_end": f.test_end, "best_val_sharpe": best["score"],
         "best_val_trades": best["val_trades"], "auc_val": best["auc_val"], "prior_inner": best["prior_inner"],
         "per_config_best": per_cfg, **fin}
    return r, model


# ------------------------------------------------------------------ données / variantes

def placebo_dow(idx4: pd.DatetimeIndex, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Jour de semaine aléatoire (uniforme 0..6) tiré par jour civil UTC, mêmes sin/cos."""
    days = idx4.tz_convert("UTC").floor("D")
    ud = days.unique()
    rng = np.random.default_rng(seed)
    dmap = pd.Series(rng.integers(0, 7, len(ud)).astype(np.float64), index=ud)
    d = dmap.reindex(days).to_numpy()
    return np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)


def build_inputs(df1: pd.DataFrame, configs: list[dict], var: str, log):
    df4 = agg4h.aggregate_4h(df1)
    F = agg4h.compute_features_4h(df4)  # features brutes (aussi échelle vol_42 des labels TB)
    X = F.copy()
    if var == "featlag1":
        X = X.shift(1)  # features décalées d'une bougie 4 h supplémentaire
    if var.startswith("dowplac"):
        s, cc = placebo_dow(df4.index, int(var[7:]))
        X["dow_sin"], X["dow_cos"] = s, cc
    Y = {c["id"]: agg4h.make_label_4h(df4, F, c["label"]) for c in configs}
    Y["__common__"] = compute_labels(df4, horizon=H_COMMON)["fwd_ret"]
    return df4, F, X, Y


def apply_variant(var: str, configs: list[dict]) -> None:
    global EXEC_DELAY_H, COST_MULTIPLIER
    if var == "base" or var.startswith("shuf") or var == "featlag1":
        return
    if var == "delay2":
        EXEC_DELAY_H = 5  # exécution à l'open de la 2e bougie 4 h après la clôture (t+2 en bougies 4 h)
    elif var in ("cost2", "cost3"):
        COST_MULTIPLIER = float(var[4:])
    elif var.startswith("drop_"):
        feat = var[5:]
        hit = 0
        for c in configs:
            if feat in c["features"]:
                c["features"] = [x for x in c["features"] if x != feat]
                hit += 1
        assert hit > 0, f"feature {feat} absente"
    elif var == "nodow":
        hit = 0
        for c in configs:
            if "dow_sin" in c["features"] or "dow_cos" in c["features"]:
                c["features"] = [x for x in c["features"] if x not in ("dow_sin", "dow_cos")]
                hit += 1
        if hit == 0:
            raise SystemExit("nodow sans objet : aucune configuration n'utilise dow_sin / dow_cos")
    elif var.startswith("dowplac"):
        for c in configs:  # placebo : remplace (ou ajoute) dow_sin / dow_cos
            c["features"] = [x for x in c["features"] if x not in ("dow_sin", "dow_cos")] + ["dow_sin", "dow_cos"]
    elif var == "tbfixed":
        hit = 0
        for c in configs:
            lab = c["label"]
            if lab["kind"] == "tb" and lab["up"]["type"] == "vol":
                v = agg4h.fixed_log_rule(c["H"])
                lab["up"] = {"type": "fixed_log", "v": round(float(lab["up"]["k"]) * v, 6)}
                lab["down"] = {"type": "fixed_log", "v": round(float(lab["down"]["k"]) * v, 6)}
                hit += 1
        assert hit > 0, "tbfixed : aucun label TB dimensionné par la volatilité"
    elif var == "tbdir":
        for c in configs:
            lab = c["label"]
            if lab["kind"] == "tb":
                lab["expire_nan"] = True
            elif lab["kind"] == "net":
                lab["kind"] = "dir"
    elif var.startswith("rule_"):
        feat = var[5:]
        assert len(configs) == 1, "rule_<f> : procédure single uniquement"
        c = configs[0]
        c["model"], c["features"], c["hps"] = "rule", [feat], [{"sign": 1.0}, {"sign": -1.0}]
    else:
        raise SystemExit(f"variante inconnue {var}")


# ------------------------------------------------------------------ essai

def run(cfg: dict, configs: list[dict], nested: bool, shuffle_seed, log, var="base", df1=None,
        skip_baseline=False) -> dict:
    seed = int(cfg["seed"])
    if df1 is None:
        df1 = harness.load_dev(log)
    assert df1.index.max() <= harness.MAX_INDEX
    log(f"essai {cfg['id']} : {'procédure emboîtée sur ' + ','.join(c['id'] for c in configs) if nested else 'single'}"
        f", seed {seed}" + (f", LABELS MÉLANGÉS seed {shuffle_seed}" if shuffle_seed is not None else "")
        + f" ; exec_delay {EXEC_DELAY_H} h sur la grille 1 h projetée, coût {COST_PER_SIDE} x{COST_MULTIPLIER}")
    if not nested:
        c = configs[0]
        log(f"  label {json.dumps(c['label'])} (H={c['H']} bougies 4 h, horizon split {c['Hh']} h) ; modèle {c['model']} ; "
            f"features {len(c['features'])} {c['features']} ; grille {len(c['hps'])} hp x {len(c['maps'])} mappings = "
            f"{len(c['hps']) * len(c['maps'])}")
    if skip_baseline:
        base_rows = [{"baseline": None, "buy_and_hold": None} for _ in range(9)]
        base_concat, base_diffs = {"baseline": None, "buy_and_hold": None}, {}
    else:
        base_rows, base_concat, base_diffs = harness.baseline_block(df1, make_folds(), log)
    df4, F, X, Y = build_inputs(df1, configs, var, log)
    all_feats = list(X.columns)
    assert df4.index.max() <= harness.MAX_INDEX and X.index.max() <= harness.MAX_INDEX
    n4, nm4 = len(df4), int(df4["missing"].sum())
    log(f"[4 h] {n4} bougies 4 h ({df4.index.min()} -> {df4.index.max()}), dont {nm4} missing ; heures missing "
        f"{int(agg4h.hourly_missing(df1).sum())}")
    folds = make_folds(horizon=agg4h.horizon_hours(H_COMMON if nested else configs[0]["H"]))

    n = len(df4)
    p_all = np.full(n, np.nan)
    enter, exit_, mh = np.full(n, np.inf), np.full(n, -np.inf), np.zeros(n, dtype=np.int64)
    fold_out = []
    for f in folds:
        if nested:
            r, model = run_fold_nested(f, df1, df4, X, Y, configs, seed, shuffle_seed, log, all_feats)
        else:
            r, model = run_fold_single(f, df1, df4, X, Y, configs[0], seed, shuffle_seed, log, all_feats)
        a = f.test_start - (8 if f.k == 1 else 4) * H1
        b = f.test_end - (3 if f.k == len(folds) else 7) * H1
        cm = (df4.index >= a) & (df4.index <= b)
        p_all[cm] = harness.predict_proba(model, X.loc[cm, r["chosen_features"]])
        enter[cm], exit_[cm], mh[cm] = r["best_map"]["enter"], r["best_map"]["exit"], r["best_map"]["min_hold"]
        tm = (df4.index >= f.test_start) & (df4.index <= f.test_end)
        yt = Y[r["chosen_config"]].loc[tm].to_numpy()
        r["auc_test"] = harness.auc(yt, p_all[tm])
        r["label_rate_test"] = float(np.nanmean(yt))
        r["n_test_4h"] = int(tm.sum())
        r["signal_cover_4h"] = [df4.index[cm][0], df4.index[cm][-1]]
        fold_out.append(r)
    assert np.isnan(p_all[df4.index < folds[0].test_start - 8 * H1]).all()

    sig4 = harness.map_positions(p_all, enter, exit_, mh)
    sig1 = agg4h.project_to_1h(sig4, df4.index, df1.index)
    assert sig1.loc[: harness.CONCAT_START - 6 * H1].isna().all()
    bt = dict(exec_delay=EXEC_DELAY_H, cost_per_side=COST_PER_SIDE, cost_multiplier=COST_MULTIPLIER)
    for f, r in zip(folds, fold_out):
        r["model"] = summarize(run_backtest(sig1, df1["open"], start=f.test_start, end=f.test_end, **bt))
    res_c = run_backtest(sig1, df1["open"], start=harness.CONCAT_START, end=harness.CONCAT_END, **bt)
    m_c = summarize(res_c)
    hold = res_c.position.to_numpy()
    chg = np.flatnonzero(np.diff(np.concatenate([[res_c.initial_position], hold])) != 0)
    off = int(np.sum((res_c.position.index[chg].hour % 4) != 0)) if len(chg) else 0
    log(f"[contrôle] changements de position {len(chg)} ; hors des opens 4 h (00/04/../20 h) : {off}"
        + (" (attendu 0 hors trous de prix)" if EXEC_DELAY_H % 4 == 1 else ""))
    dr = daily_returns(res_c.equity).to_numpy()
    from scipy.stats import kurtosis, skew
    daily = {"n_days": int(len(dr)), "mean": float(dr.mean()), "std": float(dr.std(ddof=1)),
             "sr_daily": float(dr.mean() / dr.std(ddof=1)), "skew": float(skew(dr, bias=False)),
             "kurtosis": float(kurtosis(dr, fisher=False, bias=False))}

    log()
    log("=== Dates des folds et taux de label = 1 (train propre du fold / test) ===")
    for r in fold_out:
        log(f"fold {r['fold']} : train {r['train_start']} -> {r['train_last_row']} (train_end {r['train_end']}) | "
            f"validation {r['val_start']} -> {r['val_end']} | test {r['test_start']} -> {r['test_end']} | config "
            f"{r['chosen_config']} H={r['chosen_H']} | taux y=1 train {r['prior_train']:.4f} test {r['label_rate_test']:.4f}")
    log()
    log("=== Par fold (test) : modèle / baseline SMA168 / buy & hold, mêmes folds 1 h, mêmes coûts ===")
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
    imp_mean = {fn: float(np.mean([r["importance_norm"][fn] for r in fold_out])) for fn in all_feats}
    top_feat = max(all_feats, key=lambda fn: imp_mean[fn])
    log("importances (gain normalisé) moyennes : " + ", ".join(
        f"{fn} {imp_mean[fn]:.4f}" for fn in sorted(all_feats, key=lambda q: -imp_mean[q]) if imp_mean[fn] > 0))
    log(f"feature la plus importante : {top_feat}")
    proba = pd.DataFrame({"p": p_all, "enter": enter, "exit": exit_, "min_hold": mh, "sig4": sig4}, index=df4.index)
    for cid in sorted({r["chosen_config"] for r in fold_out}):
        proba[f"y_{cid}"] = Y[cid].to_numpy()
    return {
        "id": cfg["id"], "config": cfg, "seed": seed, "shuffle_seed": shuffle_seed,
        "dataset": DEFAULT_DATASET.name, "dataset_sha256": None if skip_baseline else sha256_file(DEFAULT_DATASET),
        "index_max_read": df1.index.max(), "cost_per_side": COST_PER_SIDE, "exec_delay_hours": EXEC_DELAY_H,
        "cost_multiplier": COST_MULTIPLIER, "n_bars_4h": n4, "n_missing_4h": nm4,
        "union_ids": [c["id"] for c in configs], "effective_configs": [
            {"id": c["id"], "model": c["model"], "label": c["label"], "features": c["features"], "n_hp": len(c["hps"])}
            for c in configs],
        "grid_size_total": sum(len(c["hps"]) * len(c["maps"]) for c in configs),
        "baseline_equality": base_diffs,
        "folds": [{**r, "baseline": b["baseline"], "buy_and_hold": b["buy_and_hold"]} for r, b in zip(fold_out, base_rows)],
        "concatenated": {"model": m_c, "baseline": base_concat["baseline"], "buy_and_hold": base_concat["buy_and_hold"]},
        "delta_sharpe_vs_baseline": d_base, "delta_sharpe_vs_bh": d_bh,
        "folds_sharpe_positive": npos, "daily_oos": daily, "position_changes_off_4h_grid": off,
        "importance_norm_mean": imp_mean, "most_important_feature": top_feat,
        "_signal": sig1, "_proba": proba,
    }


# ------------------------------------------------------------------ registre

def registry_line(out: dict, date: str) -> str:
    cfg = out["config"]
    fs = ";".join(f"{r['model']['sharpe']:.3f}" for r in out["folds"])
    ft = ";".join(str(r["model"]["trades"]) for r in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = ("éligible (> baseline ; candidat désigné dans le rapport P3-R5)" if eligible
              else "abandonné (<= baseline)")
    run_cmd = f".venv\\Scripts\\python experiments\\b5.py experiments\\configs\\{cfg['id']}.json"
    reg_cmd = f".venv\\Scripts\\python experiments\\b5.py --register {cfg['id']}"
    if cfg.get("procedure") == "nested":
        ch = ";".join(r["chosen_config"] for r in out["folds"])
        desc = (f"emboîté lgbm bougies 4 h (union {','.join(out['union_ids'])}) | features=f4 (celles de la config "
                f"choisie) | label/H=par fold ({ch}) | grille hp et mapping = union des configs (blobs git vérifiés, "
                f"experiments/configs/{cfg['id']}.json) ; validation commune H=30 bougies 4 h | grille interne "
                f"{out['grid_size_total']} couples par fold | folds 1-9 (make_folds horizon 123 h pour la validation, "
                f"horizon 4H+3 h de la config choisie pour le train final)")
    else:
        lab = cfg["label"]
        if lab["kind"] == "net":
            ltxt = f"net de coûts H={lab['H']} bougies 4 h"
        else:
            ltxt = (f"triple barrière H={lab['H']} bougies 4 h haut={json.dumps(lab['up'], separators=(',', ':'))} "
                    f"bas={json.dumps(lab['down'], separators=(',', ':'))}"
                    + (" expiration=NaN (direction seule)" if lab.get("expire_nan") else ""))
        g = len(harness.hp_grid(cfg))
        mp = json.dumps(cfg["mapping"], separators=(",", ":"))
        desc = (f"{cfg['model']} bougies 4 h | features={cfg['features']} | label {ltxt} | grille hp="
                f"{json.dumps(cfg['grid'], separators=(',', ':'))} mapping quantiles={mp} | grille interne "
                f"{g}x{out['grid_size_total'] // g}={out['grid_size_total']} | folds 1-9 (make_folds horizon "
                f"{agg4h.horizon_hours(lab['H'])} h)")
    line = (f"| {cfg['id']} | {date} | {out['commit'][:7]} | seed={out['seed']} | {desc} | Sharpe/fold {fs} | "
            f"trades/fold {ft} | Sharpe concat {m['sharpe']:.4f} | trades concat {m['trades']} | "
            f"écart baseline {out['delta_sharpe_vs_baseline']:+.4f} | écart B&H {out['delta_sharpe_vs_bh']:+.4f} | "
            f"{statut} | `{run_cmd}` puis `{reg_cmd}` |")
    cells = line.split(" | ")
    return line, len(cells)


def register(tid: str) -> int:
    out = json.loads((RESULTS / f"{tid}.json").read_text(encoding="utf-8"))
    if out["code_dirty"] or out["variant"] != "base" or out["shuffle_seed"] is not None:
        raise RuntimeError("JSON non éligible au registre (code non commité ou variante)")
    lines = REGISTRE.read_text(encoding="utf-8").splitlines()
    if any(ln.split("|")[1].strip() == tid for ln in lines):
        raise RuntimeError(f"{tid} déjà au registre")
    if len(lines) >= 40:
        raise RuntimeError("budget P3-B5 : registre limité à 40 lignes")
    line, _ = registry_line(out, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    n_pipes = line.count("|")
    if n_pipes != 19:  # 18 cellules -> 19 séparateurs, aucun « | » dans le contenu
        raise RuntimeError(f"ligne de registre mal formée : {n_pipes} « | » (attendu 19)")
    with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(f"registre : ligne {tid} ajoutée ({len(lines) + 1} lignes)")
    return 0


# ------------------------------------------------------------------ autotest synthétique

def selftest(log) -> int:
    from nested import synthetic_df  # grille horaire dev complète synthétique (P3-B3), un trou
    log("AUTO-TEST sur données SYNTHÉTIQUES (aucune lecture du dataset)")
    agg4h.selftest(log)
    df1 = synthetic_df()
    ids = [f"E{i}" for i in range(29, 39)]
    cfgs = [json.loads((CONFIGS / f"{i}.json").read_text(encoding="utf-8")) for i in ids]
    small = []
    for cfg in cfgs:
        c = prep(cfg, len(small))
        c["hps"], c["maps"] = c["hps"][:1], c["maps"][:2]
        small.append(c)
    import time
    for c in (small[0], small[5]):
        t0 = time.time()
        out = run({"id": "selftest", "seed": 42}, [c], False, None, log, df1=df1, skip_baseline=True)
        assert out["position_changes_off_4h_grid"] == 0
        log(f"[selftest] {c['id']} réduit : {time.time() - t0:.1f} s")
    c = prep(cfgs[1], 0)
    t0 = time.time()
    out = run({"id": "selftest", "seed": 42}, [c], False, 3, log, df1=df1, skip_baseline=True)
    log(f"[selftest] {c['id']} grille COMPLÈTE ({len(c['hps'])}x{len(c['maps'])}), labels mélangés : "
        f"{time.time() - t0:.1f} s")
    for var in ("rule_vol_42", "tbfixed", "tbdir", "delay2", "featlag1", "dowplac51"):
        cc = [prep(cfgs[4], 0)]
        cc[0]["hps"], cc[0]["maps"] = cc[0]["hps"][:1], cc[0]["maps"][:2]
        apply_variant(var, cc)
        out = run({"id": "selftest", "seed": 42}, cc, False, None, log, var=var, df1=df1, skip_baseline=True)
        log(f"[selftest variante {var}] OK : effective {out['effective_configs'][0]['label']} "
            f"{out['effective_configs'][0]['model']} {out['effective_configs'][0]['features'][-3:]}")
        globals()["EXEC_DELAY_H"] = 1
    out = run({"id": "selftest", "seed": 42}, small[:2] + small[3:5], True, 1, log, df1=df1, skip_baseline=True)
    log("auto-test b5 : OK")
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
    out = run(cfg, configs, nested, shuffle_seed, log, var=var)
    sig = out.pop("_signal")
    proba = out.pop("_proba")
    out = {"commit": commit, "code_dirty": dirty, "variant": var, **out}
    path = args.out or (RESULTS / (f"{cfg['id']}.json" if var == "base" else f"{cfg['id']}_{var}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if var == "base" and args.out is None:
        d = RESULTS / f"adv_{cfg['id']}"
        d.mkdir(parents=True, exist_ok=True)
        sig.to_frame("signal").to_parquet(d / "signal_1h.parquet", engine="pyarrow")
        proba.to_parquet(d / "proba_4h.parquet", engine="pyarrow")
        log(f"signal 1 h et probas 4 h : {d}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    print(f"écrit : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
