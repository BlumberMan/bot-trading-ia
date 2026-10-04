"""Harnais commun d'exploration palier 3 (walk-forward strict, tuning interne au train).

Usage :
    .venv\\Scripts\\python experiments\\harness.py experiments\\configs\\E01.json [--registry]
    .venv\\Scripts\\python experiments\\harness.py --check-only
    .venv\\Scripts\\python experiments\\harness.py CONFIG --shuffle-seed S   (test labels mélangés)

Procédure (pour chaque fold k de bot.split.make_folds(horizon=H)) :
1. train = bot.split.split_fold(df, fold, horizon=H) (purge H+1 incluse), lignes sans
   NaN de features ni de label ;
2. validation interne = derniers 20 % (en lignes) du train ; train interne = lignes
   antérieures, purgées par bot.split.purge_mask (les t dont [t+1, t+1+H] chevauche la
   validation sont retirés) ;
3. pour chaque hyperparamètre de la grille : fit sur le train interne, proba sur toutes
   les bougies de la fenêtre de validation [val_start, fold.train_end] ; pour chaque
   mapping (seuil d'entrée, écart de sortie, détention min) : backtest bot.backtest
   (exec_delay=1, 0,0015/côté, ×1) de la fenêtre de validation -> Sharpe journalier
   bot.metrics.sharpe_daily. Score NaN (aucune variance, ex. jamais investi) = -inf.
   Choix = premier maximum dans l'ordre de la grille ;
4. réentraînement sur tout le train du fold avec les hyperparamètres choisis, proba sur
   les bougies t dont la position (tenue à t+1) tombe dans le test du fold :
   t ∈ [test_start - 1h, test_end - 1h] (fold 1 : à partir de test_start - 2h, pour que
   la position tenue juste avant 2021-01-01 soit héritée comme pour la baseline ;
   fold 9 : jusqu'à test_end). Le test du fold n'est jamais utilisé pour un choix.
Évaluation : probas des 9 folds concaténées, mapping appliqué en continu (état conservé
entre folds, seuils du fold propriétaire de chaque bougie), UN backtest continu ;
métriques par fold = fenêtres [test_start, test_end] du même signal (comme
scripts/run_baseline.py), plus la période concaténée 2021-01-01 -> 2025-09-30 23:00.
Pas de normalisation (arbres). Long / flat uniquement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bot.backtest import run_backtest  # noqa: E402
from bot.baseline import run_baseline, run_buy_and_hold  # noqa: E402
from bot.data import sha256_file  # noqa: E402
from bot.features import FEATURES, compute_features  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.metrics import daily_returns, sharpe_daily, summarize  # noqa: E402
from bot.split import DEFAULT_DATASET, H1, load_dataset, make_folds, purge_mask, split_fold  # noqa: E402
from extra_features import EXTRA_FEATURES, compute_extra  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RESULTS = EXP / "results"
REGISTRE = EXP / "REGISTRE.md"
BASELINE_JSON = ROOT / "results" / "baseline_metrics.json"

COST_PER_SIDE = 0.0015
EXEC_DELAY = 1
COST_MULTIPLIER = 1.0
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")  # fin de la période dev
CONCAT_START = pd.Timestamp("2021-01-01 00:00", tz="UTC")
CONCAT_END = MAX_INDEX
VAL_FRAC = 0.20
BASELINE_SHARPE_REF = -0.530  # règle de choix (JOURNAL.md)

METRIC_COLS = [("sharpe", "Sharpe", "{:>7.3f}"), ("max_drawdown", "DD max", "{:>7.2%}"),
               ("trades", "trades", "{:>6d}"), ("win_rate", "win", "{:>6.1%}"),
               ("profit_factor", "PF", "{:>6.2f}"), ("exposure", "expo", "{:>6.1%}"),
               ("total_return", "rdt tot", "{:>9.2%}")]


# ------------------------------------------------------------------ utilitaires

class Tee:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def fmt_metrics(m: dict) -> str:
    cells = []
    for key, _, f in METRIC_COLS:
        v = m[key]
        w = len(f.format(0 if "d}" in f else 0.0))
        if isinstance(v, float) and not math.isfinite(v):
            cells.append(f"{str(v):>{w}}")
        else:
            cells.append(f.format(v))
    return " | ".join(cells)


def metrics_header() -> str:
    widths = [len(f.format(0 if "d}" in f else 0.0)) for _, _, f in METRIC_COLS]
    return " | ".join(f"{n:>{w}}" for (_, n, _), w in zip(METRIC_COLS, widths))


# ------------------------------------------------------------------ données

def load_dev(log: Tee) -> pd.DataFrame:
    """Période dev uniquement (load_dataset par défaut) + assertion sur l'index max."""
    df = load_dataset()
    mx = df.index.max()
    assert mx <= MAX_INDEX, f"index max {mx} > {MAX_INDEX} : données réservées interdites"
    log(f"[garde] index max lu = {mx} <= {MAX_INDEX} : OK ({len(df)} lignes, "
        f"min {df.index.min()})")
    return df


def build_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    base = compute_features(df)
    parts = [base]
    if any(f in EXTRA_FEATURES for f in feats):
        parts.append(compute_extra(df))
    allf = pd.concat(parts, axis=1)
    unknown = [f for f in feats if f not in allf.columns]
    if unknown:
        raise ValueError(f"features inconnues : {unknown}")
    return allf[feats]


def resolve_features(spec) -> list[str]:
    if spec == "base":
        return list(FEATURES)
    if spec == "base+extra":
        return list(FEATURES) + list(EXTRA_FEATURES)
    return list(spec)


# ------------------------------------------------------------------ modèles

def make_model(kind: str, hp: dict, seed: int):
    if kind == "lgbm":
        import lightgbm as lgb
        params = dict(objective="binary", random_state=seed, n_jobs=1, deterministic=True,
                      force_row_wise=True, verbose=-1)
        params.update(hp)
        return lgb.LGBMClassifier(**params)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        params = dict(random_state=seed, early_stopping=False)
        params.update(hp)
        return HistGradientBoostingClassifier(**params)
    raise ValueError(kind)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Proba de la classe 1 ; NaN si une feature est NaN."""
    out = np.full(len(X), np.nan)
    ok = ~X.isna().any(axis=1).to_numpy()
    if ok.any():
        out[ok] = model.predict_proba(X.to_numpy()[ok])[:, 1]
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~(np.isnan(y) | np.isnan(p))
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], p[m]))


# ------------------------------------------------------------------ mapping proba -> position

def map_positions(p: np.ndarray, enter: np.ndarray, exit_: np.ndarray,
                  min_hold: np.ndarray) -> np.ndarray:
    """Machine à états long / flat.

    flat -> long si p > enter ; long -> flat si p <= exit ET détention >= min_hold
    bougies. exit = enter et min_hold = 0 <=> position = 1 si p > seuil, sinon 0.
    p NaN -> signal NaN (le moteur conserve la position), l'état est conservé.
    """
    n = len(p)
    out = np.full(n, np.nan)
    pos, held = 0, 0
    for i in range(n):
        if pos == 1:
            held += 1
        pi = p[i]
        if pi != pi:  # NaN
            continue
        if pos == 0:
            if pi > enter[i]:
                pos, held = 1, 0
        elif pi <= exit_[i] and held >= min_hold[i]:
            pos = 0
        out[i] = pos
    return out


def mapping_grid(cfg: dict) -> list[dict]:
    m = cfg["mapping"]
    out = []
    for e, g, h in itertools.product(m["enter"], m.get("exit_gap", [0.0]), m.get("min_hold", [0])):
        out.append({"enter": float(e), "exit": float(e) - float(g), "min_hold": int(h)})
    return out


def hp_grid(cfg: dict) -> list[dict]:
    g = cfg["grid"]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def window_sharpe(p: np.ndarray, idx: pd.DatetimeIndex, mp: dict, open_: pd.Series,
                  start, end) -> tuple[float, int]:
    n = len(p)
    sig = map_positions(p, np.full(n, mp["enter"]), np.full(n, mp["exit"]),
                        np.full(n, mp["min_hold"], dtype=np.int64))
    res = run_backtest(pd.Series(sig, index=idx), open_, start=start, end=end,
                       exec_delay=EXEC_DELAY, cost_per_side=COST_PER_SIDE,
                       cost_multiplier=COST_MULTIPLIER)
    return sharpe_daily(res.equity), int(len(res.trades))


# ------------------------------------------------------------------ baseline / B&H

def baseline_block(df: pd.DataFrame, folds, log: Tee) -> tuple[list[dict], dict, dict]:
    rows = []
    for f in folds:
        rows.append({"baseline": summarize(run_baseline(df, f.test_start, f.test_end)),
                     "buy_and_hold": summarize(run_buy_and_hold(df, f.test_start, f.test_end))})
    concat = {"baseline": summarize(run_baseline(df, CONCAT_START, CONCAT_END)),
              "buy_and_hold": summarize(run_buy_and_hold(df, CONCAT_START, CONCAT_END))}
    ref = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    diffs = {}
    for key in ("baseline", "buy_and_hold"):
        for mk in ("sharpe", "final_equity"):
            d = abs(concat[key][mk] - float(ref["concatenated"][key][mk]))
            dmax_f = max(abs(r[key][mk] - float(ref["folds"][i][key][mk])) for i, r in enumerate(rows))
            diffs[f"{key}.{mk}.concat"] = d
            diffs[f"{key}.{mk}.folds_max"] = dmax_f
    ok = all(v <= 1e-9 for v in diffs.values())
    log("[égalité baseline / B&H vs results/baseline_metrics.json]")
    for k, v in diffs.items():
        log(f"  |écart| {k:<32} = {v:.3e}")
    log(f"  baseline concat : Sharpe {concat['baseline']['sharpe']:.12f} (réf "
        f"{float(ref['concatenated']['baseline']['sharpe']):.12f}), equity finale "
        f"{concat['baseline']['final_equity']:.12f} (réf "
        f"{float(ref['concatenated']['baseline']['final_equity']):.12f})")
    log(f"  B&H concat      : Sharpe {concat['buy_and_hold']['sharpe']:.12f} (réf "
        f"{float(ref['concatenated']['buy_and_hold']['sharpe']):.12f}), equity finale "
        f"{concat['buy_and_hold']['final_equity']:.12f} (réf "
        f"{float(ref['concatenated']['buy_and_hold']['final_equity']):.12f})")
    log(f"  contrôle d'égalité (<= 1e-9) : {'OK' if ok else 'ÉCHEC'}")
    if not ok:
        raise RuntimeError("baseline / B&H du harnais différents de results/baseline_metrics.json")
    return rows, concat, diffs


# ------------------------------------------------------------------ procédure d'un fold

def run_fold(f, df, X, y, cfg, hps, maps, seed, shuffle_rng, log: Tee) -> dict:
    H = int(cfg["horizon"])
    kind = cfg["model"]
    tr_df, _ = split_fold(df, f, horizon=H)
    tr_idx = tr_df.index
    Xtr_all = X.loc[tr_idx]
    ytr_all = y.loc[tr_idx]
    clean = (~Xtr_all.isna().any(axis=1)) & (~ytr_all.isna())
    Xtr = Xtr_all.loc[clean]
    ytr = ytr_all.loc[clean].to_numpy().copy()
    if shuffle_rng is not None:
        ytr = shuffle_rng.permutation(ytr)
    n = len(Xtr)
    v0 = int(math.floor((1.0 - VAL_FRAC) * n))
    val_start = Xtr.index[v0]
    val_end = f.train_end
    keep_inner = purge_mask(Xtr.index, val_start, val_end, horizon=H)
    inner = (np.arange(n) < v0) & keep_inner
    Xin, yin = Xtr.iloc[inner], ytr[inner]
    Xval_rows, yval_rows = Xtr.iloc[v0:], ytr[v0:]
    # bougies de la fenêtre de validation (y compris NaN de features)
    vmask = (df.index >= val_start) & (df.index <= val_end)
    vidx = df.index[vmask]
    Xv = X.loc[vidx]
    open_v = df["open"].loc[val_start: val_end + H1]
    assert open_v.index.max() <= f.test_start - H1

    grid_scores = []
    best = None
    for hi, hp in enumerate(hps):
        mdl = make_model(kind, hp, seed).fit(Xin.to_numpy(), yin)
        pv = predict_proba(mdl, Xv)
        auc_v = auc(yval_rows, predict_proba(mdl, Xval_rows))
        for mi, mp in enumerate(maps):
            s, ntr = window_sharpe(pv, vidx, mp, open_v, val_start, val_end)
            score = s if (isinstance(s, float) and math.isfinite(s)) else -math.inf
            grid_scores.append({"hp": hi, "map": mi, "val_sharpe": score, "val_trades": ntr})
            if best is None or score > best["score"]:
                best = {"score": score, "hp_i": hi, "map_i": mi, "val_trades": ntr, "auc_val": auc_v}
    hp, mp = hps[best["hp_i"]], maps[best["map_i"]]
    model = make_model(kind, hp, seed).fit(Xtr.to_numpy(), ytr)

    # in-sample
    p_tr_rows = predict_proba(model, Xtr)
    auc_tr = auc(ytr, p_tr_rows)
    t_first = Xtr.index[0]
    is_mask = (df.index >= t_first) & (df.index <= f.train_end)
    is_idx = df.index[is_mask]
    p_is = predict_proba(model, X.loc[is_idx])
    sh_is, ntr_is = window_sharpe(p_is, is_idx, mp, df["open"].loc[t_first: f.train_end + H1],
                                  t_first, f.train_end)

    imp = None
    if kind == "lgbm":
        g = model.booster_.feature_importance(importance_type="gain")
        imp = {name: float(v) for name, v in zip(X.columns, g)}
    log(f"  fold {f.k}: train {tr_idx.min()} -> {f.train_end} ({n} lignes propres), "
        f"train interne {int(inner.sum())}, validation {val_start} -> {val_end} ({n - v0} lignes) ; "
        f"choix hp#{best['hp_i']} {hp} map {mp} : Sharpe val {best['score']:.3f} "
        f"({best['val_trades']} trades), AUC val {best['auc_val']:.4f}, AUC train {auc_tr:.4f}, "
        f"Sharpe in-sample {sh_is:.3f}")
    return {"fold": f.k, "train_start": tr_idx.min(), "train_end": f.train_end,
            "n_train": n, "n_inner_train": int(inner.sum()), "val_start": val_start,
            "val_end": val_end, "n_val": n - v0, "test_start": f.test_start, "test_end": f.test_end,
            "best_hp": hp, "best_map": mp, "best_val_sharpe": best["score"],
            "best_val_trades": best["val_trades"], "auc_val": best["auc_val"], "auc_train": auc_tr,
            "sharpe_in_sample": sh_is, "trades_in_sample": ntr_is, "grid_scores": grid_scores,
            "importance_gain": imp, "_model": model}


# ------------------------------------------------------------------ essai complet

def run_trial(cfg: dict, shuffle_seed: int | None, log: Tee) -> dict:
    seed = int(cfg["seed"])
    H = int(cfg["horizon"])
    feats = resolve_features(cfg["features"])
    df = load_dev(log)
    folds = make_folds(horizon=H)
    log(f"essai {cfg['id']} : modèle {cfg['model']}, H={H}, {len(feats)} features, seed {seed}"
        + (f", LABELS MÉLANGÉS seed {shuffle_seed}" if shuffle_seed is not None else ""))
    base_rows, base_concat, base_diffs = baseline_block(df, folds, log)

    X = build_matrix(df, feats)
    y = compute_labels(df, horizon=H)["label"]
    assert X.index.max() <= MAX_INDEX and y.index.max() <= MAX_INDEX
    hps, maps = hp_grid(cfg), mapping_grid(cfg)
    log(f"grille interne : {len(hps)} hyperparamètres x {len(maps)} mappings = {len(hps) * len(maps)}")
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    n = len(df)
    p_all = np.full(n, np.nan)
    enter = np.full(n, np.inf)
    exit_ = np.full(n, -np.inf)
    mh = np.zeros(n, dtype=np.int64)
    fold_out = []
    for f in folds:
        r = run_fold(f, df, X, y, cfg, hps, maps, seed, rng, log)
        a = f.test_start - (2 if f.k == 1 else 1) * H1
        b = f.test_end if f.k == len(folds) else f.test_end - H1
        cm = (df.index >= a) & (df.index <= b)
        p_all[cm] = predict_proba(r["_model"], X.loc[cm])
        enter[cm], exit_[cm], mh[cm] = r["best_map"]["enter"], r["best_map"]["exit"], r["best_map"]["min_hold"]
        yt = y.loc[(df.index >= f.test_start) & (df.index <= f.test_end)].to_numpy()
        pt = p_all[(df.index >= f.test_start) & (df.index <= f.test_end)]
        r["auc_test"] = auc(yt, pt)
        r["signal_cover"] = [a, b]
        del r["_model"]
        fold_out.append(r)

    sig = pd.Series(map_positions(p_all, enter, exit_, mh), index=df.index, name="model_signal")
    assert sig.loc[: CONCAT_START - 3 * H1].isna().all()
    for f, r in zip(folds, fold_out):
        res = run_backtest(sig, df["open"], start=f.test_start, end=f.test_end,
                           exec_delay=EXEC_DELAY, cost_per_side=COST_PER_SIDE,
                           cost_multiplier=COST_MULTIPLIER)
        r["model"] = summarize(res)
    res_c = run_backtest(sig, df["open"], start=CONCAT_START, end=CONCAT_END,
                         exec_delay=EXEC_DELAY, cost_per_side=COST_PER_SIDE,
                         cost_multiplier=COST_MULTIPLIER)
    m_c = summarize(res_c)
    dr = daily_returns(res_c.equity).to_numpy()
    from scipy.stats import kurtosis, skew
    daily = {"n_days": int(len(dr)), "mean": float(dr.mean()), "std": float(dr.std(ddof=1)),
             "sr_daily": float(dr.mean() / dr.std(ddof=1)),
             "skew": float(skew(dr, bias=False)),
             "kurtosis": float(kurtosis(dr, fisher=False, bias=False))}

    # ------------------------------------------------ affichage A4
    log()
    log("=== Par fold (test) : modèle / baseline SMA168 / buy & hold, mêmes folds, mêmes coûts ===")
    log(f"{'fold':<16} | {'stratégie':<9} | " + metrics_header())
    for f, r, b in zip(folds, fold_out, base_rows):
        lab = f"{f.k} {f.test_start.date()}"
        log(f"{lab:<16} | {'modèle':<9} | " + fmt_metrics(r["model"]))
        log(f"{'':<16} | {'baseline':<9} | " + fmt_metrics(b["baseline"]))
        log(f"{'':<16} | {'B&H':<9} | " + fmt_metrics(b["buy_and_hold"]))
    lab = "concat 2021-25"
    log(f"{lab:<16} | {'modèle':<9} | " + fmt_metrics(m_c))
    log(f"{'':<16} | {'baseline':<9} | " + fmt_metrics(base_concat["baseline"]))
    log(f"{'':<16} | {'B&H':<9} | " + fmt_metrics(base_concat["buy_and_hold"]))
    log()
    log("=== In-sample vs OOS par fold ===")
    log("fold | Sharpe IS (train) | Sharpe val int. | Sharpe OOS | AUC train | AUC val | AUC test")
    for r in fold_out:
        log(f"{r['fold']:>4} | {r['sharpe_in_sample']:>17.3f} | {r['best_val_sharpe']:>15.3f} | "
            f"{r['model']['sharpe']:>10.3f} | {r['auc_train']:>9.4f} | {r['auc_val']:>7.4f} | "
            f"{r['auc_test']:>8.4f}")
    d_base = m_c["sharpe"] - base_concat["baseline"]["sharpe"]
    d_bh = m_c["sharpe"] - base_concat["buy_and_hold"]["sharpe"]
    npos = sum(1 for r in fold_out if isinstance(r["model"]["sharpe"], float) and r["model"]["sharpe"] > 0)
    log()
    log(f"Sharpe OOS concaténé modèle {m_c['sharpe']:.4f} ; baseline {base_concat['baseline']['sharpe']:.4f} "
        f"(écart {d_base:+.4f}) ; B&H {base_concat['buy_and_hold']['sharpe']:.4f} (écart {d_bh:+.4f}) ; "
        f"trades {m_c['trades']} ; folds à Sharpe > 0 : {npos}/9")
    log(f"rendements journaliers OOS : {daily}")
    last_open = max(res_c.end_mark_time, *[pd.Timestamp(r['model']['end_mark_time']) for r in fold_out])
    log(f"[garde] dernier open lu par les backtests = {last_open} <= {MAX_INDEX} : {last_open <= MAX_INDEX}")
    assert last_open <= MAX_INDEX

    imp_mean = None
    if cfg["model"] == "lgbm":
        imp_mean = {k: float(np.mean([r["importance_gain"][k] for r in fold_out])) for k in feats}
    return {
        "id": cfg["id"], "config": cfg, "seed": seed, "shuffle_seed": shuffle_seed,
        "dataset": DEFAULT_DATASET.name, "dataset_sha256": sha256_file(DEFAULT_DATASET),
        "index_max_read": df.index.max(), "cost_per_side": COST_PER_SIDE, "exec_delay": EXEC_DELAY,
        "cost_multiplier": COST_MULTIPLIER, "features": feats,
        "grid_size": {"hp": len(hps), "mapping": len(maps), "total": len(hps) * len(maps)},
        "hp_grid": hps, "mapping_grid": maps,
        "baseline_equality": base_diffs,
        "folds": [{**r, "baseline": b["baseline"], "buy_and_hold": b["buy_and_hold"]}
                  for r, b in zip(fold_out, base_rows)],
        "concatenated": {"model": m_c, "baseline": base_concat["baseline"],
                         "buy_and_hold": base_concat["buy_and_hold"]},
        "delta_sharpe_vs_baseline": d_base, "delta_sharpe_vs_bh": d_bh,
        "folds_sharpe_positive": npos, "daily_oos": daily,
        "importance_gain_mean": imp_mean,
    }


def registry_line(cfg: dict, out: dict, cmd: str, date: str) -> str:
    fs = ";".join(f"{r['model']['sharpe']:.3f}" for r in out["folds"])
    ft = ";".join(str(r["model"]["trades"]) for r in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = ("éligible (> baseline ; candidat final désigné dans experiments/README.md)"
              if eligible else "abandonné (<= baseline)")
    g = out["grid_size"]
    hp = json.dumps(cfg["grid"], separators=(",", ":"))
    mp = json.dumps(cfg["mapping"], separators=(",", ":"))
    feats = cfg["features"] if isinstance(cfg["features"], str) else ",".join(cfg["features"])
    return (f"| {cfg['id']} | {date} | {out['commit'][:7]} | seed={out['seed']} | {cfg['model']} | "
            f"features={feats} | H={cfg['horizon']} | grille hp={hp} mapping={mp} | "
            f"grille interne {g['hp']}x{g['mapping']}={g['total']} | folds 1-9 (make_folds H={cfg['horizon']}) | "
            f"Sharpe/fold {fs} | trades/fold {ft} | Sharpe concat {m['sharpe']:.4f} | trades concat {m['trades']} | "
            f"écart baseline {out['delta_sharpe_vs_baseline']:+.4f} | écart B&H {out['delta_sharpe_vs_bh']:+.4f} | "
            f"{statut} | `{cmd}` |")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", type=Path)
    ap.add_argument("--registry", action="store_true", help="ajoute la ligne au REGISTRE (code commité exigé)")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    log = Tee()
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "src", "experiments/harness.py",
                "experiments/extra_features.py", "experiments/configs") != ""
    log(f"commit {commit} ; code modifié non commité (src, harnais, configs) : {dirty}")

    if args.check_only:
        df = load_dev(log)
        baseline_block(df, make_folds(), log)
        p = np.array([0.4, 0.6, np.nan, 0.55, 0.45, 0.7, 0.3])
        n = len(p)
        simple = map_positions(p, np.full(n, 0.5), np.full(n, 0.5), np.zeros(n, dtype=np.int64))
        log(f"[auto-test mapping] seuil 0.5 : {simple.tolist()}")
        assert np.array_equal(simple, np.array([0, 1, np.nan, 1, 0, 1, 0]), equal_nan=True)
        hyst = map_positions(p, np.full(n, 0.5), np.full(n, 0.4), np.full(n, 3, dtype=np.int64))
        log(f"[auto-test mapping] entrée 0.5 / sortie 0.4 / détention 3 : {hyst.tolist()}")
        assert np.array_equal(hyst, np.array([0, 1, np.nan, 1, 1, 1, 0]), equal_nan=True)
        log("check-only : OK")
        return 0

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = run_trial(cfg, args.shuffle_seed, log)
    out = {"commit": commit, "code_dirty": dirty, **out}
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shuffle_seed is None else f"_shuf{args.shuffle_seed}"
    path = args.out or (RESULTS / f"{cfg['id']}{suffix}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    log(f"écrit : {path}")
    if args.registry:
        if dirty:
            raise RuntimeError("code non commité : ligne de registre refusée")
        if args.shuffle_seed is not None:
            raise RuntimeError("le test labels mélangés n'est pas un essai du registre")
        cmd = f".venv\\Scripts\\python experiments\\harness.py experiments\\configs\\{args.config.name} --registry"
        date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        line = registry_line(cfg, out, cmd, date)
        with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        n_lines = len(REGISTRE.read_text(encoding="utf-8").splitlines())
        log(f"registre : ligne ajoutée ({n_lines} lignes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
