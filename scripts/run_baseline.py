"""Évalue la baseline SMA168 et le buy & hold sur les 9 folds de test (période dev)
et sur leur concaténation 2021-01-01 00:00 -> 2025-09-30 23:00 UTC.

Usage : .venv\\Scripts\\python scripts\\run_baseline.py [--dataset CHEMIN] [--out CHEMIN]
Écrit results/baseline_metrics.json. Le holdout n'est jamais lu.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest import BacktestResult
from bot.baseline import COST_PER_SIDE, EXEC_DELAY, SMA_WINDOW, run_baseline, run_buy_and_hold
from bot.data import sha256_file
from bot.metrics import summarize
from bot.split import DEFAULT_DATASET, DEV_END, load_dataset, make_folds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "baseline_metrics.json"
CONCAT_START = pd.Timestamp("2021-01-01 00:00", tz="UTC")
CONCAT_END = DEV_END
COLS = [("sharpe", "Sharpe", "{:>7.3f}"), ("max_drawdown", "DD max", "{:>7.2%}"),
        ("trades", "trades", "{:>6d}"), ("win_rate", "win", "{:>6.1%}"),
        ("profit_factor", "PF", "{:>6.2f}"), ("exposure", "expo", "{:>6.1%}"),
        ("total_return", "rdt tot", "{:>9.2%}"), ("annualized_return", "rdt an", "{:>8.2%}")]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def fmt_row(label: str, m: dict) -> str:
    cells = []
    for key, _, f in COLS:
        v = m[key]
        if isinstance(v, float) and not math.isfinite(v):
            w = len(f.format(0.0 if "d}" not in f else 0))
            cells.append(f"{str(v):>{w}}")
        else:
            cells.append(f.format(v))
    return f"{label:<18} | " + " | ".join(cells)


def header(first: str) -> str:
    widths = [len(f.format(0.0 if "d}" not in f else 0)) for _, _, f in COLS]
    return f"{first:<18} | " + " | ".join(f"{name:>{w}}" for (_, name, _), w in zip(COLS, widths))


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def straddling_count(results: list[BacktestResult]) -> int:
    """Trades ouverts en fin de fold k ET en début de fold k+1 (comptés deux fois)."""
    n = 0
    for r0, r1 in zip(results[:-1], results[1:]):
        if (len(r0.trades) and bool(r0.trades["open_at_end"].iloc[-1]) and len(r1.trades)
                and bool(r1.trades["open_at_start"].iloc[0])):
            n += 1
    return n


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = load_dataset(args.dataset)  # période dev uniquement, holdout exclu
    if df.index.max() > DEV_END:
        raise RuntimeError("données postérieures à DEV_END : refus")
    parquet_sha = sha256_file(args.dataset)
    commit = git("rev-parse", "HEAD")
    dirty_code = git("status", "--porcelain", "--", "src", "scripts") != ""

    folds = make_folds()
    base_res, bh_res, rows = [], [], []
    for f in folds:
        b = run_baseline(df, f.test_start, f.test_end)
        h = run_buy_and_hold(df, f.test_start, f.test_end)
        base_res.append(b)
        bh_res.append(h)
        rows.append({"fold": f.k, "test_start": f.test_start.isoformat(),
                     "test_end": f.test_end.isoformat(),
                     "baseline": summarize(b), "buy_and_hold": summarize(h)})
    base_c = run_baseline(df, CONCAT_START, CONCAT_END)
    bh_c = run_buy_and_hold(df, CONCAT_START, CONCAT_END)
    m_base_c, m_bh_c = summarize(base_c), summarize(bh_c)

    # ------------------------------------------------ cohérence folds / continu
    contiguous = all(f1.test_start - f0.test_end == pd.Timedelta(hours=1)
                     for f0, f1 in zip(folds[:-1], folds[1:]))
    contiguous &= folds[0].test_start == CONCAT_START and folds[-1].test_end == CONCAT_END
    acc, parts = 1.0, []
    for r in base_res:
        parts.append(r.equity * acc)
        acc *= r.final_equity
    chained = pd.concat(parts)
    same_index = chained.index.equals(base_c.equity.index)
    max_abs_diff = float(np.max(np.abs(chained.to_numpy() - base_c.equity.to_numpy())))
    final_diff = abs(acc - base_c.final_equity)
    trades_sum = sum(len(r.trades) for r in base_res)
    straddle = straddling_count(base_res)
    tr_from_bars = float(np.prod(1.0 + base_c.bar_return.to_numpy()) - 1.0)
    tr_from_folds = float(np.prod([r.final_equity for r in base_res]) - 1.0)
    n_pos = sum(1 for r in rows if r["baseline"]["sharpe"] > 0)
    last_eval = max([r.equity.index.max() for r in base_res + bh_res + [base_c, bh_c]])
    last_mark = max([r.end_mark_time for r in base_res + bh_res + [base_c, bh_c]])

    # ------------------------------------------------ affichage
    print(f"dataset  : {args.dataset.name}  sha256={parquet_sha}")
    print(f"dev      : {df.index.min()} -> {df.index.max()} ({len(df)} lignes, "
          f"{int(df['missing'].sum())} manquantes)")
    print(f"commit   : {commit}  code modifié non commité : {dirty_code}")
    print(f"baseline : close > SMA{SMA_WINDOW}, long/flat, exec_delay={EXEC_DELAY}, "
          f"coût/côté={COST_PER_SIDE}, cost_multiplier=1")
    print()
    for name, key in (("BASELINE SMA168", "baseline"), ("BUY & HOLD", "buy_and_hold")):
        print(f"=== {name} : par fold de test ===")
        print(header("fold (test)"))
        print("-" * len(header("fold (test)")))
        for r in rows:
            label = f"{r['fold']} {r['test_start'][:10]}"
            print(fmt_row(label, r[key]))
        print()
    print("=== Période concaténée 2021-01-01 00:00 -> 2025-09-30 23:00 UTC ===")
    print(header("stratégie"))
    print("-" * len(header("stratégie")))
    print(fmt_row("baseline SMA168", m_base_c))
    print(fmt_row("buy & hold", m_bh_c))
    print()
    print(f"folds baseline à Sharpe > 0 : {n_pos} / {len(rows)}")
    print()
    print("=== Cohérence ===")
    print(f"folds de test contigus, de {CONCAT_START} à {CONCAT_END} : {contiguous}")
    print(f"index chaîné == index continu : {same_index} ({len(chained)} bougies)")
    print(f"max |equity chaînée - equity continue| : {max_abs_diff:.3e}")
    print(f"|equity finale chaînée - continue|      : {final_diff:.3e}  "
          f"(chaînée {acc:.12f}, continue {base_c.final_equity:.12f})")
    print(f"trades : somme des folds {trades_sum} - à cheval sur deux folds {straddle} "
          f"= {trades_sum - straddle} ; continu {m_base_c['trades']}")
    print(f"rendement total : equity finale - 1 = {m_base_c['total_return']:.12f} ; "
          f"prod(1 + r_bougie) - 1 = {tr_from_bars:.12f} ; prod(folds) - 1 = {tr_from_folds:.12f}")
    print(f"dernière bougie évaluée : {last_eval} ; dernier open lu (marquage) : {last_mark} ; "
          f"<= {DEV_END} : {last_eval <= DEV_END and last_mark <= DEV_END}")

    ok = (contiguous and same_index and max_abs_diff <= 1e-9 and final_diff <= 1e-9
          and trades_sum - straddle == m_base_c["trades"]
          and abs(tr_from_bars - m_base_c["total_return"]) <= 1e-9
          and last_eval <= DEV_END and last_mark <= DEV_END)
    print(f"contrôles de cohérence : {'OK' if ok else 'ÉCHEC'}")

    out = {
        "commit": commit,
        "code_dirty": dirty_code,
        "dataset": args.dataset.name,
        "dataset_sha256": parquet_sha,
        "params": {"sma_window": SMA_WINDOW, "exec_delay": EXEC_DELAY,
                   "cost_per_side": COST_PER_SIDE, "cost_multiplier": 1.0,
                   "fees": 0.0010, "spread_slippage": 0.0005, "positions": "long/flat"},
        "period": {"start": CONCAT_START.isoformat(), "end": CONCAT_END.isoformat()},
        "folds": rows,
        "concatenated": {"baseline": m_base_c, "buy_and_hold": m_bh_c},
        "baseline_folds_sharpe_positive": n_pos,
        "consistency": {"folds_contiguous": contiguous, "same_index": same_index,
                        "max_abs_equity_diff": max_abs_diff, "final_equity_diff": final_diff,
                        "trades_sum_folds": trades_sum, "trades_straddling": straddle,
                        "trades_continuous": m_base_c["trades"],
                        "total_return_from_bars": tr_from_bars,
                        "last_evaluated_bar": last_eval.isoformat(),
                        "last_open_read": last_mark.isoformat(), "ok": ok},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(jsonable(out), fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    print(f"écrit : {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
