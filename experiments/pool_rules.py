r"""Lanceur parallèle borné (brief P3-B9), repris de pool_b7.py. Au plus 9 processus en même temps, chacun avec
OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1. Attend la fin de tous les processus (aucun processus
détaché). N'écrit jamais au registre (rules.py sans --register).

Usage : .venv\Scripts\python experiments\pool_rules.py --workers 9 --jobs FICHIER
FICHIER : une tâche par ligne = script (rules.py ou rules_report.py) puis ses arguments. Journal : FICHIER.log.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--jobs", type=Path, required=True)
    args = ap.parse_args()
    w = min(args.workers, 9)
    jobs = [ln.strip() for ln in args.jobs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    logp = args.jobs.with_suffix(".log")
    running, done, fails = [], 0, 0
    queue = list(jobs)
    with open(logp, "a", encoding="utf-8") as lg:
        lg.write(f"--- {len(jobs)} tâches, {w} workers\n")
        while queue or running:
            while queue and len(running) < w:
                j = queue.pop(0)
                parts = shlex.split(j)
                if parts[0] not in ("rules.py", "rules_report.py"):
                    raise ValueError(f"script non autorisé : {parts[0]}")
                if "--register" in parts:
                    raise ValueError("--register interdit dans le pool (registre écrit par un seul processus)")
                cmd = [str(PY), str(ROOT / "experiments" / parts[0]), *parts[1:]]
                ef = tempfile.TemporaryFile()
                pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=ef)
                running.append((pr, j, time.time(), ef))
            time.sleep(2)
            still = []
            for pr, j, t0, ef in running:
                rc = pr.poll()
                if rc is None:
                    still.append((pr, j, t0, ef))
                    continue
                ef.seek(0)
                err = ef.read().decode("utf-8", "replace")[-2000:] if rc else ""
                ef.close()
                done += 1
                fails += int(rc != 0)
                lg.write(f"rc={rc} {time.time() - t0:7.1f}s {j}\n" + (err + "\n" if err else ""))
                lg.flush()
                print(f"[{done}/{len(jobs)}] rc={rc} {time.time() - t0:.0f}s {j}", flush=True)
            running = still
    print(f"terminé : {done} tâches, {fails} échec(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
