r"""Lanceur parallèle borné (brief P3-B5, contrainte de calcul) : au plus --workers processus en même temps
(défaut 9, plafond 9), chacun avec OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1 (LightGBM
n_jobs=1 dans b5.py). Attend la fin de tous les processus (aucun processus détaché). N'écrit jamais au registre.

Usage : .venv\Scripts\python experiments\pool_b5.py --workers 9 --jobs FICHIER
FICHIER : une tâche par ligne = arguments de experiments\b5.py (ex. « experiments/configs/E29.json --variant shuf1 »).
Journal : FICHIER.log (code retour et durée de chaque tâche).
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
                cmd = [str(PY), str(ROOT / "experiments" / "b5.py"), *shlex.split(j)]
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
