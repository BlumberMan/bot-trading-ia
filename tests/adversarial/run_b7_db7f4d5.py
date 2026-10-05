"""Avocat du diable P3-A5, T9 : exécute experiments/b7.py TEL QU'À db7f4d5 (extrait par git show dans un dossier
temporaire, hors arbre) sur E40, sortie --out hors arbre, pour comparaison avec b7.py de HEAD.
Usage : .venv\Scripts\python tests\adversarial\run_b7_db7f4d5.py CHEMIN_B7_DB7 CHEMIN_SORTIE.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("b7_db7", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv = ["b7_db7", str(ROOT / "experiments" / "configs" / "E40.json"), "--out", sys.argv[2]]
sys.exit(mod.main())
