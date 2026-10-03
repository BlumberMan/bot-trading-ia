import sys
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

import bot

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_import_and_version():
    assert bot.__version__ == "0.0.1"


def test_version_matches_pyproject():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["version"] == bot.__version__


def test_python_version_matches_requires_python():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    spec = SpecifierSet(data["project"]["requires-python"])
    running = ".".join(str(p) for p in sys.version_info[:3])
    assert running in spec, f"Python {running} hors de {spec}"
