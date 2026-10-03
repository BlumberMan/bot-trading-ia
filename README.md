# bot-trading-ia

Bot de trading BTC propulsé par l'IA (projet en construction, palier par palier).
Aucune promesse de gains. Les secrets vont dans `.env` (jamais commité).

## Prérequis

- Windows 11, Python 3.14 (launcher `py`), Git.

## Installation (PowerShell, depuis un clone propre)

```powershell
git clone <url-du-repo> bot-trading-ia
cd bot-trading-ia
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
```

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

## Structure

- `src/bot/` : code du package
- `tests/` : tests pytest
- `data/` : données (`data/raw/` ignoré par git)
