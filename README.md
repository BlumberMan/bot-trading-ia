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

## Construire le dataset

Télécharge les klines publiques Binance spot BTCUSDT 1h (data.binance.vision, sans clé API)
de 2019-01 à 2026-08 dans `data/raw/` (archives mensuelles uniquement ; une archive
manquante ou un checksum faux fait échouer le build), vérifie chaque SHA256, nettoie et écrit
`data/processed/btcusdt_1h.parquet` (les archives déjà présentes et valides ne sont pas
retéléchargées) :

```powershell
.venv\Scripts\python scripts\build_dataset.py
.venv\Scripts\python scripts\check_live_equality.py   # features live == backtest
.venv\Scripts\python scripts\show_folds.py            # folds walk-forward (dev uniquement)
```

Définition des features et labels : `docs/features.md`.

Baseline SMA168 et buy & hold sur les 9 folds dev (règles : `docs/baseline.md`, résultats : `results/baseline_metrics.json`) : `.venv\Scripts\python scripts\run_baseline.py`

## Structure

- `src/bot/` : code du package
- `tests/` : tests pytest
- `scripts/` : scripts (dataset, vérifications)
- `docs/` : documentation (features, labels, split)
- `data/` : données (`data/raw/` et `data/processed/` ignorés par git)
