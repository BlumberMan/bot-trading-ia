# JOURNAL
| ID | Date | Statut | Tag |
|---|---|---|---|
| P0a-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-0a |
| P0b-B1 / R1 / V1 | 2026-10-04 | VALIDE | gonogo-v1 (posé par Iyad) |

## Récap palier 0a (2026-10-04)
- État : infra en place, package `bot` 0.0.1 (code dans src/), commit d8f1f11.
- Env : Python 3.14.7, .venv, versions figées dans requirements.lock (numpy 2.5.3, pandas 3.0.6, pytest 9.1.1).
- Tests : 3 passés (repo local + clone propre, revérifiés par le contrôleur).
- Limites connues : setuptools (build) non figé ; numpy/pandas non bornés dans pyproject (lock fait foi) ; README avec placeholder `<url-du-repo>`.
- Remote GitHub présent (BlumberMan/bot-trading-ia), rien poussé.
- Prochaine étape : palier 0b, valeurs GONOGO.md à faire valider par Iyad.

## Récap palier 0b (2026-10-04)
- GONOGO.md v1 commité par Iyad (61cb863), tag gonogo-v1 posé et poussé par Iyad.
- Backtest : Sharpe net OOS > 1,0 (journalier, rf=0, ×√365) ; DD < 25 % ; trades OOS ≥ 200 ; folds+ > 60 % ; écart baseline > 0,3 ; stabilité coûts×2, t+2, hyperparams ±20 %.
- Paper : 6 semaines, 50 trades, concordance ≥ 95 %, écart ≤ 2 pts.
- Réel : 200 €, coupure −20 %, perte jour 3 %, risque/trade 1 %, levier 1 (spot).
- Iyad a élargi les permissions (ffee3ee) et autorisé le travail autonome jusqu'au prochain point critique, avec `git push origin main --tags` à chaque palier validé.
- Prochaine étape : palier 1, données et features.

## Décisions non critiques (prises en autonomie)
- 2026-10-04 — Source de données : klines publiques Binance spot BTCUSDT (data.binance.vision). Gratuit, sans clé API, historique long, liquidité maximale ; le broker réel sera choisi au palier 5.
- 2026-10-04 — Taille de bougie : 1h. Compromis : assez de trades pour viser ≥ 200 trades OOS, coûts par trade moins pénalisants qu'en 1-15 min, latence de calcul non critique.
- 2026-10-04 — Période : 2019-01-01 → dernier mois complet disponible. Couvre plusieurs régimes (bear 2019/2022, bull 2020-21/2024) ; avant 2019 la microstructure est trop différente.
- 2026-10-04 — Stockage : fichiers bruts dans data/raw/ (ignoré par git), dataset nettoyé en Parquet (pyarrow ajouté aux dépendances), checksums des fichiers bruts vérifiés.
