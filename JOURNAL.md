# JOURNAL
| ID | Date | Statut | Tag |
|---|---|---|---|
| P0a-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-0a |
| P0b-B1 / R1 / V1 | 2026-10-04 | VALIDE | gonogo-v1 (posé par Iyad) |
| P1-B1 / R1 / V1 | 2026-10-04 | INVALIDE (1/3) : A1, archive mensuelle 2026-09 absente (404), remplacée par 30 journalières sans décision du brief | — |
| P1-B2 / R2 / V2 | 2026-10-04 | VALIDE (KO A1 levé) | palier-1 |
| P2-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-2 |

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

## Récap palier 1 (2026-10-04)
- Données : Binance spot BTCUSDT 1h, 92 archives mensuelles (2019-01 → 2026-08), 92/92 checksums ; 67200 lignes, 59 bougies manquantes (20 trous), 0 doublon, 0 violation.
- Parquet reproductible : SHA256 0f505ec657caf30771662835857ee29a09c439ada8da564ab50a76bcb4fdf336 (local ×2 + clone propre).
- 16 features à fenêtre finie (LOOKBACK 200), égalité live/batch 2000/2000 bit à bit ; labels H=4 (open t+1 → open t+5).
- Dev 2019-01 → 2025-09 (9 folds walk-forward, purge 5, embargo 24) ; holdout 2025-10-01 → 2026-08-31 verrouillé (allow_holdout=True).
- 39 tests verts. 1 INVALIDE (archive 2026-09 absente) corrigé par fin des données au 2026-08-31.
- Référence d'intégrité : tag gonogo-v1 → commit 61cb863ea30e9271c52fce7ac231967748b50fbd.
- Prochaine étape : palier 2 (baseline).

## Récap palier 2 (2026-10-04)
- Moteur de backtest (exec_delay, cost_per_side, cost_multiplier, gestion des trous) + métriques (Sharpe GONOGO) dans src/bot/ ; 68 tests verts.
- Règles baseline figées en 696f8b1 avant les résultats (29597b7) : long si close > SMA168, sinon flat ; t+1 ; 0,15 %/côté.
- OOS concaténé 2021-01 → 2025-09 : baseline Sharpe −0,530, DD 87,83 %, 747 trades, −77,31 %, 4/9 folds à Sharpe > 0.
- Buy & hold même période : Sharpe 0,781, DD 77,20 %, +292,75 %.
- À reprendre au palier 3 : la baseline hérite sa position sans coût en début de fenêtre, le B&H paie entrée et sortie par fold (écart ≤ ~0,15 %/côté/fenêtre).
- Garde : toute commande Bash contenant « GONOGO.md » est bloquée, même en lecture ; le contrôleur vérifie l'intégrité directement.
- Prochaine étape : STOP demandé par Iyad avant le palier 3.

## Consignes d'Iyad en cours
- 2026-10-04 — STOP obligatoire après la validation du palier 2, avant de lancer le palier 3 (Iyad ajoute des sous-agents).

## Décisions non critiques (prises en autonomie)
- 2026-10-04 — Source de données : klines publiques Binance spot BTCUSDT (data.binance.vision). Gratuit, sans clé API, historique long, liquidité maximale ; le broker réel sera choisi au palier 5.
- 2026-10-04 — Taille de bougie : 1h. Compromis : assez de trades pour viser ≥ 200 trades OOS, coûts par trade moins pénalisants qu'en 1-15 min, latence de calcul non critique.
- 2026-10-04 — Période : 2019-01-01 → dernier mois complet disponible. Couvre plusieurs régimes (bear 2019/2022, bull 2020-21/2024) ; avant 2019 la microstructure est trop différente.
- 2026-10-04 — Stockage : fichiers bruts dans data/raw/ (ignoré par git), dataset nettoyé en Parquet (pyarrow ajouté aux dépendances), checksums des fichiers bruts vérifiés.
- 2026-10-04 — Découpage : période de développement 2019-01-01 → 2025-09-30 (walk-forward, 9 folds de test semestriels à partir de 2021-01-01, fenêtre d'entraînement croissante) ; holdout final 2025-10-01 → 2026-09-30 réservé au palier 4, inaccessible sans flag explicite. Permet de respecter C4-03 (jeu OOS final jamais utilisé avant).
- 2026-10-04 — Labels : signal à la clôture de t, exécution à l'ouverture de t+1 ; label = 1 si open[t+1+H] > open[t+1], H = 4 bougies. Purge = H+1 bougies, embargo = 24 bougies.
- 2026-10-04 — Correction de la décision "Période" (suite P1-V1) : fin des données ramenée au 2026-08-31 23:00 UTC, dernier mois publié en archive mensuelle sur data.binance.vision (2026-09 en 404 le 2026-10-04). Archives mensuelles uniquement, pas de repli sur les journalières : un clone propre reconstruit le même fichier. Le holdout devient 2025-10-01 → 2026-08-31 (11 mois) ; la période dev et les 9 folds ne changent pas. Le holdout n'a fait l'objet d'aucune statistique de rendement, donc ce changement n'introduit aucun biais.
- 2026-10-04 — Palier 2, sens des positions : long / flat uniquement (pas de vente à découvert), cohérent avec le spot sans levier imposé en réel par GONOGO.
- 2026-10-04 — Palier 2, coûts provisoires : 0,10 % de frais (tarif taker standard Binance spot) + 0,05 % de spread/slippage, soit 0,15 % par côté, par unité de position échangée. Hypothèse prudente ; les coûts seront sourcés et affinés au palier 4.
- 2026-10-04 — Palier 2, baseline : long si close[t] > SMA168(close)[t], sinon flat ; décidé à la clôture de t, exécuté à l'ouverture de t+1. Paramètre 168 fixé a priori (1 semaine), sans aucune optimisation. Benchmark buy & hold sur la même période, avec les mêmes coûts.
- 2026-10-04 — Palier 2, évaluation : uniquement sur les 9 folds de test de la période dev (concaténés : 2021-01-01 → 2025-09-30). Le holdout n'est pas touché.
- 2026-10-04 — Features à fenêtre finie uniquement (pas d'EMA à mémoire infinie) pour que le calcul live sur une fenêtre glissante soit exactement égal au calcul backtest.
