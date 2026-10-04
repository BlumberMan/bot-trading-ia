# GONOGO — Critères de passage du bot de trading BTC

Date : 2026-10-04

Seuils fixés avant tout résultat, validés par Iyad. Figé par le tag gonogo-v1 posé par Iyad. Toute modification est interdite.

## A. BACKTEST (évalué au palier 4, données hors échantillon OOS, frais compris)

* **Sharpe net OOS > 1,0** — après frais, en dessous de 1 le bruit domine ; au-delà de 2,5 le contrôleur suspecte une fuite.
* **Définition du Sharpe** : rendements journaliers (UTC) de l'equity nette, taux sans risque = 0, calculé sur l'ensemble de la période OOS concaténée, annualisé par √365 (BTC coté 7 j/7) — indépendant de la taille de bougie.
* **Drawdown max < 25 %** — le BTC seul subit souvent des baisses > 50 %, on exige nettement mieux.
* **Trades OOS ≥ 200** — en dessous, le Sharpe n'est pas statistiquement exploitable.
* **Folds walk-forward positifs > 60 %** — la stratégie ne doit pas dépendre d'un seul régime de marché.
* **Écart vs baseline : Sharpe net modèle − Sharpe net baseline > 0,3**, mêmes folds et mêmes coûts — sinon on garde la règle simple.
* **Stabilité, frais et slippage ×2 : Sharpe net > 0,5** — les coûts réels dépassent souvent les hypothèses.
* **Stabilité, exécution retardée : exécution à t+2 au lieu de t+1 (une bougie de plus que la règle de base) : Sharpe net > 0,5** — protège contre les gains dépendant d'une exécution instantanée.
* **Stabilité, hyperparamètres ±20 % : baisse du Sharpe ≤ 30 %** — écarte un réglage trop ajusté au passé.

## B. PAPER

* **Durée minimale : 6 semaines** — couvre plusieurs régimes de volatilité.
* **Trades minimum : 50** — base minimale de comparaison.
* **Écart vs rejeu backtest sur la même période : signaux concordants ≥ 95 % et écart de rendement cumulé ≤ 2 points de pourcentage** — vérifie la conformité features / exécution live.

## C. RÉEL

* **Capital max initial : 200 €.**
* **Perte cumulée max : 20 % du capital engagé (40 € sur 200 €)** → arrêt définitif du bot, retour au palier 4.
* **Perte max journalière : 3 % du capital engagé (6 € sur 200 €)** → bot coupé jusqu'au lendemain 00:00 UTC.
* **Risque max par trade : 1 % du capital engagé (2 € sur 200 €)** de perte si le stop est touché.
* **Levier max : 1** (aucun levier, spot uniquement) pendant tout le palier 7, tant que le capital n'a pas été augmenté selon les règles ci-dessous.
* **Conditions pour augmenter le capital (toutes requises)** : ≥ 8 semaines et ≥ 50 trades réels ; Sharpe réel > 0,5 (même définition qu'en A) ; drawdown réel < 15 % ; écart vs paper dans la tolérance B ; hausse au plus ×2 par palier ; chaque hausse soumise à la décision d'Iyad.





\## D. PÉRIMÈTRE DE CALCUL (v2, ajouté par Iyad avant tout résultat de modèle)

\- "Période OOS concaténée" = l'ensemble des périodes de test walk-forward (2021-01 → 2025-09) suivies de la période réservée de 11 mois définie au palier 1. Tous les critères de la section A sans précision de folds s'y calculent : Sharpe net, drawdown max, trades OOS, tests de stabilité.

\- "Folds walk-forward positifs" et "Écart vs baseline" se calculent sur les folds walk-forward uniquement.

\- Garde-fou supplémentaire : sur la période réservée seule, Sharpe net > 0 et drawdown max < 25 %.

\- Buy \& hold : affiché sur chaque période à titre informatif, non éliminatoire.

\- Aucun seuil de la v1 n'est modifié.

