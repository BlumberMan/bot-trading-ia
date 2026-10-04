# Exploration palier 3 (chercheur ML)

Tout le code d'exploration est ici ; rien n'est en production. `src/` n'est jamais modifié.

## Harnais : `experiments/harness.py`
- Données : `bot.split.load_dataset()` par défaut (période dev seule) + assertion
  « index max <= 2025-09-30 23:00 UTC », affichée à chaque exécution ; second contrôle sur le
  dernier open lu par les backtests.
- Folds : `bot.split.make_folds(horizon=H)` / `split_fold(..., horizon=H)` (purge H+1, embargo 24),
  9 folds de test semestriels 2021-01-01 -> 2025-09-30, train expanding depuis 2019-01-01.
- Tuning interne au train : validation = derniers 20 % des lignes propres du train, train interne
  purgé par `bot.split.purge_mask` ; objectif = Sharpe net journalier (bot.metrics) du backtest
  bot.backtest (exec_delay=1, 0,0015/côté, ×1) de la fenêtre de validation ; NaN -> -inf ;
  premier maximum dans l'ordre de la grille. Réentraînement sur tout le train du fold.
- Mapping proba -> position (long/flat) : entrée si p > seuil, sortie si p <= seuil - écart et
  détention >= min_hold bougies ; écart 0 et min_hold 0 = « 1 si p > seuil sinon 0 ».
  Features NaN -> signal NaN (position conservée).
- Pas de normalisation (modèles à arbres).
- Évaluation : probas des 9 folds concaténées, un backtest continu ; métriques par fold sur les
  fenêtres de test (comme scripts/run_baseline.py) + période concaténée ; baseline SMA168 et B&H
  recalculés et comparés à results/baseline_metrics.json (écart <= 1e-9 exigé, sinon arrêt).
- LightGBM : deterministic=True, n_jobs=1, force_row_wise=True, random_state=seed.
- Sorties : `experiments/results/<ID>.json` (+ `<ID>.txt`, sortie console), sans horodatage
  d'exécution (JSON reproductible à l'identique).

## Features supplémentaires : `experiments/extra_features.py`
Formules, fenêtres et décalages dans la docstring du module (fenêtres finies, données <= t).

## Registre : `experiments/REGISTRE.md`
Une ligne = un essai (une configuration évaluée sur les 9 folds), échecs compris ; le fichier ne
contient AUCUNE ligne d'en-tête, donc `wc -l` = nombre d'essais. Lignes ajoutées par
`harness.py --registry` (refusé si le code n'est pas commité) ou à la main pour un échec ;
jamais modifiées ni supprimées. Colonnes, dans l'ordre :
ID | date | commit du code | seed | modèle | features | horizon H du label | grille hp et
mapping | taille de la grille interne | folds | Sharpe net OOS par fold (1..9) | trades par fold |
Sharpe net OOS concaténé | trades concaténés | écart de Sharpe vs baseline | écart vs B&H |
statut | commande exacte.
Statut écrit au moment de l'essai : « abandonné (<= baseline) » ou « éligible (> baseline) » ;
le candidat final (règle JOURNAL.md : Sharpe concaténé le plus élevé parmi les essais du brief,
s'il dépasse -0,530) est désigné plus bas, sans modifier le registre.

Les tests « labels mélangés » et les relances de reproductibilité ne sont pas des essais
(aucune configuration nouvelle, aucun choix) : fichiers `results/<ID>_shuf<seed>.json`, hors registre.

## Décision finale
Brief P3-B2 : 16 essais (E01-E16) sur 20 autorisés, arrêt volontaire (plateau 0,53-0,67 sur les
variantes régularisées H=24).
Règle appliquée : candidat = E10 (Sharpe net OOS concaténé +0,6695 > baseline -0,5297 sur les
mêmes folds). Configuration : `experiments/configs/E10.json`, résultat `results/E10.json`.
Réserves mesurées (`candidate_report.py`) : test labels mélangés, 5 seeds, Sharpe OOS concaténé
moyen +0,396 (min +0,167, max +0,774), 1 seed sur 5 au-dessus du candidat ; Sharpe déflaté
(N = 16) = 0,0002. La performance du candidat n'est donc pas distinguable de celle d'une
procédure à labels aléatoires avec le même mapping (long/flat à faible rotation).
Aucun essai n'a été choisi sur le test d'un fold ; le registre n'a pas été modifié.

## Brief P3-B3 : E17, procédure emboîtée (`experiments/nested.py`, `experiments/configs/E17.json`)
Choix, dans chaque fold et sur la seule validation interne au train, parmi l'union exacte des
16 espaces E01-E16 (fichiers vérifiés par blob git à 970ea49), puis réentraînement de la
configuration choisie sur tout le train du fold. H multiples : fenêtre de validation commune
définie avec H=24 (purge du plus grand H) ; règles complètes dans E17.json. Commité avant toute
exécution sur les données ; évalué une fois sur l'OOS (1 ligne de registre). Auto-test sur
données synthétiques : `nested.py ... --selftest`. Pièces : `experiments/e17_report.py`
(importe tests/adversarial/analyse_e10.py sans le modifier).
