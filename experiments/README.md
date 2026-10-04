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
Résultat (brief P3-B3, 1 essai E17, aucune correction E18-E21) : Sharpe net OOS concaténé +0,2359
(baseline -0,5297, B&H +0,7815), 315 trades, 6/9 folds > 0. Règle JOURNAL : E17 est le seul essai
du brief et dépasse la baseline -> candidat du brief. Réserves mesurées (`e17_report.py`) : labels
mélangés 10 seeds moyenne +0,174 (z de E17 +0,185) ; aléatoire mêmes trades/expo percentile 66,2 ;
B&H fractionnaire même expo 0,789 ; DSR (N = 17) 0,000006.

## Brief P3-B4 : labels nets de coûts et triple barrière (consigne d'Iyad), E18-E28
Pré-enregistrement : configurations `experiments/configs/E18.json` … `E28.json` et code
(`labels_b4.py`, `b4.py`, `corr_b4.py`, `b4_report.py`) commités ensemble AVANT toute exécution
sur les données réelles ; seuls des auto-tests sur données synthétiques ont tourné avant
(`b4.py --selftest`). Aucune configuration ajoutée ou modifiée après un résultat.
- Labels (`labels_b4.py`, docstring) : net de coûts y=1 si log(open[t+1+H]/open[t+1]) > log(1,003) ;
  triple barrière aux ouvertures, haute = max(barrière, log(1,003)), basse, temporelle H ; purge H+1.
- E18-E20 : net H=4/12/24, base ; E21 : net H=24 base+extra ; E22 : idem avec élagage Spearman
  |rho| > 0,9 calculé sur le train du fold ; E23 : TB H=12 ±1 sigma168·sqrt(H) ; E24 : TB H=24
  ±1 sigma·sqrt(H) base ; E25 : idem base+extra ; E26 : idem élagué ; E27 : TB H=24 fixe ±2 %.
- Tous LightGBM, grille hp de E10 (8 points, choisie d'après l'historique P3-B2, déclaré), mapping
  par quantiles des probas du modèle sur ses lignes d'ajustement (entrée Q 0,6/0,7/0,8/0,9 ; sortie
  Q entrée - 0,1/0,2/0,3 ; détention min 1/2/4 x H), soit 8 x 36 = 288 couples par fold.
- E28 : procédure emboîtée sur l'union E18-E27 seulement (validation commune H=24, comme E17).
- Registre : les essais tournent en parallèle sans écrire au registre ; les lignes sont ajoutées
  ensuite une par une par un seul processus (`b4.py --register EXX`).
- Corrélations (`corr_b4.py`, pas un essai) ; tests du veto (`b4_report.py`, importe
  tests/adversarial/analyse_e10.py sans le modifier).
Résultat (brief P3-B4, 11 essais E18-E28, registre 28 lignes) : tous au-dessus de la baseline
(-0,5297). Règle JOURNAL : candidat = E24 (TB H=24 ±1 sigma168·sqrt(24), base, Sharpe net OOS
concaténé +0,7994, 242 trades, 7/9 folds > 0 ; B&H +0,7815). Tests du veto (`b4_report.py`,
`results/E*_veto.txt`) : E24 T1 z +4,49 (0/10 seeds >= E24), T2 percentile 98,3 (stratifié 96,3),
T7 DSR (N = 28) 0,0033 < 0,95. Aucun des 11 essais n'a un DSR >= 0,95.
