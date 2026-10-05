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

## Brief P3-B5 : bougies 4 h (horizon plus long), E29-E39
Pré-enregistrement : configurations `experiments/configs/E29.json` … `E39.json` et code (`agg4h.py`, `b5.py`,
`b5_report.py`, `corr_b5.py`, `pool_b5.py`) commités ensemble AVANT toute exécution sur les données réelles ;
seuls des auto-tests sur données synthétiques ont tourné avant (`b5.py --selftest`, qui appelle
`agg4h.selftest`). Aucune configuration ajoutée ou modifiée après un résultat.
- Bougies 4 h, features, labels, projection 1 h : docstring de `agg4h.py`. Décision à la clôture de la bougie 4 h,
  exécution à l'open de la bougie 4 h suivante (signal projeté sur la grille 1 h à T+3h, bot.backtest inchangé,
  exec_delay 1 h). Purge = H+1 bougies 4 h (horizon 4H+3 h passé à bot.split).
- Features « f4 » (14, sans calendrier, déclaré a priori : P3-A3 a montré un artefact jour de semaine).
- E29-E31 : label net de coûts H = 6 / 12 / 30 bougies 4 h (24 h, 48 h, 5 j).
- E32-E34 : triple barrière H = 6 / 12 / 30, barrières ±1 x vol_42 (4 h) x sqrt(H).
- E35-E36 : triple barrière H = 12 / 30 à seuils FIXES a priori v = 0,027683 x sqrt(4H/24) (log), 0,027683 =
  médiane pré-2021 de vol_168 x sqrt(24) publiée en P3-A3 (C1d).
- E37-E38 : triple barrière vol H = 12 / 30, « direction seule » (expiration -> NaN).
- Tous LightGBM (n_jobs=1, deterministic), grille hp de E10 avec min_child_samples / 4 (250, 750 ; 4 fois moins de
  lignes), 8 points ; mapping par quantiles de P3-B4 (entrée Q 0,6..0,9 ; sortie Q entrée - 0,1..0,3 ; détention min
  1/2/4 x H bougies 4 h), 36 points ; 288 couples par fold. Seed 42.
- E39 : procédure emboîtée sur l'union E29-E38 seulement (validation commune H=30, blobs git vérifiés).
- Registre : essais lancés en parallèle par `pool_b5.py` (9 workers max, 1 thread chacun) sans écrire au registre ;
  lignes ajoutées ensuite une par une par un seul processus (`b5.py --register EXX`), sans « | » dans les cellules
  (contrôle : 19 séparateurs par ligne). Budget : registre <= 40 lignes (refus au-delà).
- Tests du veto et batterie : `b5_report.py` (importe tests/adversarial/analyse_e10.py sans le modifier) ;
  variantes de contrôle `b5.py CONFIG --variant ...` (hors registre) ; corrélations `corr_b5.py` (pas un essai).
Résultat (brief P3-B5, 11 essais E29-E39, registre 39 lignes) : tous au-dessus de la baseline (-0,5297).
Règle JOURNAL : candidat = E36 (TB H=30 bougies 4 h à seuils fixes a priori ±0,061901 log, Sharpe net OOS
concaténé +1,1751, 48 trades, 8/9 folds > 0 ; B&H +0,7815). Tests du veto (`b5_report.py`, `results/E*_veto.txt`) :
E36 T1 z +4,17 (0/10 seeds >= E36), T2 percentile 98,9 (stratifié 99,8), T7 DSR (N = 39) 0,0456 < 0,95.
Aucun des 11 essais n'a un DSR >= 0,95. Batterie E36 : `results/E36_veto.txt` (variantes `results/E36_<var>.json`),
reproductibilité `results/repro/E36_run2.json` (seul le champ commit diffère). Corrélations : `results/corr_b5.txt`.

## Brief P3-B7 : apport du funding (bougies 4 h), E40-E47
Pré-enregistrement : configurations `experiments/configs/E40.json` … `E47.json` et code (`b7.py`, `b7_report.py`,
`fund4h_check.py`, `corr_b7.py`, `pool_b7.py`) commités ensemble AVANT toute exécution sur les données réelles ; seul
l'auto-test sur données synthétiques a tourné avant (`b7.py --selftest`). Aucune configuration ajoutée ou modifiée après
un résultat. `src/bot/funding.py` est utilisé sans modification (load_funding() par défaut, verrouillé à DEV_END).
- Funding sur la grille 4 h (docstring de `b7.py`) : valeur à la bougie 4 h T = `funding_features` (grille 1 h) à T+3h,
  soit le dernier règlement avec fundingTime <= T + 4 h (clôture de T). Tests : `b7.py --selftest` / `fund4h_check.py`
  (règle, bornes, anti-fuite, direct = lot synthétique et dev réel, couverture par fold).
- Features de funding au modèle, jeu « fu » fixé a priori : fr_cur, fr_mean3, fr_mean21, fr_z90. Exclues a priori :
  fr_sum21 (= 21 x fr_mean21) et fr_age_h (fonction de l'heure : feature calendaire, exclue comme dans f4).
- Lignes : toutes les configurations n'utilisent que les lignes où fu est défini (à partir de ~2020-01-31), y compris
  les versions « f4 seul » : chaque paire a les mêmes lignes, hp, mapping et période de train.
- LABELS REPRIS DE P3-B5, CHOISIS AU VU DE L'OOS DE P3-B5 (déclaré ; le DSR utilise N = lignes du registre) :
  triple barrière fixe H=30 ±0,061901 (label de E36), net de coûts H=30 (E31), triple barrière vol H=6 (E32).
- E40 / E41 : TB fixe H=30, f4 seul / f4 + fu. E42 / E43 : net H=30, f4 / f4 + fu. E44 / E45 : TB vol H=6, f4 / f4 + fu.
  E46 : fu seul, label de E36. E47 : procédure emboîtée sur E41, E43, E45 (validation commune H=30, comme E39).
  Grille hp et mapping de P3-B5 (8 hp x 36 mappings = 288 couples par fold), LightGBM n_jobs=1 deterministic, seed 42.
- Registre : essais lancés en parallèle par `pool_b7.py` (9 workers max, 1 thread chacun) sans écrire au registre ;
  lignes ajoutées ensuite une par une (`b7.py --register EXX`), 19 « | » par ligne, budget registre <= 49.
- Tests du veto pour chaque essai éligible (`b7_report.py --id EXX`) : T1 labels mélangés 10 seeds ; T2 aléatoire
  mêmes trades / même exposition, 1000 tirages, grille 1 h (analyse_e10) ET grille 4 h non stratifiée et stratifiée par
  fold (fonctions de tests/adversarial/analyse_e36.py) : FAILLE si l'un des percentiles <= 95 ; T7 DSR, N = lignes du
  registre. Candidat (`--full`) : batterie de P3-B5 (b5_report --full) + barrières x0,8 / x1,2 (FAILLE si Sharpe <= 0
  ou < 0,5 S) + placebo de funding fplac1..3 (taux permutés par blocs de 30 jours ; FAILLE si la moyenne >= 0,75 S) +
  essai apparié sans funding. Corrélations : `corr_b7.py`. Paires et tableaux : `b7_report.py --pairs`.
- Règle de choix inchangée (JOURNAL.md) : candidat = Sharpe net OOS concaténé le plus élevé parmi E40-E47, s'il dépasse
  la baseline (-0,530) ; sinon aucun candidat.
Résultat (brief P3-B7, 8 essais E40-E47 sur 10 autorisés, registre 47 lignes) : tous au-dessus de la baseline (-0,5297).
Règle JOURNAL : candidat = E40 (f4 SEUL, TB fixe H=30, lignes où fu est défini ; Sharpe net OOS concaténé +1,1139,
58 trades, 7/9 folds > 0, fold 1 sans position ; B&H +0,7815). Le meilleur essai est donc une version SANS funding.
Paires « f4 + fu » - « f4 seul » (concaténé) : E41-E40 -0,2350 ; E43-E42 +0,3216 ; E45-E44 +0,3096 (`results/b7_pairs.txt`).
Veto (`results/E*_veto.txt`) : E40 T1 z +1,715 (0/10 seeds >= E40), T2 percentiles 1 h 97,4 / 1 h strat. 98,4 / 4 h 98,2 /
4 h strat. 98,1, T7 DSR (N = 47) 0,0439 < 0,95. Aucun des 8 essais n'a un DSR >= 0,95. Placebo de funding (informatif,
sur E41 et E47) : E41 moyenne 0,966 (> E41 0,879) ; E47 moyenne 0,566 (E47 0,895). Reproductibilité :
`results/repro/E40_run2.json` (seul le champ commit diffère). Corrélations : `results/corr_b7.txt`.
Écart à l'exécution : correction du formatage de la ligne de registre pour un Sharpe de fold NaN (commit 5620099,
sans effet sur les calculs ; les JSON des essais avaient été produits au commit de pré-enregistrement db7f4d5).

## Brief P3-B9 : cycle « règles », 8 règles pré-enregistrées sans tuning, E48-E55
Pré-enregistrement : configurations `experiments/configs/E48.json` … `E55.json` et code (`rules.py` moteur + familles +
évaluation + registre, `rules_report.py` tests du veto + batterie + choix du candidat, `pool_rules.py` lanceur borné)
commités ensemble AVANT toute exécution sur les données réelles ; seuls les auto-tests sur synthétique ont tourné avant
(`rules.py --selftest`, `rules_report.py --selftest`). Aucune modification de conception ensuite (une correction de bug
serait déclarée avec la preuve qu'elle ne change pas la définition).
- Définitions (moteur, barrières, familles, valeurs manquantes) : docstring de `rules.py`. Tests du veto, seuils des
  drapeaux, règle de choix du candidat : docstring de `rules_report.py`.
- Aucun tuning, aucun paramètre choisi sur nos données ; pas de seed (règles déterministes). Seeds des tests :
  T1-règles 1..100, T2 20261004 / 20261005 (analyse_e10), bootstrap 9.
- E48 A fixe, E49 A vol, E50 B20 fixe, E51 B20 vol, E52 B55 fixe, E53 B55 vol, E54 C fixe, E55 C vol.
- Registre : essais lancés sans écrire au registre ; lignes ajoutées ensuite une par une (`rules.py --register EXX`,
  refus si code modifié depuis le résultat, si le registre n'a pas exactement EXX-1 lignes, ou hors E48-E55), 19 « | »
  par ligne. N du DSR = lignes du registre (cumulé, décision d'Iyad) = 55 à la fin du brief.
