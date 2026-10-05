# Archive : rapport adversarial P3-A5 (candidat E40) et verdict associé P3-V7

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B7 ; rapport du chercheur : P3-R7 ; sorties brutes de l'adversarial : commits b89cbc0, 4edd66b.

---

RAPPORT ADVERSARIAL (réf. candidat E40)

**Contexte vérifié**
- HEAD de départ : cdf934c.
- GONOGO.md est identique au tag gonogo-v2 (`git diff gonogo-v2 HEAD -- GONOGO.md` ne renvoie rien).
- REGISTRE.md compte 47 lignes (E01 à E47).
- Aucune donnée réservée n'a été lue. Chaque script affiche :
  - prix : « index max lu = 2025-09-30 23:00 (59160 lignes) » ;
  - funding : « fundingTime max lu = 2025-09-30 16:00 (6300 règlements) ».
- Pendant mon run, le superviseur a commité 9d5be1b, qui ne modifie que JOURNAL.md (1 ligne) et aucun fichier de code.

**Commits**
- b89cbc0 : critères FAILLE / RÉSISTE (reports/adversarial/criteres_E40.txt), commités avant toute exécution.
- 4edd66b : scripts et sorties brutes.
- Ni tag ni push. `git status --porcelain` est vide. Aucun processus python résiduel.

**Calcul**
- Au plus 9 workers en même temps (xargs -P 9), OMP/MKL/OPENBLAS = 1, n_jobs = 1.
- Environ 30 s par relance complète.

**Commandes** (racine C:\Users\2\bot-trading-ia)
- `.venv\Scripts\python tests\adversarial\run_b7_variant.py VAR`. Sorties : reports/adversarial/E40_adv_VAR.json et .txt, plus des parquet pour base, delay2, cost2, cost3, start2019-01-01 et e36seed42. VAR vaut :
  - base ;
  - shuf21 à shuf25 (critère) et shuf26 à shuf35 (info) ;
  - delay2, featlag1, cost2, cost3 ;
  - drop_vol_180, tbscale0.8, tbscale1.2 ;
  - rule_vol_180, rule_vol_42, tbdir, onlyvols ;
  - start2019-01-01, start2019-07-01, start2019-10-01, start2020-01-30, start2020-04-01, start2020-07-01 (critère) ;
  - start2020-01-30T04:00, T08:00, T12:00, T16:00, T20:00, start2020-01-31T00:00, start2020-02-01T00:00, start2020-03-01T00:00 (info) ;
  - seed43 à seed47 ;
  - e36seed42 à e36seed47.
- `.venv\Scripts\python tests\adversarial\analyse_e40.py`, sortie dans reports/adversarial/analyse_e40.txt. Seeds : T2a 4040, T2b 4041, T2c 4042, LT(a) 4043, LT(b) 4044.
- `.venv\Scripts\python tests\adversarial\leak_e40.py static` et `... fold K` pour K = 1 à 9.
  - Bruit sur les prix : seed 41+K ; bruit sur le funding : seed 141+K.
  - Test statique : seeds 43 et 44.
  - Sorties : reports/adversarial/leak_e40_static.txt et leak_e40_fold_K.txt.
- b7.py de db7f4d5 a été extrait par `git show` et exécuté hors arbre : `.venv\Scripts\python tests\adversarial\run_b7_db7f4d5.py <b7_db7.py> <out.json>`. La sortie est copiée dans reports/adversarial/E40_adv_db7f4d5.json.
- `.venv\Scripts\python tests\adversarial\anteriorite_e40.py reports\adversarial\E40_adv_db7f4d5.json`, sortie dans reports/adversarial/anteriorite_e40.txt.

**Seuils**
S = 1,113918 (Sharpe de E40 reproduit). Seuils dérivés :

| Seuil | Valeur |
|---|---|
| 0,5 S | 0,5570 |
| 0,7 S | 0,7797 |
| 0,75 S | 0,8354 |
| 1,10 S | 1,2253 |

---

**T0 Reproduction** | Sharpe 1,113918 (écart 1,1e-7), 58 trades, mêmes choix que le chercheur dans les 9 folds | RÉSISTE
Sortie brute :
`Sharpe 1.113918 (JSON 1.113918) trades 58 expo 0.427528 DD 0.4677 rdt total 3.9827 ; déclaré 1.113918 : |écart| 1.14e-07`
`choix (hp, seuils) par fold identiques au chercheur : [True x9]`

**T1 Labels mélangés** (seeds 21 à 25, nouvelles) | pool de 15 seeds : p95 +0,8511, z +2,064. Mes 5 seeds seules : z +3,280 | RÉSISTE
Sortie brute :
`avocat seed 21 : +0.6504 | 22 : +0.3347 | 23 : +0.4702 | 24 : +0.6098 | 25 : +0.1537`
`pool 15 seeds : moyenne +0.5034 écart-type 0.2959 p95 +0.8511 max +0.9058`
`E40 +1.1139 : z +2.064 ; seeds >= E40 : 0/15 ; S > p95 : True ; seeds >= B&H 0.7815 : 3/15`
`(info) seeds 26..35 ... pool 25 seeds : moyenne +0.5168 sd 0.2538 p95 +0.8224 ; z +2.353 ; seeds >= S : 0/25`

**T2a Aléatoire, grille 1 h** (1000 tirages, seed 4040) | percentile 98,0 | RÉSISTE
Sortie brute : `trades tirés min/max 58/58 ; Sharpe moyenne 0.3765 p50 0.3706 p95 0.9553 max 1.6278 ; E40 percentile 98.0`

**T2b Aléatoire, grille 4 h** (1000 tirages, seed 4041) | percentile 97,5 | RÉSISTE
Sortie brute : `contrôle ... Sharpe 1.1139 trades 58 ; expo 0.4275/0.4276 ; Sharpe aléatoire 4 h : moyenne 0.4030 p95 0.9769 max 1.5044 ; E40 percentile 97.5 ; tirages >= B&H : 140/1000`

**T2c Aléatoire 4 h stratifié par fold** (1000 tirages, seed 4042) | percentile 97,9 | RÉSISTE
Sortie brute : `trades tirés min/max 57/58 ; Sharpe moyenne 0.4408 p95 0.9363 max 1.5491 ; E40 percentile 97.9`

**T2d B&H à la même exposition** | 0,7913, contre S = 1,1139 | RÉSISTE
Sortie brute : `B&H fractionnaire constant 0.4275 : Sharpe 0.7913 rdt +1.2248 DD 0.4382`

**T3 Retards** | aucune mesure ne s'effondre ni ne dépasse 1,10 S | RÉSISTE
- Signal figé exécuté une bougie 4 h plus tard : 1,0820 (97 % de S).
- Signal figé à +1 h : 1,1337 (102 %).
- delay2 re-tuné : 1,0256 (92 %).
- featlag1 re-tuné : 0,7504 (67 %).

Sortie brute :
`exec +1 bougie 4 h (delay 5 h) : Sharpe +1.0820 (97 % de S) trades 58`
`delay2 : Sharpe +1.0256 (92 % de S) trades 53 folds>0 7/9`
`featlag1 : Sharpe +0.7504 (67 % de S) trades 43 folds>0 8/9 ; Sharpe/fold 0.508;0.724;-1.379;0.447;0.876;2.714;2.264;2.081;0.131`

**T4 Coûts x2 et x3** | signal figé : 1,0127 et 0,9113 ; re-tuné : 1,0127 et 0,8715 (78 %) | RÉSISTE
Sortie brute : `cost3 : Sharpe +0.8715 (78 % de S) trades 52 folds>0 6/9 ; Sharpe/fold nan;2.441;-1.148;0.795;0.268;3.013;1.879;0.987;-0.078`

**T5a Par année** | une année négative (2022 : −0,35) ; Sharpe ≤ B&H 2 années sur 5 (2023 et 2025) | RÉSISTE
Sortie brute :
`2021 +1.8877 (B&H +0.9777) | 2022 -0.3452 (B&H -1.2928) | 2023 +1.9832 (B&H +2.3425) | 2024 +1.9211 (B&H +1.7510) | 2025 +0.1502 (B&H +0.8293)`

**T5b Par régime** | range : Sharpe −0,9129 (≤ 0) | FAILLE TROUVÉE
Sortie brute :
`hausse : 27 mois | E40 Sharpe +2.6040 rdt composé +7.6239 | B&H +3.6113`
`baisse : 18 mois | E40 Sharpe -0.3214 rdt composé -0.2378`
`range : 12 mois | E40 Sharpe -0.9129 rdt composé -0.2420 | B&H +0.1208`

**T6a Sans la meilleure feature** (vol_180, re-tuné) | 0,7825 (70 % de S) | RÉSISTE
Sortie brute : `drop_vol_180 : Sharpe +0.7825 (70 % de S) trades 71 DD 0.6482 folds>0 7/9`

**T6b Sans le meilleur mois** (2021-10) | 0,9432 (85 % de S) | RÉSISTE
Pour information, sans les 3 meilleurs mois : 0,6663.
Sortie brute : `Sharpe sans les jours du meilleur mois : 0.9432 (85 % de S) ; sans les 3 meilleurs mois : 0.6663`

**T7 DSR** (recalcul indépendant, N = 47) | DSR 0,043878, sous le seuil de 0,95 ; SR − SR0 = −0,040689 | FAILLE TROUVÉE
Sortie brute :
`N = 47 lignes (ids E01..E47) ; V = 1.931504e-03 ; SR0 journalier 0.098995 (annualisé 1.8913)`
`SR journalier E40 0.058305 (annualisé 1.1139) ; T 1734 ; skew 0.4712 ; kurtosis 14.8231 ; DSR = 0.043878 ; PSR(0) = 0.992788`
`(sensibilité) V sur E04-E47 -> DSR 0.796456 ; V sur E29-E47 -> DSR 0.746931 ; V sur E40-E47 -> DSR 0.752679`
`(info) variantes de stratégie évaluées sur le même OOS hors registre : 108 ; N_eff = 155 -> DSR 0.006248`

**T8 Fuite** (prix et funding) | RÉSISTE
- (a) Statique : 26 motifs relevés. Ce sont les définitions et gardes de src/bot/split.py, data.py et funding.py, plus des lectures de parquet internes aux rapports. b7.py appelle `load_funding()` sans argument (2 appels) et ne contient pas `allow_holdout`.
- (b) Dynamique : 9 folds sur 9 identiques, avec les prix ET les taux de funding postérieurs à test_start empoisonnés.
- (c) Labels et features : 0 écart sur 20 pour chaque contrôle, y compris le funding et le masque row_filter. Le label réimplémenté est identique.
- (d) Purge correcte.

Sortie brute :
`(b) fold 1..9 RÉSULTAT : True` (fold 1 : `opens 1 h modifiés 41661 ; taux de funding modifiés 5202 ; champs différents [] ; train_start 2020-01-30 12:00 n_train 1014 ; probas identiques`)
`(c1) 0/20 ; (c2) bougies 0/20 features 0/20 ; (c4) features funding <= T modifiées : 0/20 ; masque row_filter <= T modifié : 0/20 ; (c3) True`
`(d) pire cas {"inner_label_end_minus_val_start_h": -4.0, "final_label_bar_end_minus_test_start_h": 0.0, "val_open_max_minus_test_start_h": -124.0} ; RÉSULTAT : True`

**T9 Antériorité de db7f4d5 et portée de 5620099** | 0 écart | RÉSISTE
- 5620099 modifie une seule ligne, dans registry_line, qui n'est appelée que par register().
- b7.py de db7f4d5 et b7.py de HEAD donnent un Sharpe identique au bit près.

Sortie brute :
`db7f4d5 ancêtre de 5620099 : True ; 5620099 ancêtre de a32fb19 : True ; parent de 5620099 : db7f4d5`
`configs E40..E47, agg4h.py, b5.py, b4.py, harness.py -> OK ; diff src : aucun ; commits touchant b7.py après db7f4d5 : ['5620099']`
`5620099 numstat : 1 1 experiments/b7.py ; ligne modifiée 243 dans la fonction : registry_line ; appels de registry_line : ['register'] ; autres fichiers : []`
`commits <= db7f4d5 (82) contenant un résultat E40..E47 : [] ; E40.json : commit db7f4d5… code_dirty False variante base -> OK`
`b7.py@db7f4d5 : Sharpe 1.113918114 trades 58 ; b7.py@HEAD : 1.113918114 trades 58 ; |écart| 0.0e+00 ; choix par fold identiques : True ; T9 RÉSULTAT : écarts 0`

**LT Faible nombre de trades** | FAILLE TROUVÉE
- (a) Rendement moyen par trade : IC95 [+0,0055, +0,0673].
- (b) Sharpe par bootstrap de trades : IC95 [+0,1696, +2,0223].
- (c) Sans le meilleur trade : 0,9012 (81 %).
- (c) Sans les 3 meilleurs : **0,5322 (48 % de S, sous le seuil de 0,5570)**.
- Pour information : les 3 meilleurs trades font 64,7 % du log-P&L ; P(Sharpe bootstrap ≤ B&H) = 0,249 ; GONOGO palier 4 : 58 trades contre ≥ 200 exigés, DD 46,77 % contre < 25 % exigé.

Sortie brute :
`(a) moyenne/trade IC95 [+0.0055, +0.0673] ; P(moyenne <= 0) 0.0102`
`(b) médiane 1.1055 IC95 [+0.1696, +2.0223] ; P(Sharpe <= 0) 0.0094 ; P(Sharpe <= B&H 0.7815) 0.2493`
`(c) sans 1 : +0.9012 (81 %) ; sans 3 [2021-09-24 0.4946, 2023-09-29 0.3852, 2021-07-24 0.3647] : +0.5322 (48 % de S) ; sans 5 : +0.2796 (25 %)`

**NA Features NaN** | les heures figées font 12,25 % de l'exposition et 25,06 % du P&L (> 20 % et > 2 × 12,25 % = 24,5 %). La variante « NaN → plat » donne 0,9310 (84 %) | FAILLE TROUVÉE (marge de 0,56 point)
Sortie brute :
`features NaN 1215 (11.68 %) ; NaN ret_180/vol_180 1215`
`heures exposées 17792 ; figées 2180 (12.25 %) ; log-P&L figé +0.4024 / total +1.6060 (part du P&L 25.06 %)`
`variante NaN -> plat : Sharpe +0.9310 (84 % de S) trades 59`

**BR Barrières ±20 %** (re-tuné) | v × 0,8 : 0,7841 (70,4 %) ; v × 1,2 : 0,4877 (44 % de S, sous le seuil de 0,7 S) | FAILLE TROUVÉE
Sortie brute : `tbscale1.2 : Sharpe +0.4877 (44 % de S) trades 42 folds>0 6/9 ; Sharpe/fold 0.582;1.075;-1.468;1.256;nan;2.146;1.071;2.189;-0.531`

**C1a Règles sans apprentissage** | rule_vol_180 : 0,8358 (75,03 % de S, au-dessus du seuil de 0,8354) avec 9 trades ; rule_vol_42 : 0,5033 | FAILLE TROUVÉE (marge de 0,0004)
Sortie brute : `rule_vol_180 : Sharpe +0.8358 (75 % de S) trades 9 expo 0.527 DD 0.6833 folds>0 7/9`

**C1b LightGBM sur [vol_42, vol_180] seules** | 0,7166 (64 %) | RÉSISTE
Sortie brute : `onlyvols : Sharpe +0.7166 (64 % de S) trades 33 folds>0 7/9`

**C1c Direction seule** | 0,1744 (16 % de S) | FAILLE TROUVÉE
Sortie brute : `tbdir : Sharpe +0.1744 (16 % de S) trades 30 DD 0.6390 folds>0 5/9`

**C1d Décomposition de l'AUC** | AUC(direction | une barrière touchée) = 0,5376, au-dessus du seuil de 0,52 | RÉSISTE
Pour information : la feature brute +vol_42 a une AUC(label) de 0,5845, au-dessus de celle du modèle (0,5695).
Sortie brute : `AUC(label) 0.5695 ; AUC(hit) 0.5615 ; AUC(direction | hit) 0.5376 (n 3807)`

**ST Point de départ du train** (re-tuné, row_filter retiré) | départ 2019-07-01 : 0,4524 (41 % de S) ; départ 2020-04-01 : 0,7734 (69 %). Les deux sont sous le seuil de 0,7 S | FAILLE TROUVÉE

Les départs testés :

| Départ | Sharpe | Part de S | Statut |
|---|---|---|---|
| 2019-01-01 | 1,1751 | 105 % | contrôle, égal à E36 |
| 2019-07-01 | 0,4524 | 41 % | critère |
| 2019-10-01 | 0,9048 | 81 % | critère |
| 2020-01-30 00:00 | 1,3161 | 118 % | contrôle |
| 2020-04-01 | 0,7734 | 69 % | critère |
| 2020-07-01 | 0,9714 | 87 % | critère |

Le départ 2020-01-30 00:00 ajoute seulement 3 lignes de train avant la première ligne de E40 (12:00) et donne 1,3161 au lieu de 1,1139.

Pour information, hors critère (sonde ajoutée après lecture de ce résultat) : 8 départs entre 2020-01-30 00:00 et 2020-02-01, soit 1017 à 1005 lignes de train au fold 1.
- Sharpe : 1,3161, 0,8739, 0,8537, **1,1139**, 0,6949, 0,8793, 0,6878, 0,5941.
- Médiane 0,8638, écart-type 0,2386. S est 2e sur 8.

Sortie brute :
`start2019-07-01 : Sharpe +0.4524 (41 % de S) trades 63 ; Sharpe/fold 0.601;0.524;-2.485;-1.075;-0.062;2.907;2.353;1.662;0.845`
`critère (2019-07, 2019-10, 2020-04, 2020-07) : min +0.4524 ; seuil 0,7 S 0.7797 -> FAILLE`
`8 départs 2020-01-30 00:00 -> 2020-02-01 : moyenne +0.8767 médiane +0.8638 sd 0.2386 min +0.5941 max +1.3161 ; rang de S 2/8`

**SD Bruit de seed LightGBM et comparaison E40 / E36** | RÉSISTE
- E40, seeds 43 à 47 : moyenne 0,8784 (79 % de S), minimum 0,7395.
- Seed 42 est le meilleur des 6 seeds, pour E40 comme pour E36.
- Écart E40 − E36 : d = −0,0612, contre une tolérance 1,96 × √(sd40² + sd36²) = 0,4368. L'écart est **dans le bruit**.
- Rapporté au bruit des labels mélangés : |d| / sd du pool (0,2538) = 0,24. Écart des moyennes sur 6 seeds : +0,0334.

Sortie brute :
`E40 seeds 42..47 : [1.1139, 0.9914, 1.0254, 0.8888, 0.7468, 0.7395] ; moyenne 43..47 +0.8784 (79 % de S) ; sd 0.1532`
`E36 seeds 42..47 : [1.1751, 0.8829, 0.8877, 0.7034, 0.7674, 0.889] ; moyenne +0.8842 sd 0.1619`
`d = -0.0612 ; 1,96 x sqrt(sd40² + sd36²) = 0.4368 -> dans le bruit : True ; rang de seed 42 : E40 1/6, E36 1/6`

**F1 Fold 1 à plat** | recalcul à partir du fold 2 : E40 1,1772, contre B&H 0,7828 et E36 1,4937 | RÉSISTE
Pour information : au fold 1, E40 a 0 trade, alors que le B&H fait +20,8 % (Sharpe 0,87). Ce fold représente 10,4 % des jours OOS. Le Sharpe concaténé sans le fold 1 serait de 1,1772.
Sortie brute : `à partir du fold 2 (2021-07-01 -> fin) : E40 Sharpe +1.1772 trades 58 DD 0.4677 ; B&H +0.7828 ; E36 +1.4937 (E36 concat complet +1.1751) -> critère F1 RÉSISTE`

**C2 Placebo funding** | sans objet : E40 n'utilise aucune feature de funding.

---

Conclusion : 8 failles trouvées (T5b, T7, LT, NA, BR, C1a, C1c, ST). Les 17 autres tests résistent ; C2 est sans objet. Deux failles passent le seuil de peu : C1a de 0,0004 et NA de 0,56 point.
- T9 : le pré-enregistrement db7f4d5 précède bien les résultats, et le correctif 5620099 ne change aucun calcul.
- E40 contre E36 : l'écart (1,114 contre 1,175) reste dans le bruit des seeds.
- Point de départ du train : le résultat n'est pas stable. Décaler le départ de quelques bougies 4 h fait varier le Sharpe de 0,59 à 1,32.

Toutes les sorties brutes sont dans C:\Users\2\bot-trading-ia\reports\adversarial\ (analyse_e40.txt, leak_e40_*.txt, anteriorite_e40.txt, E40_adv_*.json/.txt, criteres_E40.txt). Les scripts sont dans C:\Users\2\bot-trading-ia\tests\adversarial\ (run_b7_variant.py, analyse_e40.py, leak_e40.py, anteriorite_e40.py, run_b7_db7f4d5.py).

---

VERDICT P3-V7 | rapport évalué : P3-R7

STATUT : INVALIDE

Score : C0 10/10 ; C3 10/12

Points KO bloquants :
- [C3-12 / VETO T7] Veto absolu : FAILLE sur le test 7. Le candidat E40 a un DSR de 0,043878 pour N = 47 lignes au registre, SR0 annualisé 1,8913, sous le seuil de 0,95. Le rapport et l'avocat du diable le déclarent tous les deux, et je l'ai revérifié dans les sorties. Aucune justification n'est recevable. Aucun des 8 essais n'atteint un DSR ≥ 0,95. Preuve attendue : un DSR ≥ 0,95 avec N = nombre de lignes du registre.
- [C3-10] Rapport adversarial : 8 FAILLES TROUVÉES sur E40 (T5b régime range −0,913 ; T7 ; LT, sans les 3 meilleurs trades 48 % de S ; NA, 25,06 % du P&L sur des heures figées ; BR, barrières ×1,2 à 44 % de S ; C1a, une règle vol_180 sans apprentissage atteint 75,03 % de S ; C1c, direction seule à 16 % de S ; ST, Sharpe instable selon le point de départ du train, de 0,45 à 1,32). Aucune n'est levée par une preuve dans P3-R7. Le rapport en reconnaît 3 et n'explique pas les 5 autres. Preuve attendue : chaque faille levée par une preuve chiffrée, sinon elle reste.

Points NON VÉRIFIÉS :
- [A2] Je n'ai pas relancé fund4h_check.py (anti-fuite et égalité direct = lot du funding sur la grille 4 h) : son script écrit dans experiments/results/. Les sorties sont commitées, et l'avocat a testé la fuite (T8, 9 folds sur 9 identiques avec prix et funding empoisonnés). Ce point ne change pas le statut.

Points d'attention (non bloquants) :
- Écart interne au rapport adversarial : le texte donne « 108 variantes hors registre ; N_eff = 155 -> DSR 0.006248 », alors que reports/adversarial/analyse_e40.txt donne « 109 ; N_eff = 156 -> DSR 0.006179 ». Le chiffre est informatif, mais le rapport ne reflète pas la sortie commitée.
- `git diff --name-only 1109c79 HEAD` contient aussi JOURNAL.md, reports/adversarial/ et tests/adversarial/. Ils viennent des commits 9d5be1b (superviseur), b89cbc0 et 4edd66b (avocat du diable). Sur le périmètre de l'implémenteur (1109c79..cdf934c), 0 fichier hors experiments/.
- Les labels sont repris au vu de l'OOS de P3-B5. C'est déclaré dans le README et dans chaque ligne du registre, mais cela reste un choix informé par l'OOS.
- Concentration : vol_180 pèse 0,294 du gain, et une règle vol_180 sans apprentissage atteint 75 % de S avec 9 trades.
- Le meilleur essai n'utilise pas le funding. La paire E41 − E40 vaut −0,235, et le placebo de funding de E41 (moyenne 0,966) dépasse E41 lui-même (0,879).
- Fold 1 à 0 trade pour 6 des 8 essais. 58 trades au total pour E40.
- Registre : 47 lignes sur 60 au plafond du palier 3.

Vérifications relancées par moi :
- `git diff gonogo-v2 -- GONOGO.md | wc -c` -> 0 ; gonogo-v2 = 68eeb61 ; tags : gonogo-v1, gonogo-v2, palier-0a, palier-1, palier-2.
- `git check-ignore .env` -> .env.
- `wc -l experiments/REGISTRE.md` -> 47 ; lignes supprimées depuis 1109c79 -> 0 ; 19 « | » sur chacune des lignes 40 à 47.
- `git diff --name-only 1109c79 cdf934c | grep -vc ^experiments/` -> 0 ; 297 fichiers, 189509 insertions, 0 suppression.
- Ordre des commits : db7f4d5 (pré-enregistrement, 14 fichiers, aucun résultat) précède 5620099 puis a32fb19..ee6400d.
- `pytest -q` -> 87 passed in 6.20s, aucun skip dans tests/ hors adversarial.
- Relance de `b7.py experiments/configs/E40.json --out <scratchpad>` -> JSON identique à results/E40.json hors champ commit ; garde à 2025-09-30 23:00 UTC : True ; importance max vol_180 0,2940.
- `grep DSR` dans results/E40_veto.txt et reports/adversarial/analyse_e40.txt -> DSR 0,043878, N 47, « VETO FINAL E40 : T7 FAILLE ».
- `git status --porcelain` -> vide, avant et après ma relance.

Tableau GONOGO (palier 4+) : N/A (palier 3).

Décision Iyad requise : NON
