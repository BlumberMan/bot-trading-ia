# Archive : rapport adversarial P3-A4 (candidat E36) et verdict associé P3-V5

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B5 ; rapport du chercheur : P3-R5 ; sorties brutes de l'adversarial : commits 0f3d75e, a76ea39.

---

RAPPORT ADVERSARIAL (réf. candidat E36)

Contexte vérifié : HEAD de départ cddbf40. GONOGO.md est identique au tag gonogo-v2 (`git diff gonogo-v2 HEAD -- GONOGO.md` ne renvoie rien). Aucune donnée réservée lue : chaque script vérifie « index max ≤ 2025-09-30 23:00 UTC » et a affiché « [garde] index max lu = 2025-09-30 23:00:00+00:00 (59160 lignes) ».

Commits :
- 0f3d75e : critères FAILLE / RÉSISTE (reports/adversarial/criteres_E36.txt), commités avant toute exécution.
- a76ea39 : scripts et sorties.
- Ni tag ni push. `git status --porcelain` est vide à la fin.

Calcul : au plus 9 workers en même temps, OMP/MKL/OPENBLAS=1, n_jobs=1 de b5. Aucun processus résiduel.

Commandes (racine C:\Users\2\bot-trading-ia) :
- `.venv\Scripts\python tests\adversarial\run_b5_variant.py VAR`, avec VAR = base, shuf11, shuf12, shuf13, shuf14, shuf15, delay2, featlag1, cost2, cost3, drop_vol_180, tbv08, tbv12, rule_vol_180, rule_vol_42, onlyvols, tbdir. Sorties : reports/adversarial/E36_adv_VAR.json/.txt, plus les parquet de signal et de probas pour base, delay2, cost2 et cost3.
- `.venv\Scripts\python tests\adversarial\analyse_e36.py`, sortie dans reports/adversarial/analyse_e36.txt. Seeds : T2a 3636, T2b 3637, T2c 3638, LT(a) 3639, LT(b) 3640.
- `.venv\Scripts\python tests\adversarial\leak_e36.py static` et `... fold K` pour K = 1 à 9 (seed de bruit 31+K ; seed 33 pour le test statique). Sorties : reports/adversarial/leak_e36_static.txt et leak_e36_foldK.txt.
- `.venv\Scripts\python tests\adversarial\anteriorite_e36.py`, sortie dans reports/adversarial/anteriorite_e36.txt.

S = Sharpe reproduit = 1,175146. Seuils dérivés : 0,5 S = 0,5876 ; 0,7 S = 0,8226 ; 0,75 S = 0,8814 ; 1,10 S = 1,2927.

**T0 Reproduction** | Sharpe 1,175146 (écart 4,9e-7), 48 trades, mêmes choix (hp, seuils) que le chercheur dans les 9 folds | RÉSISTE
Sortie brute :
`Sharpe 1.175146 (JSON 1.175146) trades 48 expo 0.456267 DD 0.5472 rdt total 5.9174 ; déclaré 1.175146 : |écart| 4.87e-07`
`choix (hp, seuils) par fold identiques au chercheur : [True x9]`

**T1 Labels mélangés** (seeds 11 à 15 ajoutées aux 10 seeds du chercheur) | pool de 15 seeds : p95 +1,0422, z +2,333. Mes 5 seeds seules : moyenne +0,6568, z +1,272, max +1,0582 (90 % de S) | RÉSISTE
Sortie brute :
`avocat seed 11 : +1.0353 trades 97 | seed 12 : +0.1217 | seed 13 : +0.6817 | seed 14 : +1.0582 trades 120 | seed 15 : +0.3873`
`pool 15 seeds : moyenne +0.4344 écart-type 0.3175 p95 +1.0422 max +1.0582`
`E36 +1.1751 : z +2.333 ; seeds >= E36 : 0/15 ; S > p95 : True ; seeds >= B&H 0.7815 : 2/15`

**T2a Aléatoire, grille 1 h** (1000 tirages, seed 3636) | percentile 99,1 | RÉSISTE
Sortie brute : `trades tirés min/max 48/48 ; Sharpe moyenne 0.4215 p50 0.4195 p95 0.9802 max 1.5428 ; E36 percentile 99.1`

**T2b Aléatoire, grille 4 h** (durées de détention réelles permutées, alignées sur les opens 4 h ; 1000 tirages, seed 3637) | percentile 98,9 | RÉSISTE
Sortie brute :
`blocs 4 h OOS 10404 ; segments 4 h 48 ; blocs exposés 4747 (expo 0.4563) ; contrôle rebacktest 1.1751 trades 48`
`trades tirés min/max 48/48 ; expo 0.4562/0.4563 ; Sharpe aléatoire 4 h : moyenne 0.3995 p95 0.9530 max 1.5539 ; E36 percentile 98.9 ; tirages >= B&H : 133/1000`

**T2c Aléatoire 4 h stratifié par fold** (1000 tirages, seed 3638) | percentile 99,8 | RÉSISTE
Sortie brute : `trades tirés min/max 48/49 ; Sharpe moyenne 0.4260 p50 0.4228 p95 0.8505 max 1.1941 ; E36 percentile 99.8`

**T2d B&H à la même exposition** | 0,7908, contre S = 1,1751 | RÉSISTE
Sortie brute : `B&H fractionnaire constant 0.4563 : Sharpe 0.7908 rdt +1.3209 DD 0.4611`

**T3 Retards** | signal figé exécuté une bougie 4 h plus tard : 0,9973 (85 % de S) ; figé à +1 h : 1,1926 (101 %) ; delay2 re-tuné : 0,5918 (50,4 %, juste au-dessus du seuil de 0,5876) ; featlag1 re-tuné : 0,7937 (68 %) | RÉSISTE
Sortie brute :
`exec +1 bougie 4 h (delay 5 h) : Sharpe +0.9973 (85 % de S) trades 48`
`delay2 : Sharpe +0.5918 (50 % de S) trades 45 folds>0 7/9 ; Sharpe/fold 0.501;1.107;-0.616;0.636;1.289;1.398;1.676;0.305;-0.090`
`featlag1 : Sharpe +0.7937 (68 % de S) trades 61 folds>0 7/9`

**T4 Coûts x2 et x3** | figé : 1,1040 et 1,0325 ; re-tuné : 1,1208 et 0,8430 | RÉSISTE
Sortie brute : `cost3 : Sharpe +0.8430 (72 % de S) trades 38 ; Sharpe/fold 0.493;1.286;-0.566;1.041;1.183;1.166;1.546;3.332;-0.272`

**T5a Par année** | aucune année négative ; Sharpe ≤ B&H 2 années sur 5 (2021 et 2023) | RÉSISTE
Sortie brute :
`2021 +0.7332 (B&H +0.9777) | 2022 +0.2241 (B&H -1.2928) | 2023 +1.3562 (B&H +2.3425) | 2024 +2.4383 (B&H +1.7510) | 2025 +1.8894 (B&H +0.8293)`

**T5b Par régime** | range : Sharpe −0,1733 (≤ 0) ; baisse : rendement composé −52,96 % (≤ −50 %) ; hausse : +2,9391 | FAILLE TROUVÉE
Sortie brute :
`hausse : 27 mois | E36 Sharpe +2.9391 rdt composé +15.1031`
`baisse : 18 mois | E36 Sharpe -0.9268 rdt composé -0.5296`
`range : 12 mois | E36 Sharpe -0.1733 rdt composé -0.0868`

**T6a Sans la meilleure feature** (vol_180, re-tuné) | 0,5323 (45 % de S, sous le seuil de 0,5 S) | FAILLE TROUVÉE
Sortie brute : `drop_vol_180 : Sharpe +0.5323 (45 % de S) trades 60 DD 0.7384 folds>0 6/9 ; Sharpe/fold 0.201;-0.295;-1.911;-0.311;1.738;2.084;2.087;2.235;0.501`

**T6b Sans le meilleur mois** (2024-11) | 1,0316 (88 % de S) | RÉSISTE
Sortie brute : `Sharpe sans les jours du meilleur mois : 1.0316 (88 % de S) ; sans les 3 meilleurs mois : 0.7821`

**T7 DSR** (recalcul indépendant, N = 39) | DSR 0,045570, sous le seuil de 0,95 | FAILLE TROUVÉE
Sortie brute :
`N = 39 ; V = 2.162807e-03 ; SR0 annualisé 1.9364 ; SR E36 0.061510 (1.1751) ; T 1734 ; skew 0.7869 ; kurtosis 14.2492 ; DSR = 0.045570 ; PSR(0) = 0.995444`
`(sensibilité) V sur E04-E39 -> DSR 0.869170 ; V sur E29-E39 -> DSR 0.794196`
`(info) 53 variantes de stratégie évaluées sur le même OOS hors registre ; N_eff = 92 -> DSR 0.010072`

**T8 Fuite** | (a) les 16 motifs relevés sont les définitions et gardes de src/bot/split.py et src/bot/data.py, plus des lectures de parquet internes à b5_report.py ; aucun accès au holdout. (b) 9 folds sur 9 identiques. (c) 0 écart sur 20 pour chaque contrôle ; label réimplémenté identique. (d) purge correcte | RÉSISTE
Sortie brute :
`(b) fold 1..9 RÉSULTAT : True (champs différents [] ; probas identiques True)`
`(c1) ... labels <= t modifiés : 0/20 ; (c2) bougies 0/20 ; features 0/20 ; (c3) égal à agg4h.make_label_4h : True`
`(d) pire cas {"inner_label_end_minus_val_start_h": -4.0, "final_label_bar_end_minus_test_start_h": 0.0, "val_open_max_minus_test_start_h": -124.0} ; RÉSULTAT : True`

**T9 Antériorité de a0ddea6** | 0 écart | RÉSISTE
Sortie brute :
`a0ddea6 ancêtre de 2db2f81 : True`
`configs E29-E39, agg4h.py, b5.py, b4.py, harness.py -> OK ; diff src : aucun`
`commits <= a0ddea6 contenant un résultat E29..E39 : []`
`E29..E39.json commit a0ddea6 code_dirty False ; GONOGO.md vs gonogo-v2 : identique ; T9 RÉSULTAT : écarts 0`

**LT Faible nombre de trades** | rendement moyen par trade : IC95 [+0,0091, +0,1014] ; Sharpe par bootstrap de trades : IC95 [+0,2313, +2,1803] ; sans le meilleur trade : 0,9565 (81 %) ; sans les 3 meilleurs : 0,5922 (50,4 %, juste au-dessus du seuil de 0,5876) | RÉSISTE
Pour information, hors critère :
- les 3 meilleurs trades font 63,7 % du log-P&L ;
- sans les 5 meilleurs trades, le Sharpe tombe à 0,2785 ;
- P(Sharpe bootstrap ≤ B&H) = 0,21 ;
- seuils GONOGO du palier 4 : 48 trades contre ≥ 200 exigés, DD 54,72 % contre < 25 % exigé.

Sortie brute :
`(a) moyenne/trade IC95 [+0.0091, +0.1014] ; P(moyenne <= 0) 0.0082`
`(b) médiane 1.2030 IC95 [+0.2313, +2.1803] ; P(Sharpe <= 0) 0.0053 ; P(Sharpe <= B&H 0.7815) 0.2103 ; P(Sharpe <= 1,0) 0.3534`
`(c) sans 1 : +0.9565 ; sans 3 : +0.5922 (50 % de S) ; sans 5 : +0.2785 (24 % de S)`

**NA Features NaN** | 11,68 % des décisions OOS ont des features NaN. Les heures figées font 15,00 % de l'exposition et −6,31 % du P&L. La variante « NaN → plat » donne 1,3857 (118 % de S) | RÉSISTE (info : selon le traitement des NaN, S varie de +18 %)
Sortie brute :
`features NaN 1215 (11.68 %) ; NaN ret_180/vol_180 1215`
`heures exposées 18988 ; figées 2848 (15.00 %) ; log-P&L figé -0.1221 / total +1.9340 (-6.31 %)`
`variante NaN -> plat : Sharpe +1.3857 (118 % de S) trades 50 DD 0.3234`

**BR Barrières ±20 %** (re-tuné) | v × 0,8 : 0,9462 (81 % de S) ; v × 1,2 : 0,3506 (30 % de S, soit une baisse de 70 % pour une tolérance GONOGO de 30 %) | FAILLE TROUVÉE
Sortie brute :
`tbv08 : Sharpe +0.9462 (81 % de S) trades 49 folds>0 8/9`
`tbv12 : Sharpe +0.3506 (30 % de S) trades 61 DD 0.6433 folds>0 5/9 ; Sharpe/fold 0.616;1.049;-2.418;-0.533;-0.343;2.317;-0.277;1.755;0.330`

**C1a Règles sans apprentissage** | rule_vol_180 : 0,8957 (76,2 % de S, au-dessus du seuil de 0,75 S) avec 6 trades ; rule_vol_42 : 0,7872 (67 %) | FAILLE TROUVÉE
Sortie brute : `rule_vol_180 : Sharpe +0.8957 (76 % de S) trades 6 expo 0.251 DD 0.2915 folds>0 4/9`

**C1b LightGBM sur [vol_42, vol_180] seules** | 0,4152 (35 % de S) | RÉSISTE
Sortie brute : `onlyvols : Sharpe +0.4152 (35 % de S) trades 46 folds>0 6/9`

**C1c Direction seule** (expiration → NaN) | 0,3823 (33 % de S, sous le seuil de 0,5 S) | FAILLE TROUVÉE
Sortie brute : `tbdir : Sharpe +0.3823 (33 % de S) trades 37 DD 0.7325 folds>0 5/9`

**C1d Décomposition de l'AUC** | AUC(direction | une barrière touchée) = 0,5374, au-dessus du seuil de 0,52 | RÉSISTE
Pour information : la feature brute +vol_42 a une AUC(label) de 0,5845, au-dessus de celle du modèle (0,5628).
Sortie brute :
`AUC(label) 0.5628 ; AUC(hit) 0.5500 ; AUC(direction | hit) 0.5374 (n 3807)`
`règle +vol_42 : AUC(label) 0.5845 AUC(hit) 0.6626`

**C1e Seuil fixe 0,027683** | recalcul 0,0276829 (écart 5,1e-8), uniquement sur les données antérieures à 2021 ; v = 0,061901 | RÉSISTE
Pour information : l'idée des barrières fixes vient du test adversarial tbfixed publié en P3-A3, mesuré sur ce même OOS (0,5609), et non compté dans N.

**C2 Placebo calendaire** | sans objet : le jeu f4 ne contient aucune feature calendaire, il n'y a donc rien à remplacer par un placebo.

Conclusion : 6 failles trouvées (T5b, T6a, T7, BR, C1a, C1c). Les 18 autres tests résistent, dont 2 à moins de 0,005 du seuil : T3 delay2 re-tuné (0,5918 pour un seuil de 0,5876) et LT sans les 3 meilleurs trades (0,5922 pour un seuil de 0,5876). C2 est sans objet.

---

VERDICT P3-V5 | rapport évalué : P3-R5
STATUT : INVALIDE

Score : C0 9/10 (C0-07 non relancé de bout en bout, voir plus bas) ; C1 6/6 dans le périmètre testé ; C3 11/12 (KO sur C3-12, avec veto T7)

Points KO bloquants :
[C3-12 / VETO T7] Règle : le Sharpe déflaté du candidat doit être fourni et positif (DSR ≥ 0,95, N = lignes du registre). Une FAILLE sur le test 7 est un veto absolu.
- Constat : les deux rapports (implémenteur et avocat du diable) déclarent T7 en FAILLE pour E36, avec DSR = 0,045570 pour N = 39 et SR−SR0 = −0,039848. Je retrouve ces valeurs dans experiments/results/E36_veto.txt:63 et reports/adversarial/analyse_e36.txt:121.
- Les 11 essais ont un DSR < 0,95.
- Preuve attendue : un candidat dont le DSR, calculé avec N = lignes du registre, atteint au moins 0,95.

[C3-10] Règle : chaque FAILLE TROUVÉE du rapport adversarial doit être expliquée par une preuve.
- Constat : 6 failles ne sont pas levées : T5b (range −0,17 ; baisse −52,96 %), T6a (sans vol_180 : 45 % de S), T7, BR (barrières ×1,2 : 30 % de S), C1a (rule_vol_180 : 76 % de S) et C1c (direction seule : 33 % de S).
- Le rapport P3-R5 les reconnaît sans fournir de preuve qui les lève.
- Preuve attendue : une preuve pour chaque faille, ou un rapport adversarial sans FAILLE.

Points NON VÉRIFIÉS :
[C0-07] Je n'ai pas relancé b5.py E36 de bout en bout. J'ai seulement comparé E36.json et repro/E36_run2.json (seule la clé commit diffère). L'avocat du diable a relancé de son côté (écart 4,9e-7). Ce point ne change pas le statut.

Points d'attention (non bloquants) :
- Rapport adversarial : 53 variantes ont été évaluées sur le même OOS hors registre (N_eff = 92, DSR 0,010). Le seuil fixe 0,027683 vient d'un test adversarial (P3-A3) mesuré sur ce même OOS. La sélection n'est donc pas aveugle à l'historique de recherche.
- E36 : 48 trades au total, entre 2 et 15 par fold ; les 3 meilleurs trades portent 63,7 % du log-P&L.
- Deux tests résistent à moins de 0,005 du seuil : T3 delay2 (0,5918 pour 0,5876) et LT sans les 3 meilleurs trades (0,5922).
- Selon le traitement des NaN, S varie de +18 % (11,7 % des décisions OOS sont figées).
- Il reste 1 essai au registre (39/40) pour ce brief et 21 sur 60 pour le palier 3.

Vérifications relancées par moi :
- `git diff gonogo-v2 -- GONOGO.md | wc -l` -> 0
- `git rev-parse gonogo-v2` -> 68eeb61c…
- `git log c2e09f9..HEAD` -> a0ddea6 (20:22:50), puis 2db2f81 (20:29:45), cddbf40, 0f3d75e, a76ea39 : le pré-enregistrement précède bien les résultats
- `git log --format=%h -- configs/E29..E39.json` -> a0ddea6 pour les 11 fichiers
- `git diff --name-only c2e09f9 cddbf40` hors experiments/ -> 0 fichier
- `git diff --name-only cddbf40 HEAD` hors reports/adversarial et tests/adversarial -> 0 fichier
- `git diff --numstat c2e09f9 HEAD -- REGISTRE.md` -> 11 ajouts, 0 suppression
- `grep -c '^| E[0-9]' REGISTRE.md` -> 39, égal au N déclaré
- lignes `-| E` dans l'historique du registre -> 0 (aucune ligne supprimée)
- `pytest -q -rs` -> 72 passed, 0 skip
- `git status --porcelain` -> vide
- `git check-ignore .env` -> .env
- comparaison E36.json et repro/E36_run2.json -> seule la clé commit diffère
- grep DSR dans E36_veto.txt et analyse_e36.txt -> DSR 0,045570, N = 39, FAILLE

Tableau GONOGO : N/A (palier 3)

Décision Iyad requise : NON
