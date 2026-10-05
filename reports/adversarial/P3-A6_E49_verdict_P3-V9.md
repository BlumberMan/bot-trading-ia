# Archive : rapport adversarial P3-A6 (candidat E49, cycle « règles ») et verdict associé P3-V9

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B9 ; rapport du chercheur : P3-R9 ; sorties brutes de l'adversarial : commits 5db0b8c, cb79e1d.

---

RAPPORT ADVERSARIAL (réf. candidat E49)

Contexte vérifié : HEAD de départ f021b4b. `git diff gonogo-v2 HEAD -- GONOGO.md` est vide (rc 0). Les critères FAILLE / RÉSISTE ont été commités avant toute exécution (5db0b8c, reports/adversarial/criteres_E49.txt). Avant ce commit, j'avais déjà lu les sorties du chercheur, ce que le fichier déclare. Les seuils reprennent ceux de criteres_E40.txt.
Scripts et sorties brutes : commit cb79e1d. Ni tag ni push. `git status --porcelain` est vide.
Données : load_dataset() par défaut uniquement. Garde « index max = 2025-09-30 23:00 UTC » affichée dans chaque sortie. 9 processus au plus, OMP/MKL/OPENBLAS = 1.

Commandes (depuis C:\Users\2\bot-trading-ia) :
- `.venv\Scripts\python experiments\rules.py experiments\configs\E49.json --out reports\adversarial\E49_adv_repro.json`
- `.venv\Scripts\python tests\adversarial\adv_e49.py t1 | t2 | battery | t8 | t9`
- t9 se lance avec `ADV_SCRATCH=<scratchpad>` : rules.py de ba68dde y est extrait, hors du dépôt.

Sorties : C:\Users\2\bot-trading-ia\reports\adversarial\E49_adv_{repro,t1,t2,battery,t8,t9}.{txt,json}. Script : C:\Users\2\bot-trading-ia\tests\adversarial\adv_e49.py.

S = Sharpe reproduit = +0,606844, 242 trades.

T0 Reproduction | S +0,606844 (écart < 1e-6 avec le déclaré), 242 trades, DD 66,35 %, cohérence moteur / bot.backtest True | RÉSISTE | « concat 2021-25 | règle | 0.607 | 66.35% | 242 | … ; Sharpe OOS concaténé règle 0.6068 ; baseline -0.5297 ; B&H 0.7815 »

T1(r) Reconstruction des séries | identité : Sharpe +0,606843931 avec make_synthetic (écart 1,89e-14) et avec ma réimplémentation indépendante (écart 2,55e-15). Sur les seeds 101-200 : écart maximal entre make_synthetic et ma version 6,86e-14, NaN identiques, multiset des log-rendements conservé | RÉSISTE | « [r] reconstruction : écart max make_synthetic vs mienne 6.86e-14 ; NaN identiques True ; multiset conservé True -> RÉSISTE »

T1(s) T1-règles (seeds 101-200, blocs de 720) | moyenne +0,2261, écart-type 0,3309, p95 +0,7771, 11/100 séries ≥ S, z +1,151. Pool de 200 séries (avec les seeds 1-100 du chercheur) : p95 +0,8087, 24/200 séries ≥ S | FAILLE TROUVÉE | « [s] 100 séries (101..200) : moyenne +0.2261 sd 0.3309 min -0.6330 max +1.1516 p95 +0.7771 ; NaN 0 ; séries >= S 11/100 ; z +1.151 -> FAILLE TROUVÉE »

T2a Aléatoire, grille 1 h, mêmes 242 segments, exposition 0,5185 (seed 4901, 1000 tirages) | S au 92,9e percentile ; p95 aléatoire +0,6864 | FAILLE TROUVÉE | « T2a grille 1 h (seed 4901, 1000) : trades 242..242 ; expo 0.5184..0.5187 ; moyenne +0.1626 sd 0.3160 p50 +0.1647 p95 +0.6864 max +1.1161 ; S percentile 92.9 »
Note : la fonction bt_from_h appliquée aux positions réelles redonne +0,606012, contre S = +0,606844 (écart dû aux opens NaN). Cela ne change pas la conclusion.

T2b Aléatoire stratifié par fold (seed 4902, 1000 tirages) | S au 58,5e percentile ; p95 +1,0422 | FAILLE TROUVÉE | « T2b stratifié par fold (seed 4902, 1000) : moyenne +0.5467 sd 0.2911 p50 +0.5562 p95 +1.0422 ; S percentile 58.5 »

T2c B&H à exposition égale (fraction constante 0,5185) | Sharpe +0,7898, supérieur à S +0,6068 | FAILLE TROUVÉE | « T2c B&H fractionnaire constant 0.5185 : Sharpe +0.7898 rdt +1.5310 ; S +0.6068 -> FAILLE TROUVÉE »

T3a Exécution à t+2, moteur re-simulé | +0,2110 (35 % de S), 233 trades | FAILLE TROUVÉE | « T3a t+2 moteur re-simulé : Sharpe +0.2110 ( 35 % de S) trades 233 rdt +0.0398 »

T3b Exécution à t+2, cibles figées | +0,6067 (100 % de S) | RÉSISTE | « T3b t+2 cibles figées (exec_delay=2) : Sharpe +0.6067 ( 100 % de S) trades 242 »

T3c Features retardées d'une bougie journalière | +0,1781 (29 % de S), 246 trades | FAILLE TROUVÉE | « T3c features retardées d'1 jour (moteur t+1) : Sharpe +0.1781 ( 29 % de S) trades 246 rdt -0.0288 »

T4 Coûts ×2 | +0,2024 (33 % de S, inférieur au seuil de 0,5 S) | FAILLE TROUVÉE | « T4 coûts x2 : Sharpe +0.2024 ( 33 % de S) rdt +0.0266 DD 0.7282 »

T4 Coûts ×3 | −0,2023 | FAILLE TROUVÉE | « T4 coûts x3 : Sharpe -0.2023 ( -33 % de S) rdt -0.5031 DD 0.8274 »

T5a Années | 2 années à Sharpe < 0 (2022 et 2025) ; 4 années sur 5 sous le B&H | FAILLE TROUVÉE | « 2021 +1.6793 | B&H +0.9777 ; 2022 -2.0233 | -1.2928 ; 2023 +0.9212 | +2.3425 ; 2024 +0.7762 | +1.7510 ; 2025 -0.4485 | +0.8293 ; années Sharpe < 0 : 2 ; années Sharpe <= B&H : 4/5 »

T5b Régimes | Sharpe range −1,7392, rendement composé en baisse −73,4 % | FAILLE TROUVÉE | « hausse : 27 mois | Sharpe +3.0176 … ; baisse : 18 mois | Sharpe -2.5564 composé -0.7343 ; range : 12 mois | Sharpe -1.7392 composé -0.4522 -> FAILLE TROUVÉE »

T6a Sans l'horizon dominant (vote 2 sur 2) | horizon dominant L = 120 : sans lui, +0,4135 (68 % de S) | RÉSISTE | « sans L=20 (60,120) >=2 : +0.7950 ; sans L=60 (20,120) >=2 : +0.4722 ; sans L=120 (20,60) >=2 : +0.4135 ; [info] L=20 seul +0.0210, L=60 seul +0.6315, L=120 seul +0.2560 ; vote >=1 : +0.2090 / +0.0606 / +0.3710 »
Observation chiffrée : retirer L = 20 fait passer le Sharpe à +0,795, au-dessus de S. L = 60 seul donne déjà +0,63.

T6b Sans le meilleur mois (2021-01) | +0,3693 (61 % de S) ; sans les 3 meilleurs mois : −0,0288 (information) | RÉSISTE | « meilleurs mois : 2021-01 +0.5403, 2021-03 +0.4648, 2024-02 +0.3038 ; Sharpe +0.3693 ; [info] sans 3 meilleurs -0.0288 »

T7 DSR, recalcul indépendant avec N = 55 | DSR 0,003356 ; SR0 annualisé 1,8434 ; PSR(0) 0,9083. Mêmes valeurs avec V calculée depuis sr_daily et depuis Sharpe/√365 | FAILLE TROUVÉE | « (i) sr_daily : N 55 ; V 1.740711e-03 ; E[max] 2.312597 ; SR0 journalier 0.096486 (ann. 1.8434) ; SR 0.031764 ; T 1734 ; skew 0.4472 ; kurt 9.2599 ; DSR 0.003356 ; PSR(0) 0.908305 »

T8 Fuite | Statique : aucun accès à la période réservée sur le chemin d'exécution (allow_holdout n'apparaît que dans les définitions de src/bot/split.py et src/bot/funding.py). Dynamique : 30 instants t0 sur 30, signaux, b, cibles et sorties décidées jusqu'à t0 sont inchangés, et le test n'est pas vide. Dernier open lu : 2025-09-30 23:00. La décision prise à la dernière clôture n'est pas exécutée | RÉSISTE | « 30/30 inchangés <= t0 : True ; test non vide : True -> RÉSISTE » ; « backtest invariant à la décision de la dernière clôture : True »

T9 Antériorité | ba68dde est bien ancêtre de HEAD. Aucun résultat E48-E55 dans l'arbre de ba68dde ni dans son historique. E49.json déclare ba68dde avec code_dirty False. Aucun écart entre ba68dde et HEAD sur src, harness, rules_report et les configs. 60e7a4a ne modifie que le corps de registry_line (hunk -U0 « -463 +463,5 », dans les lignes 461-491). rules.py de ba68dde, exécuté hors arbre, donne les mêmes cibles, les mêmes 341 trades et le même Sharpe (+0,606843930915), avec 242 trades OOS dans les deux cas. La ligne 49 du registre est égale à registry_line(E49.json) | RÉSISTE | « cibles identiques True ; trades identiques True (341) ; Sharpe +0.606843930915 vs +0.606843930915 ; ligne 49 == registry_line : True -> RÉSISTE »
Deux constats hors critère :
- Lancé à HEAD, register() refuserait E49, car le diff entre ba68dde et HEAD contient experiments/rules.py. L'enregistrement s'est donc fait avec la correction de 60e7a4a présente dans l'arbre de travail mais pas encore commitée. Le garde-fou ne compare que des commits.
- Mon premier contrôle automatique du périmètre du diff s'appuyait sur l'en-tête du hunk et a affiché FAILLE à tort. Je l'ai corrigé (analyse -U0 et ast) avant le commit cb79e1d, et la sortie committée est celle corrigée.

BR Barrières de prix b ×0,8 | −0,0394 | FAILLE TROUVÉE | « BR b x0,8 : Sharpe -0.0394 ( -7 % de S) trades 283 »

BR Barrières de prix b ×1,2 | +0,4721 (78 % de S) | RÉSISTE | « BR b x1,2 : Sharpe +0.4721 ( 78 % de S) trades 216 »

BR Barrière temporelle 96 h | +0,1201 (20 % de S) | FAILLE TROUVÉE | « BR temps 96 h : Sharpe +0.1201 ( 20 % de S) trades 269 »

BR Barrière temporelle 144 h | +0,6889 (114 % de S) | RÉSISTE | « BR temps 144 h : Sharpe +0.6889 ( 114 % de S) trades 224 »

LTa Bootstrap par trade, rendement moyen (seed 4904, 10 000 tirages) | IC95 [−0,00246 ; +0,01146] ; médiane du trade −0,00005 ; P(moyenne ≤ 0) = 0,0961 | FAILLE TROUVÉE | « (a) 242 trades ; moyenne +0.00457 médiane -0.00005 ; IC95 moyenne [-0.00246 ; +0.01146] ; P(moy <= 0) 0.0961 »

LTb Bootstrap circulaire par blocs de 20 jours, Sharpe (seed 4905, 10 000 tirages) | IC95 [−0,3828 ; +1,5893] ; P(Sharpe ≤ 0) = 0,1164 | FAILLE TROUVÉE | « (b) … IC95 [-0.3828 ; +1.5893] ; P(Sharpe <= 0) 0.1164 »

LTc Sans le meilleur trade | +0,5118 (84 % de S) | RÉSISTE | « retirés 2021-01-28 +0.1895 »

LTc Sans les 3 meilleurs trades | +0,3686 (61 % de S) | RÉSISTE | « retirés 2021-01-28 +0.1895, 2021-02-15 +0.1366, 2021-02-06 +0.1343 »

LTc Sans les 5 meilleurs trades | +0,2294 (38 % de S) ; les 5 trades retirés datent tous de janvier à mars 2021 | FAILLE TROUVÉE | « … 2021-03-05 +0.1290, 2021-03-01 +0.1246 -> FAILLE TROUVÉE »

PH Heure de décision (bougies journalières décalées) | à 23:00 UTC, +0,6068. Autres heures : 19:00 +0,1544 ; 15:00 +0,0458 ; 11:00 +0,0828 ; 07:00 +0,3546 ; 03:00 +0,0053. Médiane des 5 autres +0,0828, sous 0,5 S (+0,3034). L'heure 23:00 donne le meilleur des 6 résultats | FAILLE TROUVÉE | « 5 phases : min +0.0053 médiane +0.0828 (0,5 S = +0.3034) -> FAILLE TROUVÉE »

F1 Sans le fold 1 (2021-07-01 → 2025-09-30) | +0,0916 (rendement −10,6 %), contre +0,7828 pour le B&H | FAILLE TROUVÉE | « 2021-07-01 -> 2025-09-30 : Sharpe +0.0916 trades 213 rdt -0.1063 | B&H +0.7828 -> FAILLE TROUVÉE »

Seeds utilisées : T1 101 à 200 ; T2a 4901 ; T2b 4902 ; T8 4903 (puis 4903+i pour chaque coupure) ; LTa 4904 ; LTb 4905. Toutes les autres variantes sont déterministes.
Commits : 5db0b8c (critères, avant exécution) ; cb79e1d (script tests/adversarial/adv_e49.py et sorties brutes).

Conclusion : 18 failles trouvées sur 31 tests :
- T1(s) ;
- T2a, T2b, T2c ;
- T3a, T3c ;
- T4 ×2 et T4 ×3 ;
- T5a, T5b ;
- T7 ;
- BR ×0,8 et BR 96 h ;
- LTa, LTb, LTc sans les 5 meilleurs trades ;
- PH ;
- F1.

13 tests résistent : T0, T1(r), T3b, T6a, T6b, T8, T9, BR ×1,2, BR 144 h, LTc sans le meilleur trade, LTc sans les 3 meilleurs trades, et deux contrôles de reproduction et de reconstruction inclus dans T0 et T1(r).

---

VERDICT P3-V9 | rapport évalué : P3-R9
STATUT : INVALIDE
Score : C0 10/10 ; C3 9/12

Points KO bloquants :
[C3-07 / VETO T1] Veto absolu, T1-règles (remplace les labels mélangés). Constat : FAILLE sur E49. Le rapport donne p95 +0,810 et 13/100 séries ≥ S. Le rapport adversarial, sur les seeds 101-200, donne p95 +0,777 et 11/100 séries ≥ S. Elle concerne aussi les 6 autres essais éligibles. Preuve attendue : S > p95 sur la batterie de séries mélangées. Le veto ne laisse aucune justification recevable.
[VETO T2] Veto absolu, stratégie aléatoire. Constat : E49 au 90,2e percentile sur la grille 1 h et au 59,0e en stratifié. L'adversarial donne 92,9 et 58,5. Aucun essai éligible ne dépasse le 95e percentile. Preuve attendue : les deux percentiles > 95.
[C3-12 / VETO T7] Veto absolu, Sharpe déflaté. Constat : DSR = 0,003356 pour N = 55, avec SR0 annualisé 1,8434 contre S 0,6068 ; statistique −2,7108. Recalculé par moi à partir des valeurs intermédiaires : cohérent. C'est une FAILLE au sens du test 7. Preuve attendue : un DSR qui passe le test 7.
[C3-10] Les FAILLES TROUVÉES du rapport adversarial doivent être expliquées par une preuve. Constat : 18 failles sur 31 tests, dont T1(s), T2a/b/c, T3a, T3c, T4 ×2/×3, T5a/b, T7, BR ×0,8, BR 96 h, LTa/b/c-top5, PH et F1. Aucune n'est levée par une preuve. Preuve attendue : chaque faille expliquée par une preuve. Pour T1, T2 et T7, aucune explication n'est recevable.

Points NON VÉRIFIÉS : aucun bloquant.
- Je n'ai pas relancé T1, T2 ni la batterie (calcul lourd) : je m'appuie sur les sorties commitées, recoupées par les recalculs indépendants de l'adversarial.
- Le dossier contient un résultat déjà KO, ce qui suffit au verdict.

Points d'attention (non bloquants) :
- A6 : `git diff --stat 548d421 HEAD` n'est plus limité à experiments/ à HEAD. 16 fichiers sont hors de ce dossier : reports/adversarial/* et tests/adversarial/adv_e49.py. Ils viennent des commits adversariaux 5db0b8c et cb79e1d. À f021b4b, le dernier commit de l'implémenteur, le diff est bien limité à experiments/.
- Garde-fou de register(), constat T9 de l'adversarial : les lignes E52 à E55 ont été enregistrées alors que la correction 60e7a4a était dans l'arbre de travail mais pas encore commitée. Le garde-fou ne compare que des commits. Le résultat n'est pas affecté (T9 : même Sharpe et mêmes trades avec le rules.py de ba68dde).
- Écart IS/OOS pour E49 : 1,262 sur 2019-2020 contre 0,607 en OOS. Plausible, pas de soupçon OOS ≥ IS.
- E49 n'a que 5 folds positifs sur 9. Retiré du fold 1, le Sharpe tombe à +0,0916 (adversarial F1). Il reste sous le B&H et sous le B&H à exposition égale (0,7898).
- 4 essais ont un Sharpe concaténé ≤ 0 (E48, E50, E52, E54), et 7 sont éligibles. Le rapport conclut que E49 est le candidat selon la règle du JOURNAL, tout en le déclarant sous VETO. Aucun candidat ne passe le veto.

Vérifications relancées par moi :
- `git diff gonogo-v2 -- GONOGO.md | wc -l` -> 0
- `git rev-parse gonogo-v2` -> 68eeb61, identique à la cible commit (tag non déplacé)
- `git log --oneline 548d421..HEAD` -> 5 commits : ba68dde, 60e7a4a et f021b4b (P3-exp), puis 5db0b8c et cb79e1d (P3-adv)
- `git diff --name-only 548d421 f021b4b | grep -v ^experiments/` -> vide
- `git ls-tree -r ba68dde experiments/results | grep E48-55/rules_` -> vide (pré-enregistrement antérieur aux résultats)
- `git diff --stat ba68dde HEAD -- experiments/configs experiments/rules_report.py src scripts tests/test*` -> vide
- `git diff --stat ba68dde 60e7a4a` -> experiments/rules.py seul, +5/−1
- `grep -c '^| E' experiments/REGISTRE.md` -> 55
- lignes supprimées dans l'historique Git de REGISTRE.md -> 0
- `pytest -q` -> 105 passed, 0 skip
- `rules.py --selftest` -> « auto-tests : OK », 49 lignes [OK]
- `rules.py configs/E49.json --out <scratchpad>` -> garde « index max 2025-09-30 23:00 UTC OK » ; écarts baseline/B&H 0,000e+00 ; JSON identique hors commit à results/E49.json : True ; signal_1h.parquet égal : True
- `git status --porcelain` après exécution -> vide
- `git check-ignore .env` -> .env (ignoré)
- `git grep` sur des motifs de clé en clair -> aucun résultat
- Recalcul manuel du DSR pour E49 (E[max] 2,3126 ; SR0 0,096486 ; dénominateur 0,99392 ; statistique −2,7108 ; Φ = 0,00336) -> cohérent avec le rapport
- Écart baseline au registre : 0,6068 + 0,5297 = 1,1365 -> cohérent

Tableau GONOGO : N/A (palier 3).

Décision Iyad requise : NON au titre de ce verdict, qui n'est pas un point critique. Le cycle « règles » n'a produit aucun essai qui passe le veto. La suite relève de l'orchestrateur et d'Iyad, conformément à JOURNAL.md.

Fichiers relus :
- C:\Users\2\bot-trading-ia\experiments\results\E49_veto.txt
- C:\Users\2\bot-trading-ia\experiments\results\E49.json
- C:\Users\2\bot-trading-ia\experiments\REGISTRE.md
- C:\Users\2\bot-trading-ia\reports\adversarial\E49_adv_*.txt
