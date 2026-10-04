# Archive : rapport adversarial P3-A3 (candidat E24) et verdict associé P3-V4

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B4 ; rapport du chercheur : P3-R4 ; sorties brutes de l'adversarial : commits 162ae11, 7838c35.

---

RAPPORT ADVERSARIAL (réf. candidat E24)

Contexte : HEAD de départ f39fabc (vérifié). GONOGO.md identique au tag gonogo-v2 (git diff vide). Données : load_dataset() par défaut uniquement. Garde « index max <= 2025-09-30 23:00 UTC » dans chaque script, sortie : `[garde] index max lu = 2025-09-30 23:00:00+00:00 (59160 lignes)`. Je n'ai rien ajouté à REGISTRE.md et je n'ai modifié ni experiments/ ni src/.
Commits : 162ae11 « P3-adv: critères FAILLE/RÉSISTE E24 fixés avant exécution » (reports/adversarial/criteres_E24.txt, commité avant toute exécution), puis 7838c35 « P3-adv: batterie adversariale E24 … » (scripts et sorties brutes). `git status --porcelain` est vide. Pas de tag ni de push.
Calcul : au plus 9 workers en parallèle, OMP/MKL/OPENBLAS = 1, LightGBM n_jobs=1. Tous les processus sont terminés.
Scripts : tests/adversarial/run_b4_variant.py, analyse_e24.py, leak_e24.py, anteriorite_e24.py.
Commandes :
- `.venv\Scripts\python tests\adversarial\run_b4_variant.py <VAR>` avec VAR = base, shuf11 à shuf15, delay2, featlag1, cost2, cost3, drop_vol_168, only_vol168, rule_vol, tbdir, tbfixed, nodow, dowplac51, dowplac52, dowplac53. Sorties dans reports/adversarial/E24_adv_<VAR>.json/.txt.
- `.venv\Scripts\python tests\adversarial\analyse_e24.py --draws 1000 --seed 2424` (sortie dans reports/adversarial/analyse_e24.txt).
- `.venv\Scripts\python tests\adversarial\leak_e24.py fold K` pour K = 1 à 9, et `leak_e24.py static`.
- `.venv\Scripts\python tests\adversarial\anteriorite_e24.py`.
S = 0,799410, le Sharpe reproduit par ma relance (seed 42).

T0 Reproduction | Sharpe 0,799410, 242 trades, écart 2,57e-07, choix par fold identiques 9/9 | RÉSISTE | `Sharpe 0.799410 (JSON 0.799410) trades 242 expo 0.403282 DD 0.5841 rdt total 2.0014 ; déclaré 0.79941 : |écart| 2.57e-07` ; `choix (hp, seuils) par fold identiques au chercheur : [True x9]`

T1 Labels mélangés (mes seeds 11 à 15, procédure complète) | mes 5 seeds : moyenne −0,0476. Pool de 15 seeds : moyenne +0,0886, écart-type 0,1879, p95 +0,3201. z = +3,783. Aucune seed >= S (0/15) | RÉSISTE | `avocat seed 11 : -0.2309 / 12 : -0.2211 / 13 : +0.1063 / 14 : -0.1236 / 15 : +0.2315` ; `pool 15 seeds : moyenne +0.0886 écart-type 0.1879 p95 +0.3201 max +0.4769` ; `E24 +0.7994 : z +3.783 ; seeds >= E24 : 0/15 ; S > p95 : True`

T2 Aléatoire, mêmes segments et même exposition (1000 tirages, seed 2424) | S au 98,4e percentile, p95 = 0,6676 | RÉSISTE | `trades tirés min/max 241/242 ; Sharpe moyenne 0.0742 p50 0.0709 p95 0.6676 max 1.2153 ; E24 percentile 98.4`

T2b B&H à exposition égale | B&H à fraction constante 0,4033 : Sharpe 0,7917, contre S = 0,7994 (écart +0,0077). Le DD de E24 vaut 58,41 % contre 41,82 % pour ce B&H, et son rendement +200 % contre +114 %. Stratifié par fold (seed 2425) : 97,6e percentile | RÉSISTE, de justesse | `B&H fractionnaire constant 0.4033 : Sharpe 0.7917 rdt +1.1445 DD 0.4182 | E24 Sharpe 0.7994 rdt +2.0014 DD 0.5841` ; `stratifié par fold (seed 2425, 1000 tirages) : moyenne 0.1852 p95 0.7060 ; E24 percentile 97.6`

T3 Retards | signal figé, exécution t+2 : 0,6400 (80 % de S). Re-tuné delay2 : 0,8763 (109,6 % de S ; le seuil « amélioration » de 1,10·S vaut 0,8794). Re-tuné featlag1 : 0,4987 (62 %) | RÉSISTE. Le re-tuné à t+2 fait mieux que la base et passe à 0,003 sous le seuil d'amélioration | `exec_delay=2 : Sharpe +0.6400 (80 % de S)` ; `delay2 : Sharpe +0.8763 (110 % de S) trades 248` ; `featlag1 : Sharpe +0.4987 (62 % de S) trades 271`

T4 Coûts x2 et x3 | signal figé : x2 +0,3957, x3 −0,0059. Re-tuné : x2 +0,3581, x3 +0,0572 (7 % de S) | FAILLE TROUVÉE (Sharpe x3 figé <= 0) | `coûts x2 : Sharpe +0.3957 (50 % de S) trades 242` ; `coûts x3 : Sharpe -0.0059 (-1 % de S) trades 242 DD 0.7389 rdt -0.2983` ; `cost3 : Sharpe +0.0572 (7 % de S) rdt -0.2179`

T5a Par année | une année négative (2022, −1,2054). Sharpe <= B&H sur 3 années sur 5 (2023, 2024, 2025) | RÉSISTE | `2021 : E24 +1.3278 | B&H +0.9777` ; `2022 : -1.2054 | -1.2928` ; `2023 : +1.6962 | +2.3425` ; `2024 : +1.4374 | +1.7510` ; `2025 : +0.8190 | +0.8293`

T5b Par régime (mois civil B&H > +5 % = hausse, < −5 % = baisse, sinon range) | Sharpe en range −0,3072 (<= 0). Rendement composé en baisse −69,3 % (<= −50 %) | FAILLE TROUVÉE | `hausse : 27 mois | E24 Sharpe +2.9830 rdt composé +10.1261 | B&H +3.6113` ; `baisse : 18 mois | E24 Sharpe -1.6748 rdt composé -0.6930 | B&H -2.6954 -0.9486` ; `range : 12 mois | E24 Sharpe -0.3072 rdt composé -0.1212 | B&H +0.1208`

T6a Sans la meilleure feature (vol_168, re-tuné) | 0,0128, soit 2 % de S | FAILLE TROUVÉE | `drop_vol_168 : Sharpe +0.0128 (2 % de S) trades 293 expo 0.542 DD 0.8197 rdt -0.3265`

T6b Sans le meilleur mois (2024-11, +41,4 %) | 0,6170, soit 77 % de S. Sans les 3 meilleurs mois : 0,3190 | RÉSISTE | `Sharpe sans les jours du meilleur mois : 0.6170 (77 % de S) ; sans les 3 meilleurs mois : 0.3190`

T7 Sharpe déflaté, recalcul indépendant (N = 28 lignes de REGISTRE.md) | DSR 0,003317 (seuil 0,95). PSR(0) = 0,9604 | FAILLE TROUVÉE | `N = 28 lignes (ids E01..E28) ; V = 2.714584e-03 ; SR0 journalier 0.106537 (annualisé 2.0354)` ; `SR journalier E24 0.041843 ; T 1734 ; skew 0.5114 ; kurtosis 13.7737 ; DSR = 0.003317 ; PSR(0) = 0.960441` ; sensibilité (variance calculée sur E04 à E28 seulement) : `DSR = 0.710301`

T8 Fuite | (a) statique : aucun appel allow_holdout=True dans experiments/ ni src/ (seulement les définitions dans bot/split.py ; voir leak_e24_grep.txt). (b) Empoisonnement du futur, folds 1 à 9 (seeds 32 à 40) : 9/9 identiques. (c1) Label TB, troncature à t+1+H (seed 33) : 0/20 écart. (c2) vol_168 avec le futur perturbé : 0/20 écart. (d) Purge, pire cas en heures : −1, −1, −25 | RÉSISTE | `(b) fold 1..9 RÉSULTAT : True` ; `(c1) ... labels <= t modifiés : 0/20` ; `(c2) ... vol_168 <= t modifiée : 0/20` ; `(d) pire cas : {"inner_label_end_minus_val_start_h": -1.0, "final_label_end_minus_test_start_h": -1.0, "val_open_max_minus_test_start_h": -25.0}` ; `(d) RÉSULTAT : True`

T9 Antériorité | blobs de E18.json à E28.json, b4.py et labels_b4.py identiques entre e0249a3 et HEAD, aucun commit après e0249a3. Aucun fichier de résultat E18 à E28 dans e0249a3 ni dans ses ancêtres. e0249a3 (17:27:41) est ancêtre de 8faaf2a (17:45:23). src/, harness.py et extra_features.py inchangés | RÉSISTE | `T9 RÉSULTAT : écarts 0` (sortie complète : anteriorite_e24.txt)

C1a Circularité : modèle « vol_168 seule » (même procédure) | −0,0619 | RÉSISTE | `only_vol168 : Sharpe -0.0619 (-8 % de S) trades 102`

C1b Circularité : règle ±vol_168 sans apprentissage (36 mappings, choix sur la validation) | 0,5739, soit 72 % de S (seuil 75 %), avec seulement 33 trades | RÉSISTE, de justesse | `rule_vol : Sharpe +0.5739 (72 % de S) trades 33 expo 0.510 DD 0.5447`

C1c Circularité : label « direction seule » (expirations mises à NaN, 33 234 lignes sur 55 233, soit 60 %) | 0,3229, soit 40 % de S (seuil 50 %) | FAILLE TROUVÉE | `[tbdir] lignes label défini 55233, expirées -> NaN 33234, taux y=1 restant 0.5197` ; `tbdir : Sharpe +0.3229 (40 % de S) trades 173`

C1d Circularité : triple barrière à seuils fixes (r_log = 0,027683, médiane de vol_168·√24 avant 2021) | 0,5609, soit 70 % de S | RÉSISTE | `tbfixed : Sharpe +0.5609 (70 % de S) trades 122`

C1e Décomposition de l'AUC OOS | AUC(label) 0,6275 ; AUC(hit) 0,6321 ; AUC(direction | hit) 0,5455 (>= 0,52). Corrélation de Spearman entre la proba et −vol_168 : 0,4234. L'essentiel du pouvoir discriminant porte sur le fait qu'une barrière soit touchée, pas sur la direction | RÉSISTE au critère | `AUC(label) 0.6275 (n 40295) ; AUC(hit) 0.6321 (taux hit 0.4063) ; AUC(direction | hit) 0.5455 (n 16372)` ; `règle -vol_168 : AUC(label) 0.5525 ; AUC(hit) 0.5469 ; AUC(direction | hit) 0.5282` ; `Spearman(proba E24, -vol_168) OOS : 0.4234`

C2a Sans dow_sin ni dow_cos (re-tuné) | 0,2517, soit 31 % de S | FAILLE TROUVÉE | `nodow : Sharpe +0.2517 (31 % de S) trades 296 expo 0.622 DD 0.7582`

C2b Placebo : jour de semaine aléatoire par jour civil (seeds 51, 52, 53) | 0,6011, 0,1394 et 0,5129 ; moyenne 0,4178, soit 52 % de S (seuil 50 %) | RÉSISTE, de justesse. Un jour tiré au hasard récupère à lui seul jusqu'à 75 % de S (seed 51) | `dowplac51 : +0.6011 (75 % de S)` ; `dowplac52 : +0.1394 (17 %)` ; `dowplac53 : +0.5129 (64 %)` ; `placebo jour de semaine : moyenne +0.4178 (52 % de S)`

Conclusion : 6 failles trouvées (T4, T5b, T6a, T7, C1c, C2a). Quatre tests résistent de justesse : T2b, T3 delay2 re-tuné, C1b et C2b.
Fichiers : C:\Users\2\bot-trading-ia\reports\adversarial\criteres_E24.txt, analyse_e24.txt, anteriorite_e24.txt, leak_e24_*.txt, E24_adv_*.json/.txt ; C:\Users\2\bot-trading-ia\tests\adversarial\run_b4_variant.py, analyse_e24.py, leak_e24.py, anteriorite_e24.py.

---

VERDICT P3-V4 | rapport évalué : P3-R4
STATUT : INVALIDE
Score : C0 8/10 ; C3 9/13

Points KO bloquants :
[VETO T7 / C3-12] Veto absolu (Sharpe déflaté) : le rapport et le rapport adversarial indiquent FAILLE TROUVÉE sur le test 7 pour le candidat E24. DSR = 0,003317 < 0,95 et SR − SR0 = −0,064694 < 0 (E24_veto.txt l.63 ; analyse_e24.txt l.88). Aucun des 11 essais n'a un DSR ≥ 0,95. Ce veto ne peut être levé par aucune justification. Preuve attendue : un DSR positif (SR > SR0) et ≥ 0,95 pour un candidat, avec N = nombre de lignes de REGISTRE.md.
[C3-10] Le rapport adversarial liste 6 FAILLES TROUVÉES : T4 (coûts ×3 sur signal figé −0,0059), T5b (range −0,31 ; baisse −69,3 %), T6a (sans vol_168 : +0,0128), T7, C1c (direction seule : 40 % de S), C2a (sans jour de semaine : 31 % de S). Aucune n'est compensée par une preuve chiffrée : le rapport P3-R4 §10 les reconnaît et ne les explique pas. Preuve attendue : chaque faille levée par une preuve chiffrée.
[C3-08] Une feature dominante n'est pas expliquée. vol_168 (gain 0,265) est aussi l'échelle des barrières du label : la circularité est reconnue, le Sharpe tombe à 2 % de S sans elle, et l'AUC porte sur « barrière touchée » (0,632) plutôt que sur la direction (0,546). Le jour de semaine (dow_sin + dow_cos : 0,288) n'est pas expliqué non plus, et le placebo récupère jusqu'à 75 % de S. Preuve attendue : un pouvoir prédictif démontré qui ne dépende ni de la construction du label ni d'un artefact calendaire.
[C3-09] L'écart IS/OOS n'est pas plausible au fold 1 (Sharpe val 5,04 contre OOS 0,48 ; AUC train 0,769 contre test 0,590), ni aux folds 3 et 4 (val 2,46 et 0,38 contre OOS −0,70 et −1,92). Le rapport le déclare sans l'expliquer. Preuve attendue : une explication chiffrée de la dégradation.

Points NON VÉRIFIÉS :
[C3-07] Labels mélangés : je n'ai pas relancé les 15 seeds. La cohérence entre les sorties déclarées du chercheur (10 seeds) et celles de l'avocat (5 seeds) n'a pas été recontrôlée. Cela reste sans effet sur le statut.

Points d'attention (non bloquants) :
- REGISTRE.md, lignes E22 et E26 : « |rho| » casse le tableau Markdown. Le défaut est déclaré.
- La grille hp de E10 a été conçue en P3-B2 après des résultats OOS (contamination partielle de l'espace de recherche). Le point est déclaré.
- Plusieurs tests ne résistent que de justesse : T2b (+0,0077 contre le B&H à exposition égale), T3 delay2 re-tuné (0,003 sous le seuil), C1b (72 % pour un seuil de 75 %), C2b (52 % pour un seuil de 50 %).
- Écart de E24 contre le B&H : +0,0179 de Sharpe, avec un rendement inférieur (+200 % contre +293 %).
- Budget du palier 3 : 28/60 consommés. C'est le 3e INVALIDE d'affilée sur le palier 3 (P3-V2, P3-V3, P3-V4).

Vérifications relancées par moi :
- `git diff gonogo-v2 -- GONOGO.md | wc -l` -> 0
- `git rev-parse gonogo-v2^{commit}` -> 68eeb61…, tag inchangé
- `git status --porcelain` -> 0 ligne
- `git diff --name-only c84dd5f f39fabc | grep -v ^experiments/` -> 0 fichier hors experiments/ pour les commits P3-exp
- `git diff --numstat c84dd5f HEAD -- experiments/REGISTRE.md` -> 11 ajouts, 0 suppression
- `grep -c '^| E' experiments/REGISTRE.md` -> 28, conforme au N déclaré
- `git log -- experiments/configs/E18..E28.json` -> un seul commit chacun (e0249a3)
- `git log --date=iso` -> e0249a3 (17:27:41) avant 8faaf2a (17:45:23), puis f39fabc ; critères adversariaux 162ae11 avant la batterie 7838c35
- `pytest -q -rs` -> 72 passed in 5.36s, 0 skip
- `git check-ignore .env` -> .env
- comparaison E24.json / repro/E24_run2.json -> seule différence : ['commit']
- `grep DSR E24_veto.txt / analyse_e24.txt` -> DSR = 0.003317, SR − SR0 = −0.064694, FAILLE

Tableau GONOGO : N/A (palier 3).

Décision Iyad requise : OUI. C'est le 3e INVALIDE consécutif sur le palier 3, ce qui déclenche l'arrêt prévu par CLAUDE.md. Iyad doit décider de la suite.

Fichiers :
- C:\Users\2\bot-trading-ia\experiments\results\E24_veto.txt
- C:\Users\2\bot-trading-ia\reports\adversarial\analyse_e24.txt
- C:\Users\2\bot-trading-ia\experiments\REGISTRE.md
