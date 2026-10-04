# Archive : rapport adversarial P3-A2 (candidat E17) et verdict associé P3-V3

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B3 ; rapport du chercheur : P3-R3 ; sorties brutes de l'adversarial : commits b94be6c, 4f8a77e, 740ca27.

---

RAPPORT ADVERSARIAL (réf. candidat E17)

Cadre : palier 3. Seules les données dev ont été lues (load_dataset() par défaut). Chaque script vérifie que l'index max vaut au plus 2025-09-30 23:00 UTC ; l'index max effectivement lu est 2025-09-30 23:00 UTC. Le code audité est inchangé depuis 26f81e3 : E17.json, nested.py, harness.py, extra_features.py et src/ ont les mêmes blobs à 26f81e3, à gonogo-v2 (68eeb61) et à HEAD. Je n'ai rien écrit dans experiments/ ni dans REGISTRE.md.

Commits :
- b94be6c : critères FAILLE / RÉSISTE (reports/adversarial/criteres_E17.txt), commités avant toute exécution et inchangés depuis.
- 4f8a77e : scripts et sorties conservées.
- 740ca27 : batterie complète.

`git status --porcelain` est vide.

Reprises : deux arrêts pendant la passe.
- 1re reprise (plantage de session) :
  - conservées, car leur sortie se termine par « RÉSULTAT : True » et le processus est sorti avec le code 0 : empoisonnement des folds 1, 2 et 3 ;
  - relancées, car peu coûteuses : leak_e17 static, grep et antériorité. Ces sorties sont complètes : code de sortie 0, ou marqueur « ## fin » ;
  - tout le reste a été relancé.
- 2e reprise (gel de la machine) :
  - conservé : tout ce qui précède, commité en 4f8a77e ;
  - relancés en deux lots successifs d'au plus 9 workers, avec OMP, MKL et OPENBLAS à 1 thread et n_jobs=1 :
    - lot 1 : base, shuf11 à shuf15, delay2, featlag1, cost2 (15:46 à 16:16) ;
    - lot 2 : cost3, drop_sma_dev_24, empoisonnement des folds 4 à 9 (16:16 à 16:39) ;
  - tous terminés avec le code 0, sans processus détaché ;
  - analyse_e17.py lancée ensuite, en un seul processus.

Commandes (lancées depuis C:\Users\2\bot-trading-ia) :
- Variantes : `.venv\Scripts\python tests\adversarial\run_nested_variant.py <VAR>`, avec VAR = base, shuf11 à shuf15, delay2, featlag1, cost2, cost3, drop_sma_dev_24. Seed du modèle : 42.
- Analyse : `.venv\Scripts\python tests\adversarial\analyse_e17.py --draws 1000 --seed 777`. Seeds 777 (tirages aléatoires) et 778 (tirages stratifiés).
- Fuite : `.venv\Scripts\python tests\adversarial\leak_e17.py static` (seed 7) et `leak_e17.py fold K --seed 7` pour K = 1 à 9 (seed d'empoisonnement 7+K).

Résultats (S = Sharpe net OOS concaténé de E17 reproduit) :

T0 Reproduction | S = 0,235865 ; |écart| avec le déclaré = 2,64e-07 ; 315 trades ; choix identiques au chercheur sur les 9 folds | RÉSISTE | « Sharpe 0.235865 (JSON 0.235865) trades 315 expo 0.559064 DD 0.7168 rdt total -0.0260 ; déclaré 0.235865 : |écart| 2.64e-07 » ; « choix par fold identiques au chercheur : [True x9] »

T1 Labels mélangés (seeds 11 à 15 mises en commun avec les seeds 1 à 10 du chercheur) | mes 5 seeds : +0,6225 / +0,4207 / +0,1043 / −0,0956 / +0,4384, moyenne +0,2980. Sur les 15 seeds : moyenne +0,2154, p95 +0,6494, z de E17 = +0,065 ; 7 seeds sur 15 font au moins aussi bien que E17. La performance ne tombe pas vers 0, et E17 ne se distingue pas du hasard | FAILLE TROUVÉE | « pool 15 seeds : moyenne +0.2154 écart-type 0.3150 p95 +0.6494 max +0.7025 » ; « E17 +0.2359 : z +0.065 ; seeds mélangées >= E17 : 7/15 ; S > p95 : False »

T2 Aléatoire, même nombre de trades et même exposition (1000 tirages, seed 777) | E17 au 64,4e percentile ; p95 = 0,6048 | FAILLE TROUVÉE | « trades tirés min/max 314/315 ; Sharpe moyenne 0.1169 p50 0.1295 p95 0.6048 max 1.1231 ; E17 percentile 64.4 »

T2b B&H à exposition égale | B&H fractionnaire constant à 0,5591 : Sharpe 0,7891 et rendement +166,8 %, contre E17 : Sharpe 0,2359 et rendement −2,6 %. Tirages stratifiés par fold (seed 778) : E17 au 73,9e percentile | FAILLE TROUVÉE | « B&H fractionnaire constant 0.5591 : Sharpe 0.7891 rdt +1.6682 DD 0.5373 | E17 rdt -0.0260 DD 0.7168 » ; « stratifié par fold (seed 778, 1000 tirages) : moyenne 0.0888 p95 0.5133 ; E17 percentile 73.9 »

T3 Retards | signal figé à t+2 : +0,2800, soit 119 % de S (amélioration). Procédure re-tunée : delay2 +0,4147, soit 176 % de S (amélioration) ; featlag1 −0,0385 (effondrement). Avec modèles et seuils figés, retarder les features d'une bougie revient à exécuter à t+2 | FAILLE TROUVÉE | « exec_delay=2 : Sharpe +0.2800 (119 % de E17) » ; « delay2 : Sharpe +0.4147 (176 % de E17) trades 162 » ; « featlag1 : Sharpe -0.0385 (-16 % de E17) trades 198 folds>0 4/9 »

T4 Coûts x2 et x3 | signal figé : −0,1691 (x2) et −0,5755 (x3). Procédure re-tunée : −0,1237 (x2) et −0,0165 (x3) | FAILLE TROUVÉE | « coûts x2 : Sharpe -0.1691 … rdt -0.6217 » ; « coûts x3 : Sharpe -0.5755 … rdt -0.8533 » ; « cost2 : Sharpe -0.1237 … folds>0 4/9 » ; « cost3 : Sharpe -0.0165 … folds>0 3/9 »

T5a Par année | une seule année négative (2022), mais E17 fait moins bien que le B&H sur 4 années sur 5 (2021, 2023, 2024, 2025) | FAILLE TROUVÉE | « 2021 : E17 +0.5632 | B&H +0.9777 » ; « 2022 : -0.9125 | -1.2928 » ; « 2023 : +0.1789 | +2.3425 » ; « 2024 : +1.3394 | +1.7510 » ; « 2025 : +0.0445 | +0.8293 »

T5b Par régime | hausse +2,35 ; range +0,37 ; baisse : rendement composé −89,4 %, sous le seuil de −50 % | FAILLE TROUVÉE | « baisse : 18 mois, 547 jours | E17 Sharpe -2.1374 rdt composé -0.8944 | B&H Sharpe -2.6954 rdt composé -0.9486 »

T6a Sans la meilleure feature (sma_dev_24, procédure re-tunée) | +0,2323, soit 98 % de S | RÉSISTE | « drop_sma_dev_24 : Sharpe +0.2323 (98 % de E17) trades 245 expo 0.494 DD 0.6370 folds>0 5/9 »

T6b Sans le meilleur mois (2021-02, +38,2 %) | 0,0861, soit 37 % de S, sous le seuil de 50 % ; sans les 3 meilleurs mois : −0,1961 | FAILLE TROUVÉE | « Sharpe sans les jours du meilleur mois : 0.0861 (37 % de E17) ; sans les 3 meilleurs mois : -0.1961 »

T7 Sharpe déflaté (N = 17 lignes de REGISTRE.md) | DSR = 0,000006, sous le seuil de 0,95 ; PSR(0) = 0,696 | FAILLE TROUVÉE | « V = 4.131690e-03 ; SR0 journalier 0.117507 (annualisé 2.2450) » ; « SR journalier E17 0.012346 ; T 1734 ; skew 0.1007 ; kurtosis 11.4106 ; DSR = 0.000006 ; PSR(0) = 0.696431 »

T8 Fuite | RÉSISTE sur les 4 volets :
- (a) Recherche statique : aucun allow_holdout, aucune date postérieure à la période dev, aucun shift négatif dans experiments/ ni dans tests/adversarial/. Seul cas : sha256_file lit les octets du parquet complet pour calculer une empreinte, sans exploiter aucune valeur.
- (b) Empoisonnement du futur : sur les 9 folds, les 16 enregistrements de validation, le choix et les probas du modèle final sont identiques.
- (c) Features extra (non couvertes par la passe P3-A1) : 0 troncature en écart sur 20.
- (d) Purge et validation commune H=24 : le pire cas est −1 h pour les labels du train interne, de la validation et du train final ; le dernier open lu par la validation est 25 h avant test_start.

Sortie brute : « (b) fold k RÉSULTAT : True » pour k = 1 à 9 (exemple : « (b) fold 5 (seed poison 12) : opens modifiés 24154 (>= 2023-01-01) ; 16 records de validation identiques True ; choix identique True (E15 / E15) ; probas du modèle final identiques True ») ; « (c) features extra : 20 troncatures (seed 7), troncatures en écart : 0 » ; « (d) pire cas … {"inner_label_end_minus_val_start_h": -1.0, "val_label_end_minus_test_start_h": -1.0, "final_label_end_minus_test_start_h": -1.0, "val_open_max_minus_test_start_h": -25.0} ».

T9 Antériorité et union | RÉSISTE :
- E17.json (blob 2a30538) et nested.py (blob 80a1b5b) sont identiques à 26f81e3, gonogo-v2 et HEAD.
- 26f81e3 (13:51:06) est ancêtre de a283d69 (14:31:54).
- Aucun fichier de résultat E17 ne figure dans l'arbre de 26f81e3.
- Les 16 blobs déclarés sont égaux à ceux de 970ea49, de HEAD et des commits de résultat de E01 à E16 (chaque config n'a été touchée que par 1 commit) ; l'union compte 3600 couples.
- Constat hors critère : E17_drop_sma_dev_24.json a été produit au commit a283d69, les autres JSON à 26f81e3, tous avec code_dirty False. Les configs E13 à E16 de l'union ont été conçues après les résultats OOS de E01 à E12 (commit e086ae0) : l'espace de recherche lui-même a été choisi en connaissant les OOS 2021–2025.

Sortie brute : « 26f81e3 ancêtre de a283d69 ? OUI » ; « E01 … E16 … OK » (16/16) ; « diff src/ experiments/ gonogo-v2..HEAD (vide = inchangé) ».

Fichiers :
- C:\Users\2\bot-trading-ia\reports\adversarial\criteres_E17.txt
- C:\Users\2\bot-trading-ia\reports\adversarial\analyse_e17.txt
- C:\Users\2\bot-trading-ia\reports\adversarial\E17_adv_*.json / .txt / E17_adv_base_signal_full.csv
- C:\Users\2\bot-trading-ia\reports\adversarial\leak_e17_static.txt, leak_e17_grep.txt, leak_e17_fold1..9.txt, anteriorite_e17.txt
- C:\Users\2\bot-trading-ia\tests\adversarial\run_nested_variant.py, analyse_e17.py, leak_e17.py

Conclusion : 9 failles trouvées (T1, T2, T2b, T3, T4, T5a, T5b, T6b, T7) ; 4 tests résistent (T0, T6a, T8, T9).

---

VERDICT P3-V3 | rapport évalué : P3-R3

STATUT : INVALIDE

Score : C0 7/8 bloquants OK (C0-07 vérifié par comparaison des sorties, sans réexécution complète) ; C3 8/12 bloquants OK (KO : C3-07, C3-10, C3-12, plus le veto)

Points KO bloquants :
- [VETO] Veto absolu. Le rapport adversarial indique FAILLE TROUVÉE sur T1 (labels mélangés), T2 (stratégie aléatoire, E17 au 64,4e percentile, sous le p95) et T7 (DSR 0,000006). Statut INVALIDE, aucune justification n'est recevable.
- [C3-07] Labels mélangés : la performance ne tombe pas près de 0. Je retrouve, à partir de E17_shuf1..10.json : moyenne +0,174, écart-type 0,334, z de E17 = +0,185. Avec les 15 seeds adversariales, z = +0,065 et 7 seeds sur 15 font au moins aussi bien que E17. Preuve attendue : une performance mélangée proche de 0 et une séparation nette du candidat.
- [C3-10] 9 failles adversariales ne sont pas compensées par une preuve : T1, T2, T2b, T3, T4, T5a, T5b, T6b, T7. Le rapport P3-R3 reconnaît lui-même que l'aléatoire et le B&H fractionnaire restent défavorables. Il ne traite ni T4 (coûts x2 : −0,169 ; x3 : −0,576), ni T6b (37 % de S sans le meilleur mois), ni T5b (rendement composé −89,4 % en baisse). Preuve attendue : chaque faille expliquée par une preuve chiffrée.
- [C3-12] DSR non positif au sens attendu : SR − SR0 = −0,105161 (journalier), DSR = 0,000006. Les valeurs sont confirmées dans experiments/results/E17_report.txt. Preuve attendue : un DSR du candidat positif au sens attendu.

Points NON VÉRIFIÉS :
- [C0-07] Je n'ai pas relancé E17 moi-même (environ 40 min de calcul). J'ai seulement vérifié que E17.json et repro/E17_run2.json sont identiques hors champ commit (True). Ce point n'est pas déterminant pour le statut, déjà INVALIDE par le veto.

Points d'attention (non bloquants) :
- C0-08 : la sélection dans E17 se fait par validation interne, et l'antériorité est prouvée. Mais d'après l'adversarial (T9), les configurations E13 à E16 de l'union ont été conçues après avoir vu les OOS de E01 à E12. L'espace de recherche reste donc informé par l'OOS 2021–2025. Ce biais résiduel n'est pas mesuré.
- Section 5 du rapport : la liste des tags ne contient pas gonogo-v2, qui a été posé après le rapport (68eeb61, 15:12). Pas d'incohérence à la date du rapport.
- `git diff --stat 970ea49 HEAD` touche aussi GONOGO.md, CLAUDE.md, .claude/ et JOURNAL.md, mais par des commits tiers (Iyad et la session principale : 68eeb61, a4d80c5, a322196, bdb869b, c1cf991). Les 3 commits de l'implémenteur ne touchent que experiments/.
- Fold 5 : 0 trade, et la configuration a été choisie sur 1 seul trade de validation. Écart validation / OOS très important (fold 1 : 5,51 contre 0,58).
- La ligne E17 du registre met les configurations choisies dans la colonne H. Limite déclarée.
- E17 est sous le B&H sur 8 folds sur 9, et sous le B&H fractionnaire à exposition égale (0,789 contre 0,236).

Vérifications relancées par moi :
- `git diff gonogo-v2 -- GONOGO.md` -> vide ; le tag gonogo-v2 pointe sur 68eeb61.
- `git status --porcelain` -> vide.
- `git check-ignore .env` -> .env.
- `pytest -q` -> 72 passed in 5.45s, 0 skip.
- `git show --name-only` sur 26f81e3, a283d69 et 048133d -> aucun fichier hors experiments/.
- Pour E01 à E16 : `git rev-parse 970ea49:experiments/configs/E{i}.json` et `git hash-object` donnent le même hash -> 16/16 identiques, conformes aux blobs déclarés.
- `git log --date=iso` -> 26f81e3 (13:51:06) précède a283d69 (14:31:54).
- `git diff 26f81e3 HEAD -- experiments/nested.py experiments/configs/E17.json src/` -> vide.
- REGISTRE.md -> 17 lignes ; 0 ligne retirée dans `git log -p 970ea49..HEAD`.
- E17.json comparé à repro/E17_run2.json, hors champ commit -> identiques.
- E17.json concaténé -> Sharpe 0,235865 ; 315 trades ; expo 0,5591 ; baseline −0,5297 ; B&H 0,7815. Conforme au rapport.
- Choix internes par fold relus dans E17.json -> conformes au rapport (E06, E08, E14, E11, E15, E10, E14, E10, E14 ; scores identiques).
- Labels mélangés, seeds 1 à 10 -> moyenne 0,174 ; écart-type 0,3342 ; z +0,185. Conforme au rapport.
- DSR dans E17_report.txt -> V 4,131690e-03 ; SR0 0,117507 ; DSR 0,000006. Conforme au rapport.

Tableau GONOGO : N/A (palier 3).

Décision Iyad requise : NON. Il s'agit du 3e verdict INVALIDE consécutif sur le palier 3 (P3-V1 ?, P3-V2, P3-V3, à confirmer dans JOURNAL.md). Si c'est bien le cas, la règle d'escalade du superviseur s'applique.

Fichiers :
- C:\Users\2\bot-trading-ia\experiments\results\E17.json
- C:\Users\2\bot-trading-ia\experiments\results\repro\E17_run2.json
- C:\Users\2\bot-trading-ia\experiments\results\E17_report.txt
- C:\Users\2\bot-trading-ia\reports\adversarial\analyse_e17.txt
- C:\Users\2\bot-trading-ia\reports\adversarial\criteres_E17.txt
