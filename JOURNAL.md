# JOURNAL
| ID | Date | Statut | Tag |
|---|---|---|---|
| P0a-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-0a |
| P0b-B1 / R1 / V1 | 2026-10-04 | VALIDE | gonogo-v1 (posé par Iyad) |
| P1-B1 / R1 / V1 | 2026-10-04 | INVALIDE (1/3) : A1, archive mensuelle 2026-09 absente (404), remplacée par 30 journalières sans décision du brief | — |
| P1-B2 / R2 / V2 | 2026-10-04 | VALIDE (KO A1 levé) | palier-1 |
| P2-B1 / R1 / V1 | 2026-10-04 | VALIDE | palier-2 |
| P3-B1 / R1 / V1 | 2026-10-04 | VALIDE (env : scikit-learn 1.9.1, lightgbm 4.7.0, 72 tests) | — |
| P3-B2 / R2 / A1 / V2 | 2026-10-04 | INVALIDE (1/3) : candidat E10 (Sharpe OOS 0,669, 16 essais) ; KO C3-07 labels mélangés +0,36, C3-10 8 failles adversariales, C3-12 DSR 0,0002, C0-08 conception guidée par l'OOS | — |
| NOUVEAU CYCLE GONOGO (Iyad) | 2026-10-04 | GONOGO v2 commité et tagué par Iyad (68eeb61) : section D « périmètre de calcul », aucun seuil v1 modifié ; intégrité désormais vérifiée contre gonogo-v2 | gonogo-v2 (posé par Iyad) |
| P3-B3 / R3 | 2026-10-04 | rapport reçu (E17, Sharpe OOS 0,236) ; avocat P3-A2 interrompu par le plantage de session, en reprise ; pas encore de verdict | — |
| P3-B3 / R3 / A2 / V3 | 2026-10-04 | INVALIDE (2/3 d'affilée ; P3-V1 était VALIDE) : candidat E17 ; VETO tests adversariaux 1, 2, 7 ; KO C3-07 (labels mélangés z +0,065 sur 15 seeds), C3-10 (9 failles), C3-12 (DSR 0,000006) ; C0-08 levé sur la procédure. Archive : reports/adversarial/P3-A2_E17_verdict_P3-V3.md | — |
| P3-B4 / R4 / A3 / V4 | 2026-10-04 | INVALIDE (3/3 d'affilée → STOP, décision d'Iyad requise) : candidat E24 (triple barrière H=24, Sharpe OOS 0,799) ; T1 et T2 résistent ; VETO T7 (DSR 0,0033) ; KO C3-10 (6 failles), C3-08 (vol_168 circulaire, jour de semaine), C3-09 (écart IS/OOS). Archive : reports/adversarial/P3-A3_E24_verdict_P3-V4.md | — |

## Récap palier 3 au STOP (2026-10-04)
- Registre : 28 essais sur 60 (E01–E28). Aucun candidat validé. Données réservées jamais lues.
- E10 (Sharpe 0,669, conception guidée par l'OOS) : rejeté (labels mélangés, aléatoire, DSR).
- E17 (procédure emboîtée sur E01–E16, 0,236) : ne se distingue pas du hasard.
- E24 (triple barrière vol, 0,799 ; B&H 0,781) : passe labels mélangés (z +3,8) et aléatoire (98,4e pct) mais DSR 0,0033 (SR0 annualisé 2,04 avec N=28) ; avantage surtout sur « barrière touchée » (volatilité), pas sur la direction ; dépend de vol_168 et du jour de semaine.
- Corrélations > 0,9 : vol_24/range_24, ret_24/ret_24_z (variantes élaguées E22, E26 moins bonnes).
- Contraintes en vigueur : gonogo-v2, 9 cœurs max, veto absolu T1/T2/T7, archivage adversarial.
- Prochaine étape : décision d'Iyad sur la suite du palier 3.
- 2026-10-05 — P3-B5 / R5 / A4 / V5 : INVALIDE (cycle 2, 1/3). Candidat E36 (bougies 4 h, triple barrière fixe 5 jours) : Sharpe OOS 1,175, 48 trades ; T1 et T2 résistent ; VETO T7 (DSR 0,0456, N=39, SR0 annualisé 1,94) ; KO C3-10 (6 failles : régimes, vol_180, barrières ×1,2, règle vol sans apprentissage, direction seule). Archive : reports/adversarial/P3-A4_E36_verdict_P3-V5.md. Registre 39/60.
- 2026-10-04 — DÉCISION D'IYAD : OUI, le palier 3 continue avec de nouvelles pistes (option 2), dans le même cadre : GONOGO v2 et veto inchangés, budget restant 32 essais. Le compteur d'INVALIDE d'affilée repart à 0 pour ce nouveau cycle.

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
- 2026-10-04 — STOP obligatoire après la validation du palier 2, avant de lancer le palier 3 (Iyad ajoute des sous-agents). LEVÉ : agents chercheur-ml et avocat-du-diable ajoutés (c5dd5e2), garde corrigé ; palier 3 lancé en autonomie jusqu'au prochain point critique.
- 2026-10-04 — Iyad confirme être l'auteur du commit be40d4a « notifications ntfy » (modifie CLAUDE.md et .claude/), demande du contrôleur en P3-V2.
- 2026-10-04 — Consigne d'Iyad pour la suite du palier 3 (GONOGO inchangé) : dans les prochains briefs du chercheur, tester (a) un label net de coûts : positif seulement si le rendement sur H dépasse le coût aller-retour (2 × 0,15 % = 0,30 %) ; (b) éventuellement un label triple barrière. Chaque nouveau label = 1 essai du REGISTRE. Faire vérifier les corrélations entre features et signaler les paires |corr| > 0,9. Cette consigne d'Iyad autorise cet ajout au périmètre des briefs correctifs.
- 2026-10-04 — NOUVEAU CYCLE décidé par Iyad : GONOGO v2 (tag gonogo-v2 → 68eeb61). Raison donnée par Iyad : ambiguïté du périmètre OOS détectée avant tout résultat de modèle. Précision factuelle du superviseur : au moment du tag, 17 essais du palier 3 existaient déjà sur les folds de la période dev (2021-01 → 2025-09) ; aucun résultat n'existait sur la période réservée, et aucune évaluation GONOGO (palier 4) n'avait eu lieu. Contenu de la section D : « période OOS concaténée » = folds walk-forward 2021-01 → 2025-09 + période réservée de 11 mois ; folds positifs et écart baseline sur les folds uniquement ; garde-fou période réservée seule : Sharpe net > 0 et DD < 25 % ; B&H informatif. À partir de maintenant, toutes les vérifications d'intégrité se font contre gonogo-v2 (controleur.md et implementeur.md mis à jour par Iyad dans le même commit, ainsi que le veto absolu sur les tests adversariaux 1, 2 et 7, et la règle d'archivage adversarial dans CLAUDE.md).
- 2026-10-04 — Plantage de session pendant P3-A2 (avocat sur E17) ; Iyad a fermé les processus Python. Aucun essai à refaire : le registre compte 17 lignes (E01–E17), qui restent dans le budget (17/60).
- Palier 3 : budget 20 essais par brief, 60 au total (CLAUDE.md). Afficher à titre informatif chaque résultat comparé au buy & hold sur les mêmes folds (pas un critère GONOGO).
- Rappel : données réservées au palier 4 (holdout) = 2025-10-01 00:00 → 2026-08-31 23:00 UTC (la ligne "Découpage" ci-dessous est corrigée par la ligne "Correction de la décision Période").

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
- 2026-10-04 — Palier 3, déroulé : (1) implémenteur ajoute les dépendances ML ; (2) chercheur-ml explore (≤ 20 essais/brief) ; (3) avocat-du-diable sur le candidat ; (4) contrôleur ; (5) si VALIDE, implémenteur intègre le candidat dans src/ avec tests, reproduction à l'identique, nouvel avocat + contrôleur, puis tag palier-3.
- 2026-10-04 — Palier 3, comparaison à la baseline : positions du modèle concaténées sur les 9 folds puis un seul backtest continu (même moteur, même coût 0,15 %/côté, même traitement des débuts de fenêtre que la baseline), métriques par fold calculées comme dans run_baseline.py. Corrige l'asymétrie AT-1 de P2-V1 : modèle et baseline sont traités de la même façon.
- 2026-10-04 — Palier 3, définition d'un essai : une configuration (modèle + features + label + procédure de tuning) évaluée sur les 9 folds OOS = 1 ligne de REGISTRE.md. Le tuning interne au train (validation temporelle purgée dans le train) fait partie de la procédure ; la taille de sa grille est notée dans la ligne.
- 2026-10-04 — Palier 3, règle de choix du candidat, fixée avant tout essai : parmi les essais du brief, le candidat est celui qui a le Sharpe net OOS concaténé le plus élevé, à condition qu'il dépasse celui de la baseline sur les mêmes folds (−0,530). Sinon, aucun candidat. Les seuils GONOGO ne sont évalués qu'au palier 4.
- 2026-10-04 — Palier 3, mapping proba → position : long si proba > seuil, sinon flat ; seuil et hyperparamètres choisis sur une validation interne au train (derniers 20 % du train, purge H+1), objectif = Sharpe net avec coûts ; réentraînement sur tout le train avant de prédire le test.
- 2026-10-04 — Palier 3, correctif P3-B3 (suite P3-V2, KO C0-08) : la sélection entre configurations ne se fait plus en comparant les résultats OOS. Une procédure « emboîtée » choisit, dans chaque fold et sur la seule validation interne au train, parmi l'union des 16 espaces de configuration E01–E16 déclarés avant (y compris les mauvais). Elle est commitée avant toute exécution et évaluée une seule fois sur l'OOS (= 1 ligne de registre). Aucune nouvelle conception après avoir vu son résultat.
- 2026-10-04 — Palier 3, correctif P3-B4 (suite P3-V3, dernier avant le STOP des 3 INVALIDE) : nouveaux labels selon la consigne d'Iyad (label net de coûts, triple barrière), avec la discipline de P3-B3. Toutes les configurations sont pré-enregistrées dans un seul commit avant toute exécution, sans nouvelle conception après résultat. On ajoute une procédure emboîtée sur ces seules nouvelles configurations, et un contrôle des corrélations entre features (|corr| > 0,9). Les tests du veto (labels mélangés, aléatoire à exposition égale, DSR) sont calculés par le chercheur pour chaque candidat éligible.
- 2026-10-04 — Palier 3, cycle 2, brief P3-B5 : piste « horizon plus long » sur bougies de 4 h agrégées à partir des bougies 1 h existantes. Pas de nouvelle source de données à ce stade : une source comme le funding demanderait un nouveau pipeline en src/ et ne sera ouverte que si cette piste échoue. Raison : les essais E18–E28 montrent que l'avantage porte sur la volatilité à 24 h ; les coûts pèsent moins sur des trades plus longs. Budget du brief : 12 essais au maximum (registre ≤ 40), tous pré-enregistrés dans un seul commit, procédure emboîtée incluse. Mêmes folds, mêmes coûts, même veto.
- 2026-10-05 — Palier 3, cycle 2, après P3-V5 : la piste « horizon plus long » sur le même jeu de données plafonne au veto T7 (Sharpe exigé ≈ 1,94 annualisé, N=39), et ses gains reposent sur la volatilité. Décision : ouvrir une source d'information nouvelle, le taux de funding du contrat perpétuel BTCUSDT (Binance USDⓈ-M, archives publiques data.binance.vision, sans clé). C'est une piste citée dans l'option 2 validée par Iyad. Étapes : P3-B6 (implémenteur) pipeline funding dans src/ avec tests anti-fuite et égalité live/batch, période réservée verrouillée ; puis P3-B7 (chercheur) essais pré-enregistrés avec features de funding. Le funding n'est qu'une feature : on trade toujours le spot, long/flat. Disponibilité : un taux réglé à fundingTime n'est utilisable qu'à la clôture de la bougie 1 h qui contient fundingTime (fundingTime ≤ t + 1 h).
- 2026-10-05 — P3-B6 / R6 / V6 : NON VÉRIFIABLE (aucun KO). Pièce manquante : C0-02, la sortie de `git diff gonogo-v2 -- GONOGO.md`. L'implémenteur ne l'a pas lancée, en se fiant à une note périmée du journal ; le contrôleur a vu cette commande refusée par le système de permissions. Rectification : depuis la correction du garde par Iyad (commit 68eeb61 et précédents), la lecture `git diff gonogo-v2 -- GONOGO.md` n'est plus bloquée par le garde (vérifié par le superviseur le 2026-10-04 : exit 0, sortie vide) ; la note « Garde : toute commande Bash contenant GONOGO.md est bloquée, même en lecture » du récap P2 n'est plus valable.
- 2026-10-05 — P3-B6bis / R6bis / V6bis : VALIDE. GONOGO intact contre gonogo-v2 (diff vide, exit 0). Le pipeline funding (src/bot/funding.py, commits a58fab5, 7385794, 042a5f3) est donc validé : 80/80 archives, SHA256 151c60bb…8462 reproduit depuis un clone propre, live = batch 2000/2000, 87 tests. Pas de tag : c'est une brique du palier 3, pas sa fin.
- 2026-10-05 — Palier 3, P3-B7 : essais avec features de funding sur bougies 4 h. La valeur à la bougie 4 h t est le dernier règlement avec fundingTime ≤ clôture de t. Budget de 10 essais au maximum (registre ≤ 49), tous pré-enregistrés. Les labels repris (triple barrière fixe H=30, net de coûts H=30, triple barrière vol H=6) ont été choisis au vu des résultats OOS de P3-B5 : je le déclare, et le DSR (N cumulé) en tient compte. Contrôle exigé : la même configuration avec et sans funding, pour isoler l'apport du funding.
- 2026-10-05 — P3-B7 / R7 / A5 / V7 : INVALIDE. Candidat E40 (config de E36 sur train à partir de 2020, sans funding) : Sharpe OOS 1,114 ; VETO T7 (DSR 0,0439, N=47, SR0 annualisé 1,89) ; KO C3-10 (8 failles, dont l'instabilité au point de départ du train, de 0,45 à 1,32, et la seed 42 meilleure sur 6). Le funding n'apporte rien : paire E41 − E40 = −0,235, placebo au-dessus du vrai. Archive : reports/adversarial/P3-A5_E40_verdict_P3-V7.md. Registre 47/60. Décompte : 2e candidat INVALIDE du cycle 2 (V5, V7), le VALIDE de V6bis portant sur la brique funding et non sur un candidat.
- 2026-10-05 — STOP superviseur (blocage) : le veto T7 exige un Sharpe annualisé ≈ 1,89 avec N=47, et ce seuil monte à chaque essai. Les meilleurs candidats plafonnent à environ 1,1, et vers 0,88 en moyenne sur les seeds. Dépenser les 13 essais restants sans piste capable d'approcher ce niveau consommerait le budget presque sans chance d'aboutir. Décision demandée à Iyad.
- 2026-10-05 — Consigne d'Iyad : tout commiter et pousser sur GitHub, sans attendre la fin du palier. Le superviseur pousse donc `git push origin main --tags` à chaque étape importante (verdict, archive), en plus de chaque palier validé.
- 2026-10-05 — Funding : décision de garder la règle stricte fundingTime ≤ t + 1 h, sans arrondir la gigue (jusqu'à 47 ms) : c'est prudent et sans fuite. Une partie des règlements arrive donc avec 1 h de retard. Le funding n'existe qu'à partir de 2020-01 ; les folds ont donc moins d'historique de funding dans leur train.
- 2026-10-04 — Contrainte de calcul (décision non critique demandée par Iyad ; cause : gel du terminal, processeur à 99 % avec une vingtaine de processus Python lancés par l'avocat du diable sur E17). Plusieurs essais peuvent tourner en parallèle, mais le total des cœurs utilisés en même temps par tous les essais et tests ne dépasse jamais 9 (machine à 12 cœurs logiques) : n_jobs répartis en conséquence, OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1 dans les workers. Les lignes de REGISTRE.md sont écrites une par une, jamais par deux processus en même temps. Tout essai interrompu apparaît dans REGISTRE.md avec le statut « interrompu » et compte dans le budget ; aucun fichier de résultat partiel ne reste sans ligne correspondante.
- 2026-10-04 — Point après le 2e gel : aucun essai du chercheur interrompu (E17 terminé avant le gel, 17 lignes au registre). Les sorties interrompues sont des tests adversariaux P3-A2, qui ne sont pas des essais : elles seront déclarées dans le rapport de l'avocat.
- 2026-10-04 — Features à fenêtre finie uniquement (pas d'EMA à mémoire infinie) pour que le calcul live sur une fenêtre glissante soit exactement égal au calcul backtest.
