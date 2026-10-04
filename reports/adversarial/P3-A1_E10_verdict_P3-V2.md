# Archive : rapport adversarial P3-A1 (candidat E10) et verdict associé P3-V2

Archivé par le superviseur (règle « ARCHIVAGE ADVERSARIAL » de CLAUDE.md). Textes intégraux, inchangés.
Brief du candidat : P3-B2 ; rapport du chercheur : P3-R2 ; sorties brutes de l'adversarial : commit f6e0e30.

---

RAPPORT ADVERSARIAL (réf. candidat E10)

Contexte vérifié : HEAD 4bbaacf au départ. Mes fichiers sont dans le commit f6e0e30 « P3-adv: ... » (scripts, configs dérivées, sorties .json/.txt). Les séries .csv sont régénérables et ne sont pas commitées. REGISTRE.md n'a pas changé (16 lignes). experiments/ et src/ n'ont pas été modifiés. Garde vérifiée dans chaque script : `index max lu = 2025-09-30 23:00:00+00:00`. Le modèle utilise la seed 42 partout.
Reproduction : `.venv\Scripts\python tests\adversarial\run_variant.py base` donne Sharpe concat 0.6695, 194 trades, expo 0.4731, DD 0.5228, 9/9 folds positifs. Le bloc `concatenated.model` est identique à experiments/results/E10.json (True).

Scripts :
- C:\Users\2\bot-trading-ia\tests\adversarial\run_variant.py : relance la procédure complète du harnais importé, avec une perturbation.
- C:\Users\2\bot-trading-ia\tests\adversarial\analyse_e10.py : tests sur le signal OOS reproduit.
- C:\Users\2\bot-trading-ia\tests\adversarial\leak_e10.py : test de fuite dynamique.
Sorties : C:\Users\2\bot-trading-ia\reports\adversarial\ (E10_adv_*.json/.txt, analyse_e10.txt, leak_e10.txt).

Critères fixés avant de lire les résultats :
- T1 : FAILLE si la moyenne mélangée dépasse 0,2.
- T2 : FAILLE si E10 est sous le 95e percentile.
- T3 : FAILLE si Sharpe ≤ 0 (effondrement) ou supérieur à 0,6695 (amélioration).
- T4 : FAILLE si Sharpe ≤ 0.
- T5 : FAILLE si une sous-période a un Sharpe ≤ 0.
- T6 : FAILLE si le Sharpe restant est sous 50 % du candidat.
- T7 : FAILLE si DSR < 0,95.
- T8 : FAILLE si un accès est trouvé.

1. Labels mélangés (5 seeds 6 à 10, procédure complète) | moyenne +0.3168 (écart-type 0.3065) ; sur 10 seeds avec celles du chercheur : moyenne +0.3565, 1 sur 10 ≥ candidat (seed 4 : 0.7736), z du candidat = 1.185 | FAILLE TROUVÉE
Commande : `.venv\Scripts\python tests\adversarial\run_variant.py shufS` pour S = 6 à 10
```
[adv] variante shuf6 : ... Sharpe concat 0.3101 trades 26 expo 0.3773 DD 0.5566 folds>0 5/9
[adv] variante shuf7 : ... Sharpe concat 0.3498 trades 49 expo 0.3302 DD 0.5900 folds>0 3/9
[adv] variante shuf8 : ... Sharpe concat -0.1915 trades 41 expo 0.3941 DD 0.7676 folds>0 3/9
[adv] variante shuf9 : ... Sharpe concat 0.5606 trades 28 expo 0.4810 DD 0.5481 folds>0 5/9
[adv] variante shuf10 : ... Sharpe concat 0.5549 trades 113 expo 0.3174 DD 0.3555 folds>0 5/9
seeds6-10 [ 0.3101  0.3498 -0.1915  0.5606  0.5549] moy 0.3168 std 0.3065
10 seeds moy 0.3565 std 0.2643 >=0.6695: 1 z= 1.185
```

2. Stratégie aléatoire, mêmes trades (194 segments), même exposition (19690/41616 bougies), coûts 0,0015 ×1, 2021-01-01 → 2025-09-30, 1000 tirages | E10 au 89.6e percentile ; p95 = 0.8065 | FAILLE TROUVÉE
Commande : `.venv\Scripts\python tests\adversarial\analyse_e10.py --draws 1000 --seed 20261004`
```
trades tirés min/max 193/194 ; expo min/max 0.4730/0.4733
Sharpe aléatoire : moyenne 0.2366 écart-type 0.3362 p5 -0.3139 p50 0.2258 p95 0.8065 max 1.4137
E10 0.6695 -> percentile 89.6 ; > p95 : False
```

2b. Buy & hold avec la même exposition | B&H fractionnaire constant à 47,31 % : Sharpe 0.7906, supérieur à E10 (0.6695), DD 47.4 % contre 52.3 % ; aléatoire stratifié par fold (mêmes trades et même exposition dans chaque fold), 1000 tirages : E10 au 86.0e percentile | FAILLE TROUVÉE
Même commande ; seed 20261005 pour le tirage stratifié.
```
B&H fractionnaire constant 0.4731 : Sharpe 0.7906 rdt 1.3777 DD 0.4742
plan par fold (barres, segments, barres exposées) : [(4344, 34, 3845), (4416, 38, 2141), (4344, 1, 80), (4416, 7, 566), (4344, 1, 282), (4416, 7, 1820), (4368, 19, 2205), (4416, 72, 2766), (6552, 19, 5985)]
stratifié par fold (seed 20261005, 1000 tirages) : moyenne 0.4313 p50 0.4282 p95 0.7937 ; E10 percentile 86.0
```

3a. Exécution à t+2 | Avec le signal E10 figé, le Sharpe passe à 0.7632 : il s'améliore (+0.094) et le DD baisse. Avec la procédure complète re-tunée (exec_delay=2) : 0.6446, 9/9 | FAILLE TROUVÉE (amélioration sur signal figé)
Commandes : `analyse_e10.py` (ci-dessus) et `.venv\Scripts\python tests\adversarial\run_variant.py delay2`
```
exec_delay=2  : Sharpe 0.7632 trades 194 expo 0.4731 DD 0.4835 rdt 2.0091
[adv] variante delay2 : exec_delay=2 cost_mult=1.0 Sharpe concat 0.6446 trades 151 expo 0.4758 DD 0.5784 folds>0 9/9
```

3b. Features retardées d'une bougie (X.shift(1), procédure complète) | Le Sharpe s'effondre à −0.3121, DD 85.7 %, 3 folds positifs sur 9 | FAILLE TROUVÉE
Commande : `.venv\Scripts\python tests\adversarial\run_variant.py featlag1`
```
[adv] variante featlag1 : exec_delay=1 cost_mult=1.0 Sharpe concat -0.3121 trades 143 expo 0.4332 DD 0.8566 folds>0 3/9
featlag1 [-0.176, -0.102, -2.9, 1.408, 'nan', 1.158, -0.643, -0.178, 0.63] [55, 1, 19, 1, 0, 9, 14, 20, 28]
```

4. Coûts ×2 et ×3 | Signal figé : 0.3765 (×2) et 0.0829 (×3), avec un rendement total de −22.1 % à ×3. Procédure re-tunée : 0.5977 (×2) et 0.3627 (×3, 6 folds positifs sur 9). Le critère Sharpe > 0 tient, mais tous ces Sharpe sont sous le B&H (0.7815) | RÉSISTE
Commandes : `analyse_e10.py` ; `run_variant.py cost2` ; `run_variant.py cost3`
```
coûts x2      : Sharpe 0.3765 trades 194 expo 0.4731 DD 0.5398 rdt 0.3943
coûts x3      : Sharpe 0.0829 trades 194 expo 0.4731 DD 0.6970 rdt -0.2214
[adv] variante cost2 : exec_delay=1 cost_mult=2.0 Sharpe concat 0.5977 trades 116 expo 0.5453 DD 0.5576 folds>0 9/9
[adv] variante cost3 : exec_delay=1 cost_mult=3.0 Sharpe concat 0.3627 trades 127 expo 0.6178 DD 0.6077 folds>0 6/9
```

5a. Par année | Les 5 années sont positives. E10 est sous le B&H en 2021, 2023, 2024 et 2025 ; il fait mieux uniquement en 2022 | RÉSISTE
```
2021 : E10 Sharpe +0.7772 rdt +0.3506 trades 71 expo 0.683 | B&H Sharpe +0.9777 rdt +0.5931
2022 : E10 Sharpe +0.5855 rdt +0.1013 trades 8 expo 0.074 | B&H Sharpe -1.2928 rdt -0.6432
2023 : E10 Sharpe +0.9917 rdt +0.1565 trades 8 expo 0.240 | B&H Sharpe +2.3425 rdt +1.5485
2024 : E10 Sharpe +1.0770 rdt +0.4048 trades 91 expo 0.566 | B&H Sharpe +1.7510 rdt +1.2064
2025 : E10 Sharpe +0.3032 rdt +0.0324 trades 19 expo 0.913 | B&H Sharpe +0.8293 rdt +0.2140
```

5b. Par régime | Définition a priori : rendement B&H du mois civil, hausse > +5 %, baisse < −5 %, range sinon. Range : Sharpe −0.0968 (le B&H fait +0.1208). Baisse : −1.4636 | FAILLE TROUVÉE
```
hausse : 27 mois, 823 jours | E10 Sharpe +2.6243 rdt composé +8.2829 | B&H Sharpe +3.6113 rdt composé +79.5458
baisse : 18 mois, 547 jours | E10 Sharpe -1.4636 rdt composé -0.7089 | B&H Sharpe -2.6954 rdt composé -0.9486
range  : 12 mois, 364 jours | E10 Sharpe -0.0968 rdt composé -0.0769 | B&H Sharpe +0.1208 rdt composé -0.0519
```

6a. Sans la meilleure feature | sma_dev_24 est bien la première au gain moyen (3397.6, contre 3288.7 pour vol_168). Sans elle, le Sharpe tombe à 0.1970, soit 29 % du candidat, avec un DD de 83.9 % | FAILLE TROUVÉE
Commande : `.venv\Scripts\python tests\adversarial\run_variant.py drop_sma_dev_24`
```
[adv] variante drop_sma_dev_24 : exec_delay=1 cost_mult=1.0 Sharpe concat 0.1970 trades 223 expo 0.4881 DD 0.8385 folds>0 7/9
drop folds [0.064, 0.331, -3.181, -1.358, 1.44, 1.038, 2.473, 0.712, 0.731]
```

6b. Sans le meilleur mois (2021-02, +37.5 %) | Sharpe 0.5172, soit 77 % du candidat. Sans les 3 meilleurs mois : 0.2057 | RÉSISTE
```
Sharpe sans les jours du meilleur mois : 0.5172 ; meilleur mois mis à plat : 0.5130 ; sans les 3 meilleurs mois : 0.2057
```

7. Sharpe déflaté, N = 16 lignes de REGISTRE.md, recalcul indépendant à partir des rendements journaliers reproduits | DSR = 0.000238 ; SR0 annualisé = 2.2733, contre 0.6695 pour E10. Le recalcul est identique à la valeur déclarée | FAILLE TROUVÉE
```
V = 4.367657e-03 ; SR0 journalier = 0.118989 (annualisé 2.2733)
SR E10 journalier = 0.035043 ; T = 1734 ; skew 0.0927 ; kurtosis 12.7973 ; dénominateur 1.000186
DSR = 0.000238 ; PSR(0) = 0.927655
```

8. Fuite | Recherche statique (grep) dans src/, experiments/ et scripts/ : aucun appel `allow_holdout=True` ou `holdout_period(` hors tests/test_split.py, aucun `shift(-`, `bfill` ou fenêtre centrée. Test dynamique, `.venv\Scripts\python tests\adversarial\leak_e10.py --seed 7` : tout est identique | RÉSISTE
Constats sans faille de code :
- load_dataset() lit tout le parquet en mémoire, période réservée comprise, avant de filtrer (src/bot/split.py:74).
- La période OOS 2021-2025 a servi 16 fois de retour pour concevoir les essais. La grille de E10 a été définie après avoir vu les résultats de E06 et E09.
```
(a) features : 20 troncatures aléatoires, valeurs <= t identiques bit à bit : True
(a) labels : open > t+1+H modifiés, label[<=T] identiques : True
(a) folds : pour chaque fold, dernier open utilisé par un label de train < test_start : True
(b) fold 1..5 : choix identiques True, 288 scores identiques True, probas ... identiques True
(b) futur empoisonné à partir de 2023-01-01 00:00:00+00:00 : aucune influence sur les folds 1..5 : True
```

Conclusion : 8 failles trouvées (tests 1, 2, 2b, 3a, 3b, 5b, 6a, 7). 4 tests résistent (4, 5a, 6b, 8).

---

VERDICT P3-V2 | rapport évalué : P3-R2

STATUT : INVALIDE

Score : C0 8/10 ; C3 8/12

Points KO bloquants :
- [C3-07] Règle violée : avec les labels mélangés, la performance doit être proche de 0. Constat : sur la procédure complète, la moyenne est de +0,396 sur 5 seeds (+0,357 sur 10 seeds avec celles de l'adversarial). La seed 4 donne +0,7736, au-dessus du candidat (+0,6695) ; je l'ai relancée et j'obtiens un JSON identique. Le z du candidat face aux seeds mélangées vaut 1,185. Preuve attendue : un test labels mélangés dont la performance est proche de 0, et un candidat qui s'en distingue nettement.
- [C3-10] Règle violée : toute FAILLE TROUVÉE du rapport adversarial doit être expliquée par une preuve. Constat : 8 failles (tests 1, 2, 2b, 3a, 3b, 5b, 6a, 7), aucune expliquée par une preuve.
  - Stratégie aléatoire à mêmes trades et même exposition : E10 au 89,6e percentile (86,0e en stratifié par fold).
  - B&H fractionnaire à 47,31 % : Sharpe 0,7906, au-dessus de E10.
  - Exécution à t+2 sur signal figé : 0,7632, meilleur que t+1.
  - Features retardées d'une bougie : −0,3121.
  - Sans sma_dev_24 : 29 % du candidat.
  - DSR : 0,000238.
  Preuve attendue : une preuve chiffrée pour chaque faille.
- [C3-12] Règle violée : le Sharpe déflaté doit être fourni et positif. Constat : dans mon recalcul, la statistique (SR − SR0) vaut −3,494 et le DSR 0,000238. Le SR journalier vaut 0,035043, sous SR0 = 0,118989 (2,27 annualisé). Preuve attendue : un DSR montrant que le Sharpe du candidat dépasse SR0.
- [C0-08] Règle violée : aucune variante ne doit être choisie sur les données OOS. Constat :
  - le rapport (section 10) reconnaît « des choix de conception guidés par les résultats OOS des essais précédents » ;
  - l'adversarial (test 8) écrit : « la grille de E10 a été définie après avoir vu les résultats de E06 et E09 » ;
  - le DSR ne compense pas ce biais (point C3-12).
  Preuve attendue : des variantes dont la conception n'utilise pas l'OOS des folds, ou une correction du biais de sélection qui passe.

Points NON VÉRIFIÉS :
- [C1-06] Aucun test d'égalité batch/live pour les features : N/A tant que le candidat n'est pas intégré dans src/.

Points d'attention (non bloquants) :
- A7, git status : à mon passage, `git status --porcelain` n'est pas vide. 12 fichiers reports/adversarial/*.csv non suivis, produits par l'avocat-du-diable après le rapport (le rapport annonçait un état vide à 4bbaacf).
- A7, diff ea6ff55..HEAD : il touche aussi .claude/hooks/notif.py, .claude/settings.json et CLAUDE.md, via be40d4a « notifications ntfy ». Tous les commits ont le même auteur Git (Blumber, nordine.iyad@gmail.com), donc je ne peux pas attribuer be40d4a par l'auteur. Ce commit n'a pas de ligne Co-Authored-By, contrairement aux commits des agents. Iyad doit confirmer qu'il en est l'auteur.
- Le commit f6e0e30 (adversarial) ajoute tests/adversarial/ (39 fichiers sous tests/ et reports/). pytest reste à 72 passed.
- Trades : la somme par fold vaut 198 contre 194 en concaténé (baseline : 749 contre 747). C'est cohérent avec des positions à cheval sur deux folds, mais la cause n'est pas documentée. L'equity est cohérente : le produit des equity par fold vaut 2,494772364002465 et le concaténé 2,494772364002473.
- In-sample contre OOS :
  - Sharpe IS absent (nan) aux folds 3 et 5 ;
  - OOS ≥ IS au fold 8 (0,973 contre 0,686) ;
  - fold 5 à 3,068 avec 1 seul trade ;
  - les folds 3 et 5 n'ont qu'un trade OOS chacun.
- Tuning : seuil d'entrée en bord de grille (0,61) dans 6 folds sur 9 ; la validation interne ne compte que 5 à 7 trades dans certains folds.
- E10 contre B&H : −0,112 de Sharpe ; E10 est sous le B&H en 2021, 2023, 2024 et 2025.
- Essais hors registre : tests mélangés et relances. Ce n'est pas une variante, donc acceptable.
- Budget : 16 essais sur 20 utilisés, 44 restants sur le plafond palier 3.

Vérifications relancées par moi :
- `git rev-parse gonogo-v1` -> 61cb863 ; `git diff gonogo-v1 -- GONOGO.md` -> 0 octet ; `git diff HEAD -- GONOGO.md` -> 0 octet.
- `git tag -l` -> gonogo-v1, palier-0a, palier-1, palier-2.
- `git log ea6ff55..HEAD` -> le harnais 326f933 (12:38:26) précède les configs f7e3eb4 (12:38:37) et E01 9771189 (12:40:09) ; la règle du candidat est dans JOURNAL ea6ff55 (12:32:52), avant les essais.
- `wc -l experiments/REGISTRE.md` -> 16 ; `git log -p REGISTRE | grep -c '^-[^-]'` -> 0.
- `grep -rn "allow_holdout\|holdout_period" experiments/` -> aucune sortie (rc=1).
- Champ `code_dirty` des 21 JSON de résultats -> tous False ; `index_max_read` -> 2025-09-30T23:00 partout.
- `pytest -q -rs` -> 72 passed in 5.94s, 0 skip.
- `git check-ignore .env` -> .env.
- `harness.py E10.json --out` ×2 (scratchpad) -> diff vide ; seule la ligne "commit" diffère de results/E10.json ; Sharpe 0,6695, 194 trades, 9/9 ; garde OK ; contrôle d'égalité baseline/B&H OK.
- `harness.py E10.json --shuffle-seed 4` -> Sharpe 0,7736, 46 trades ; JSON identique à E10_shuf4.json (hors "commit").
- Recalcul DSR indépendant depuis les JSON -> V = 0,0043677, SR0 = 0,118989, z = −3,494, DSR = 0,000238 : identique au rapport.
- Somme des trades par fold contre concat -> 198 contre 194 ; produit des equity par fold contre concat -> 2,494772 = 2,494772.
- `git diff --stat ea6ff55 4bbaacf` hors experiments/ -> .claude/hooks/notif.py, .claude/settings.json, CLAUDE.md (be40d4a).

Tableau GONOGO : N/A (palier 3).

Décision Iyad requise : OUI (confirmer que be40d4a, qui modifie CLAUDE.md et .claude/, est bien de lui). Pas de décision de palier.

Fichiers relus :
- C:\Users\2\bot-trading-ia\experiments\harness.py
- C:\Users\2\bot-trading-ia\experiments\REGISTRE.md
- C:\Users\2\bot-trading-ia\experiments\results\E10.json
- C:\Users\2\bot-trading-ia\experiments\results\E10_shuf4.json
- C:\Users\2\bot-trading-ia\JOURNAL.md
