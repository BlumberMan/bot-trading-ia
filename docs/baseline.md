# Baseline de référence (palier 2)

Stratégie simple, figée AVANT toute exécution sur les vraies données. C'est la barre
que le modèle IA devra battre au palier 3 (GONOGO : Sharpe net modèle − Sharpe net
baseline > 0,3, mêmes folds, mêmes coûts). Aucun paramètre n'est optimisé.

## Règle

- Instrument : BTCUSDT spot, bougies 1h (`data/processed/btcusdt_1h.parquet`).
- Positions : long / flat uniquement (0 ou 1), pas de short, pas de levier.
- Signal à la clôture de la bougie t :
  - `SMA168[t]` = moyenne arithmétique des 168 clôtures `close[t-167] … close[t]`
    (bougie t incluse, clôturée ; 168 h = 7 jours) ;
  - `signal[t] = 1` si `close[t] > SMA168[t]` (strictement), sinon `0` ;
  - si l'une des 168 bougies de la fenêtre manque (ou historique insuffisant),
    `SMA168[t]` et `signal[t]` valent NaN : la position précédente est conservée
    (aucun trade forcé).
- Paramètre : fenêtre 168, fixée a priori (une semaine), aucune optimisation.
- La SMA utilise le même cœur de calcul à fenêtre finie que les features
  (`bot.features._rmean`), donc identique en backtest et en live.

## Exécution

- Décision à la clôture de t, exécution à l'ouverture de t+1 (`exec_delay = 1`).
- La position décidée à t est tenue de `open[t+1]` à `open[t+2]` ; rendement de la
  bougie t : `open[t+1] / open[t] − 1` appliqué à la position tenue pendant t.
- Bougie manquante (open NaN) : pas d'exécution possible, la position est tenue ;
  le P&L du trou est réalisé à la première ouverture disponible ; une cible en
  attente est exécutée à cette première ouverture disponible.
- Dernière bougie de la période dev (2025-09-30 23:00 UTC) : l'ouverture suivante
  appartient au holdout et n'est pas lue ; rendement 0 (marquage au dernier open).

## Coûts

- 0,10 % de frais + 0,05 % de spread/slippage = **0,15 % par côté**
  (`cost_per_side = 0.0015`), multipliés par `|Δposition|` et prélevés sur l'equity
  au moment de l'exécution. Un aller-retour complet coûte donc ≈ 0,30 %.

## Période évaluée

- Uniquement les 9 folds de test walk-forward de la période dev
  (`bot.split.make_folds`) et leur concaténation
  **2021-01-01 00:00 → 2025-09-30 23:00 UTC**. Le holdout (≥ 2025-10-01) n'est pas lu
  (`load_dataset()` par défaut).
- La SMA est calculée sur tout l'historique dev (depuis 2019-01-01) ; la SMA à t
  n'utilise que des bougies ≤ t. La baseline n'a pas d'entraînement : les folds
  servent seulement de fenêtres d'évaluation.
- La stratégie tourne en continu : au début de chaque fenêtre, la position tenue à
  la bougie précédente est héritée sans coût (son coût d'entrée appartient à la
  fenêtre précédente, ou à 2020 pour le fold 1). Les fenêtres contiguës chaînées
  donnent donc exactement le backtest continu 2021-01-01 → 2025-09-30. Aucune
  liquidation forcée en fin de fenêtre : la position ouverte est marquée à l'open
  qui suit la dernière bougie (s'il est dans la période dev).

## Benchmark buy & hold

- Mêmes fenêtres, mêmes coûts : achat (position 1) à l'ouverture de la première
  bougie évaluée, coût 0,15 % ; vente au marquage final (open suivant la dernière
  bougie évaluée s'il est dans la période dev, sinon dernier open connu), coût 0,15 %.
- Chaque fold est un buy & hold indépendant (achat et vente dans le fold) ; la
  période concaténée est un seul buy & hold 2021-01-01 → 2025-09-30.

## Métriques (`bot.metrics`)

- Sharpe (définition GONOGO) : rendements journaliers (jours UTC) de l'equity nette,
  taux sans risque 0, sur toute la période évaluée, annualisé par √365 (ddof = 1).
- Drawdown max sur l'equity nette par bougie ; trades = aller-retours (segments de
  position > 0) ; win rate = part des trades à rendement net > 0 ; profit factor =
  somme des gains nets / |somme des pertes nettes| ; exposition = part des bougies
  en position ; rendement total ; rendement annualisé (année de 8760 bougies).
- Un trade à cheval sur deux folds apparaît dans les deux (marqué `open_at_end`
  puis `open_at_start`), clippé à chaque fenêtre ; il n'est compté qu'une fois dans
  la période concaténée.

## Lancer

```powershell
.venv\Scripts\python scripts\run_baseline.py
```

Résultats : `results/baseline_metrics.json`.
