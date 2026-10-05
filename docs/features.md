# Features, labels et découpage (palier 1)

Code : `src/bot/features.py`, `src/bot/labels.py`, `src/bot/split.py`.

## Conventions

- Bougies 1h BTCUSDT spot, index `open_time` (UTC). La bougie `t` couvre `[t, t+1h[` ; elle est
  clôturée à `t+1h`. Une feature à l'index `t` est calculable à la clôture de `t`.
- `c[t]`, `o[t]`, `h[t]`, `l[t]`, `v[t]` : close, open, high, low, volume de la bougie `t`.
- `r1[t] = log(c[t] / c[t-1])`.
- Toutes les fenêtres sont finies (aucune EMA, aucun lissage récursif). Chaque valeur est une somme
  dans un ordre fixe sur sa propre fenêtre : le calcul live (`compute_features_live`, sur les
  `LOOKBACK = 200` dernières bougies) est bit à bit égal au calcul batch (`compute_features`).
- **Fenêtre** = nombre de bougies `[t-fenêtre+1, t]` lues. Si l'une d'elles est manquante (ou
  hors données), la feature vaut NaN. Les lignes NaN sont exclues en aval, jamais imputées.
- **Décalage** = 0 pour toutes les features : la valeur à `t` utilise la bougie `t` (clôturée) et
  les précédentes, jamais `t+1` ou au-delà. Index max utilisé = `t`.
- Écart-type : ddof = 1. Moyenne simple = somme / n.

## Features

| feature | formule | fenêtre (bougies) | décalage | données utilisées (indices) |
|---|---|---|---|---|
| ret_1 | log(c[t] / c[t-1]) | 2 | 0 | close [t-1, t] |
| ret_4 | log(c[t] / c[t-4]) | 5 | 0 | close [t-4, t] |
| ret_24 | log(c[t] / c[t-24]) | 25 | 0 | close [t-24, t] |
| ret_168 | log(c[t] / c[t-168]) | 169 | 0 | close [t-168, t] |
| vol_24 | std(r1[t-23..t]) | 25 | 0 | close [t-24, t] |
| vol_168 | std(r1[t-167..t]) | 169 | 0 | close [t-168, t] |
| rsi_14 | 100 · G / (G + P), G = moyenne simple des hausses de close sur 14 variations, P = idem baisses ; 50 si G + P = 0 | 15 | 0 | close [t-14, t] |
| sma_dev_24 | c[t] / mean(c[t-23..t]) − 1 | 24 | 0 | close [t-23, t] |
| sma_dev_168 | c[t] / mean(c[t-167..t]) − 1 | 168 | 0 | close [t-167, t] |
| logvol_z_168 | (lv[t] − mean(lv[t-167..t])) / std(lv[t-167..t]), lv = log(1 + v) ; NaN si std = 0 | 168 | 0 | volume [t-167, t] |
| range_1 | (h[t] − l[t]) / c[t] | 1 | 0 | high, low, close [t] |
| range_24 | mean(range_1[t-23..t]) | 24 | 0 | high, low, close [t-23, t] |
| hour_sin | sin(2π · heure(t) / 24) | 0 | 0 | open_time de t uniquement |
| hour_cos | cos(2π · heure(t) / 24) | 0 | 0 | open_time de t uniquement |
| dow_sin | sin(2π · jour_semaine(t) / 7), lundi = 0 | 0 | 0 | open_time de t uniquement |
| dow_cos | cos(2π · jour_semaine(t) / 7), lundi = 0 | 0 | 0 | open_time de t uniquement |

Fenêtre max = 169 bougies ; `LOOKBACK` = 169 + 31 de marge = 200.

Note : le z-score de volume utilise `log(1 + v)` et non `log(v)`, car le dataset contient des
bougies à volume nul (log(0) = −∞). Pour les volumes BTC horaires (centaines à milliers de BTC),
l'écart avec `log(v)` est négligeable.

## Labels

Convention : le signal est calculé à la clôture de la bougie `t` ; l'exécution se fait à
l'ouverture de `t+1` ; la sortie à l'ouverture de `t+1+H`, avec `H = HORIZON = 4`.

| colonne | définition | données utilisées |
|---|---|---|
| label | 1 si o[t+1+H] > o[t+1], sinon 0 (float : 0.0 / 1.0 / NaN) | open [t+1, t+1+H] |
| fwd_ret | log(o[t+1+H] / o[t+1]) | open [t+1, t+1+H] |

NaN si une bougie de `[t+1, t+1+H]` est manquante (règle stricte : même une bougie intermédiaire)
ou si `t+1+H` dépasse les données. Les labels de la période dev sont calculés sur la seule période
dev : les 5 dernières bougies de dev (2025-09-30 19:00 → 23:00) ont un label NaN, aucune donnée
du holdout n'est lue.

## Découpage et normalisation

- Dev : 2019-01-01 00:00 → 2025-09-30 23:00 UTC. Holdout : 2025-10-01 00:00 → 2026-08-31 23:00 UTC.
  `load_dataset()` renvoie la période dev ; le holdout n'est accessible qu'avec
  `allow_holdout=True` (`load_dataset(..., allow_holdout=True)`, `holdout_period(df, allow_holdout=True)`).
- Walk-forward expanding : train depuis 2019-01-01, 9 folds de test semestriels (2021H1 … 2024H2,
  puis 2025-01-01 → 2025-09-30), fin de fold à 23:00 UTC du dernier jour.
- Purge : toute ligne `t` dont la fenêtre de label `[t+1, t+1+H]` chevauche le test est retirée du
  train, soit les `H+1 = 5` bougies précédant le début du test. Dernière ligne train autorisée :
  `test_start − 6 h`.
- Embargo : 24 bougies après chaque fin de test sont retirées du train (`purge_mask`). En expanding
  window le train précède toujours le test, l'embargo est donc sans effet sur les 9 folds. Il
  s'applique à la frontière dev → holdout : l'évaluation sur le holdout (palier 4) commence à
  `HOLDOUT_EVAL_START = 2025-10-02 00:00 UTC` (les 24 premières bougies du holdout ne sont pas
  évaluées ; elles peuvent servir de contexte de calcul des features).
- Normalisation : `StandardScaler` (numpy), moyenne et écart-type (ddof = 0) ajustés uniquement sur
  le train du fold (`fit`), appliqués tels quels au test (`transform`). Écart-type nul → 1.

## Funding du perpétuel BTCUSDT (palier 3, P3-B6)

Code : `src/bot/funding.py`, `scripts/build_funding.py`, `scripts/check_funding.py`. Le funding ne
sert que de feature : on trade toujours le spot, long / flat.

- Source : archives mensuelles publiques `data/futures/um/monthly/fundingRate/BTCUSDT/` sur
  data.binance.vision, SHA256 vérifié par `.CHECKSUM` (logique `bot.data.fetch_archive` réutilisée).
  Pas de repli journalier : une archive mensuelle absente (404) fait échouer le build.
- Période : 2020-01 → 2026-08. Le contrat date de 2019-09, mais les archives 2019-09 → 2019-12
  n'existent pas (404 constaté le 2026-10-05) : première archive = 2020-01.
- Colonnes des archives : `calc_time` (→ `fundingTime`, UTC, ms ou µs détecté),
  `last_funding_rate` (→ `fundingRate`, float64), `funding_interval_hours`. Pas de prix de
  marquage dans ces archives (colonne `markPrice` conservée si elle apparaît).
- Nettoyage : tri, doublons de `fundingTime` supprimés et comptés, intervalles contrôlés (8 h
  attendues ; écarts > 1 s listés, gigue de quelques ms comptée à part), valeurs |rate| > 0,01
  comptées et conservées. Sortie : `data/processed/btcusdt_funding.parquet`.
- Verrou : `load_funding()` ne renvoie que les règlements avec `fundingTime <= DEV_END`
  (2025-09-30 23:00 UTC). `load_funding(..., allow_holdout=True)` (strictement `True`) donne
  aussi la période réservée. `assert_no_holdout(df)` lève `HoldoutAccessError` sinon.

### Règle de disponibilité (alignement 1 h)

La bougie `t` (index `open_time`) clôture à `t + 1 h`. À la clôture de `t`, le taux utilisable est
celui du **dernier règlement avec `fundingTime <= t + 1 h`** (`funding_on_grid`). Jamais de
règlement avec `fundingTime > t + 1 h` ; NaN avant le premier règlement. `age_h` = délai en heures
entre ce règlement et `t + 1 h`.
Exemples : le règlement de 08:00:00.000 est utilisable à la clôture de la bougie 07:00 ; un
règlement horodaté 08:00:00.002 (gigue présente dans les archives) ne l'est qu'à la clôture de la
bougie 08:00. Pas d'arrondi des horodatages.

### Features de funding

Fenêtres comptées en **règlements** (pas en bougies). La valeur à la bougie `t` est celle calculée
au dernier règlement disponible `i` (règle ci-dessus) sur les règlements `[i-n+1, i]`.
Décalage = 0 en règlements ; en bougies, la valeur ne change qu'à la bougie dont la clôture suit
un nouveau règlement. NaN si moins de `n` règlements ou NaN dans la fenêtre.

| feature | formule | fenêtre (règlements) | durée nominale |
|---|---|---|---|
| fr_cur | r[i] | 1 | 8 h |
| fr_mean3 | sum(r[i-2..i]) / 3 | 3 | 1 jour |
| fr_mean21 | sum(r[i-20..i]) / 21 | 21 | 7 jours |
| fr_z90 | (r[i] − m) / s, m = sum(r[i-89..i]) / 90, s = écart-type ddof = 1 ; NaN si s ≤ 1e-12 | 90 | 30 jours |
| fr_sum21 | sum(r[i-20..i]) | 21 | 7 jours |
| fr_age_h | (t + 1 h − fundingTime[i]) en heures | – | – |

`s ≤ 1e-12` → NaN : les taux sont publiés à 1e-8 près ; une fenêtre constante (ex. 90 × 0,0001)
donne un écart-type non nul de l'ordre de 1e-20 par arrondi flottant, qui produirait un z-score
arbitraire.

Cœur unique `funding_features_core(rates)` (lit les `FUNDING_LOOKBACK = 90` derniers taux), appelé
en lot par `funding_features(funding, index_1h)` (règlement par règlement) et en direct par
`funding_features_live(recent, t)` (règlements récents ; ceux avec `fundingTime > t + 1 h` sont
ignorés). Égalité live = lot vérifiée bit à bit (`tests/test_funding.py`, `scripts/check_funding.py`).

## Agrégation 4 h et journalière

Module : `src/bot/resample.py` (définition reprise de `experiments/agg4h.py`, validée en P3-V5).

- Entrée : bougies 1 h sur une grille horaire complète et régulière, index `open_time` UTC.
  Une heure est **manquante** si `missing` est vrai ou si une valeur OHLCV est NaN.
- `resample_ohlcv(df_1h, rule)`, `rule` ∈ {`"4h"`, `"1d"`} ; k = 4 ou 24 heures.
- **Fenêtre et alignement** : la bougie d'horodatage `T` (open_time de sa 1re heure) couvre les
  heures `T, T+1h, …, T+(k-1)h`. `T` est aligné sur la grille UTC : 00/04/08/12/16/20 h en 4 h,
  00:00 en journalier (jour UTC 00:00 → 23:59). Une entrée en autre fuseau est convertie en UTC.
- **Valeurs** : open = open(T) ; high = max des k high ; low = min des k low ;
  close = close(T+(k-1)h) ; volume = somme des k volumes de gauche à droite (ordre fixe).
- **Bougies manquantes** : une seule heure manquante rend la bougie manquante (`missing=True`,
  OHLCV = NaN), en 4 h comme en journalier.
- **Pas de bougie partielle** : une bougie n'est émise que si ses k heures sont dans l'entrée
  (début ou fin de fenêtre au milieu d'une bougie → bougie non émise).
- **Décalage** : la bougie `T` est clôturée à `T + k h` (`close_time(T, rule)`) et n'est
  utilisable qu'à partir de cet instant ; elle ne lit que des heures `< T + k h`.
- **Live** : `resample_ohlcv_live(hours, rule, n_bars=None)` prend les dernières heures
  clôturées et renvoie les bougies complètes (les `n_bars` dernières si demandé). Même cœur
  `_aggregate` que le lot.
- Vérifications : `tests/test_resample.py` (cas à la main, manquantes, partielles, anti-fuite,
  alignement, live = lot sur 500 instants) ; `scripts/check_resample.py` (période dev : égalité
  bit à bit avec `experiments/agg4h.py`, live = lot sur 2000 instants en 4 h et en 1 j).
