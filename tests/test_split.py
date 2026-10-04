import numpy as np
import pandas as pd
import pytest

from bot.labels import HORIZON
from bot.split import (DEV_END, DEV_START, EMBARGO, HOLDOUT_END, HOLDOUT_START, PURGE,
                       HoldoutAccessError, StandardScaler, dev_period, holdout_period,
                       load_dataset, make_folds, purge_mask, split_fold)

H1 = pd.Timedelta(hours=1)
EXPECTED = [
    ("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-09-30"),
]


def full_frame():
    idx = pd.date_range(DEV_START, HOLDOUT_END, freq="h", name="open_time")
    return pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)


def test_constants():
    assert DEV_END == pd.Timestamp("2025-09-30 23:00", tz="UTC")
    assert HOLDOUT_START == DEV_END + H1
    assert HOLDOUT_END == pd.Timestamp("2026-08-31 23:00", tz="UTC")
    assert PURGE == HORIZON + 1 and EMBARGO == 24


def test_fold_dates_exact():
    folds = make_folds()
    assert len(folds) == 9
    for f, (a, b) in zip(folds, EXPECTED):
        assert f.train_start == DEV_START
        assert f.test_start == pd.Timestamp(f"{a} 00:00", tz="UTC")
        assert f.test_end == pd.Timestamp(f"{b} 23:00", tz="UTC")
        assert f.test_end <= DEV_END
    for f1, f2 in zip(folds, folds[1:]):
        assert f2.test_start == f1.test_end + H1  # tests contigus, sans chevauchement


def test_split_no_overlap_and_purge():
    df = dev_period(full_frame())
    for f in make_folds():
        tr, te = split_fold(df, f)
        assert len(tr.index.intersection(te.index)) == 0
        assert tr.index.max() < te.index.min()
        assert tr.index.max() == f.train_end
        # fenêtre de label du dernier train [t+1, t+1+H] avant le début du test
        assert tr.index.max() + (1 + HORIZON) * H1 < f.test_start
        assert (te.index.min() - tr.index.max()) / H1 >= HORIZON + 1
        # les H+1 bougies précédant le test sont retirées
        purged = df.loc[f.test_start - PURGE * H1: f.test_start - H1]
        assert len(purged) == PURGE and len(purged.index.intersection(tr.index)) == 0
        assert te.index.min() == f.test_start and te.index.max() == f.test_end


def test_purge_mask_embargo_generic():
    idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
    ts, te = idx[100], idx[119]
    keep = purge_mask(idx, ts, te, horizon=4, embargo=24)
    assert keep[:95].all()
    assert not keep[95:144].any()  # purge (5) + test (20) + embargo (24)
    assert keep[144:].all()


def test_holdout_inaccessible_by_default(tmp_path):
    df = full_frame()
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    dev = load_dataset(p)
    assert dev.index.max() == DEV_END and dev.index.min() == DEV_START
    with pytest.raises(HoldoutAccessError):
        holdout_period(df)
    with pytest.raises(HoldoutAccessError):
        holdout_period(df, allow_holdout=1)  # seul True explicite est accepté
    ho = holdout_period(df, allow_holdout=True)
    assert ho.index.min() == HOLDOUT_START and ho.index.max() == HOLDOUT_END
    assert load_dataset(p, allow_holdout=True).index.max() == HOLDOUT_END
    with pytest.raises(HoldoutAccessError):
        split_fold(df, make_folds()[0])  # données post-dev refusées


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(5, 2, size=(1000, 3)), columns=list("abc"),
                      index=pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC"))
    train, test = df.iloc[:700], df.iloc[700:]
    s1 = StandardScaler().fit(train)
    test_mod = test * 1000 + 7
    s2 = StandardScaler().fit(train)
    s2.transform(test_mod)
    assert np.array_equal(s1.mean_, s2.mean_) and np.array_equal(s1.scale_, s2.scale_)
    assert np.allclose(s1.mean_, train.mean().to_numpy())
    assert np.allclose(s1.scale_, train.std(ddof=0).to_numpy())
    z = s1.transform(train)
    assert np.allclose(z.mean().to_numpy(), 0) and np.allclose(z.std(ddof=0).to_numpy(), 1)
    zt = s1.transform(test)
    assert np.allclose(zt.to_numpy(), (test.to_numpy() - s1.mean_) / s1.scale_)
    with pytest.raises(ValueError):
        StandardScaler().fit(np.array([[1.0, np.nan]]))
