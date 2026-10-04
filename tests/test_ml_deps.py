"""Smoke tests des dépendances ML (palier 3) : import et déterminisme minimal.

Données synthétiques uniquement, aucun accès au dataset.
"""
import numpy as np


def _donnees_synthetiques(seed=0, n=200, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def test_import_sklearn():
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401

    assert sklearn.__version__


def test_import_lightgbm():
    import lightgbm

    assert lightgbm.__version__


def _proba_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _donnees_synthetiques()
    m = HistGradientBoostingClassifier(max_iter=20, random_state=42)
    m.fit(X, y)
    return m.predict_proba(X)


def _proba_lightgbm():
    import lightgbm as lgb

    X, y = _donnees_synthetiques()
    m = lgb.LGBMClassifier(
        n_estimators=20,
        num_leaves=7,
        random_state=42,
        deterministic=True,
        n_jobs=1,
        verbose=-1,
    )
    m.fit(X, y)
    return m.predict_proba(X)


def test_sklearn_hgb_deterministe():
    p1 = _proba_sklearn()
    p2 = _proba_sklearn()
    assert p1.shape == (200, 2)
    np.testing.assert_array_equal(p1, p2)


def test_lightgbm_deterministe():
    p1 = _proba_lightgbm()
    p2 = _proba_lightgbm()
    assert p1.shape == (200, 2)
    np.testing.assert_array_equal(p1, p2)
