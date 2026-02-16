import numpy as np

from train_multimodal_retrieval import _ksg_mi, _mixed_mi_discrete_continuous


def test_ksg_mi_runs_with_per_sample_radii():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    y = rng.normal(size=(50, 2))
    value = _ksg_mi(x, y, k=3)
    assert np.isfinite(value)


def test_mixed_mi_runs_with_per_sample_radii():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(60, 4))
    m = np.repeat(np.arange(6), 10)
    value = _mixed_mi_discrete_continuous(m, z, k=3)
    assert np.isfinite(value)
