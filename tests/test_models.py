from __future__ import annotations

import numpy as np
import pytest
import torch

from red_herring.baselines import BASELINES, PCALogReg, project
from red_herring.deep import (
    ChromaCNNConfig,
    ChromatogramCNN,
    DeepCNN,
    TrainConfig,
    default_device,
    rt_shift,
)


def test_project():
    X = np.arange(12, dtype=float).reshape(1, 3, 4)
    np.testing.assert_array_equal(project(X, "mean"), [[4, 5, 6, 7]])
    np.testing.assert_array_equal(project(X, "max"), [[8, 9, 10, 11]])
    with pytest.raises(ValueError, match="Unknown feature"):
        project(X, "sum")


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("feature", ["mean", "max"])
def test_baselines_fit_separable_species(spectra, name, feature):
    model = BASELINES[name](seed=0, feature=feature).fit(spectra.X, spectra.y)
    out = model.outputs(spectra.X)
    probs = out["probs"]
    assert probs.shape == (len(spectra), 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs.argmax(axis=1) == spectra.y).mean() > 0.9


def test_pca_logreg_exposes_logits_and_embedding(spectra):
    subset = np.arange(0, len(spectra), 8)  # 8 samples spanning all species
    assert len(np.unique(spectra.y[subset])) == 4
    out = PCALogReg(n_components=50).fit(spectra.X[subset], spectra.y[subset]).outputs(spectra.X)
    assert out["logits"].shape == (len(spectra), 4)
    assert out["emb"].shape == (len(spectra), 7)  # capped at n_train - 1


def test_rt_shift_moves_bins_and_zero_fills():
    x = torch.arange(1, 6, dtype=torch.float32).repeat(64, 1)[..., None]  # [64, 5, 1]
    out = rt_shift(x, max_shift=2, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    for row in out[..., 0]:
        nonzero = row[row > 0]
        assert len(nonzero) >= 3
        assert torch.all(torch.diff(nonzero) == 1)  # contiguous, order preserved
    assert not torch.equal(out, x)
    assert torch.equal(rt_shift(x, 0), x)


def test_chromatogram_cnn_shapes():
    model = ChromatogramCNN(ChromaCNNConfig(mz_max=24, cnn_channels=8, num_classes=3))
    x = torch.rand(5, 12, 24)
    assert model.embed(x).shape == (5, 16)
    assert model(x).shape == (5, 3)


def test_default_device_is_a_torch_device():
    assert isinstance(default_device(), torch.device)


def test_deep_cnn_learns_separable_species(spectra):
    model = DeepCNN(
        seed=0,
        train_cfg=TrainConfig(epochs=30, batch_size=8, max_shift=1),
        model_cfg=ChromaCNNConfig(mz_max=24, cnn_channels=16, num_classes=4),
        device=torch.device("cpu"),
    )
    out = model.fit(spectra.X, spectra.y).outputs(spectra.X, batch_size=7)
    assert out["probs"].shape == (len(spectra), 4)
    assert out["logits"].shape == (len(spectra), 4)
    assert out["emb"].shape == (len(spectra), 32)
    np.testing.assert_allclose(out["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert (out["probs"].argmax(axis=1) == spectra.y).mean() > 0.9


def test_deep_cnn_skips_singleton_batches(spectra):
    """A trailing batch of one sample would break BatchNorm in train mode."""
    model = DeepCNN(
        train_cfg=TrainConfig(epochs=1, batch_size=len(spectra) - 1),
        model_cfg=ChromaCNNConfig(mz_max=24, cnn_channels=4, num_classes=4),
        device=torch.device("cpu"),
    )
    model.fit(spectra.X, spectra.y)
