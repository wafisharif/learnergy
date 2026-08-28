"""Comprehensive sample() tests against the actual fork's import paths."""
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
logging.disable(logging.CRITICAL)

torch.manual_seed(0)
np.random.seed(0)


class _SimpleSeqDataset(Dataset):
    """Minimal (sample, target) dataset for RTRBM.fit()'s DataLoader loop."""

    def __init__(self, data: np.ndarray):
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], 0


def test_rtrbm_shape_and_binary():
    from learnergy.models.temporal.rtrbm import RTRBM
    model = RTRBM(n_visible=8, n_hidden=6)
    s = model.sample(n_samples=5, n_steps=12, gibbs_steps=20)
    assert s.shape == (5, 12, 8)
    assert torch.all((s == 0) | (s == 1))
    print("RTRBM: shape + binary values: PASS")


def test_rtgaussian_shape_and_continuous():
    from learnergy.models.temporal.rt_gaussian_rbm import RTGaussianRBM
    model = RTGaussianRBM(n_visible=8, n_hidden=6, normalize=False, input_normalize=False)
    s = model.sample(n_samples=5, n_steps=12, gibbs_steps=20)
    assert s.shape == (5, 12, 8)
    frac_binary = ((s == 0) | (s == 1)).float().mean().item()
    assert frac_binary < 0.5, f"Gaussian sample looks binary ({frac_binary:.2f} at 0/1) -- wrong visible_sampling used"
    print(f"RTGaussianRBM: shape + continuous values (frac at exactly 0/1={frac_binary:.3f}): PASS")


def test_rtvariance_shape_and_continuous():
    from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
    model = RTVarianceGaussianRBM(n_visible=8, n_hidden=6)
    s = model.sample(n_samples=5, n_steps=12, gibbs_steps=20)
    assert s.shape == (5, 12, 8)
    frac_binary = ((s == 0) | (s == 1)).float().mean().item()
    assert frac_binary < 0.5
    print(f"RTVarianceGaussianRBM: shape + continuous values (frac at exactly 0/1={frac_binary:.3f}): PASS")


def test_rtvariance_respects_learned_sigma():
    """Generated sample variance should track a forced per-feature sigma."""
    from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
    model = RTVarianceGaussianRBM(n_visible=6, n_hidden=4)
    with torch.no_grad():
        model.sigma.copy_(torch.tensor([0.05, 0.05, 0.05, 3.0, 3.0, 3.0]))
    s = model.sample(n_samples=300, n_steps=1, gibbs_steps=30)  # (300, 1, 6)
    per_feature_std = s[:, 0, :].std(dim=0)
    low_sigma_std = per_feature_std[:3].mean().item()
    high_sigma_std = per_feature_std[3:].mean().item()
    print(f"RTVarianceGaussianRBM: per-feature sample std, low-sigma features={low_sigma_std:.3f}, "
          f"high-sigma features={high_sigma_std:.3f}")
    assert high_sigma_std > low_sigma_std, (
        "Generated variance does not track learned sigma -- sample() may be using "
        "the deterministic mean instead of the actual noisy draw (the tuple-order risk)."
    )
    print("RTVarianceGaussianRBM: generated variance tracks learned sigma: PASS")


def test_rtgaussian_sample_is_not_degenerate_constant():
    """Regression check: repeated sample() calls must not be deterministic."""
    from learnergy.models.temporal.rt_gaussian_rbm import RTGaussianRBM
    model = RTGaussianRBM(n_visible=6, n_hidden=4, normalize=False, input_normalize=False)
    s1 = model.sample(n_samples=3, n_steps=8, gibbs_steps=20)
    s2 = model.sample(n_samples=3, n_steps=8, gibbs_steps=20)
    assert not torch.allclose(s1, s2), "Two calls produced identical output -- sample() is deterministic (old bug)"
    within_call_std = s1[0].std(dim=0).mean().item()
    assert within_call_std > 1e-4, "Trajectory is nearly constant across time -- likely degenerate"
    print(f"RTGaussianRBM: not degenerate/deterministic (within-seq std={within_call_std:.4f}): PASS")


def test_gibbs_steps_actually_matters():
    """gibbs_steps should measurably change the sampled distribution."""
    from learnergy.models.temporal.rtrbm import RTRBM
    torch.manual_seed(1)
    model = RTRBM(n_visible=10, n_hidden=8)
    with torch.no_grad():
        model.a.copy_(torch.linspace(-3, 3, 10))

    torch.manual_seed(2)
    s_short = model.sample(n_samples=500, n_steps=1, gibbs_steps=1)
    torch.manual_seed(2)
    s_long = model.sample(n_samples=500, n_steps=1, gibbs_steps=200)

    mean_short = s_short[:, 0, :].mean(dim=0)
    mean_long = s_long[:, 0, :].mean(dim=0)
    diff = (mean_short - mean_long).abs().mean().item()
    print(f"RTRBM: mean abs diff in per-feature activation rate, gibbs_steps=1 vs 200: {diff:.4f}")
    assert diff > 1e-3, "gibbs_steps had no measurable effect -- parameter may be a no-op"
    print("RTRBM: gibbs_steps measurably changes the sampled distribution: PASS")


def test_rtdbn_sample_delegates():
    from learnergy.models.temporal.rtdbn import RTDBN
    model = RTDBN(model=("variance_gaussian",), n_visible=6, n_hidden=(4,))
    s = model.sample(n_samples=3, n_steps=5, gibbs_steps=10)
    assert s.shape == (3, 5, 6)
    print("RTDBN: sample() delegates correctly to single RTRBM layer: PASS")


def test_rtdbn_multilayer_sample_works():
    """RTDBN.sample() generates via top-layer Gibbs + top-down ancestral pass for n_layers > 1."""
    from learnergy.models.temporal.rtdbn import RTDBN
    model = RTDBN(
        model=("variance_gaussian", "variance_gaussian"),
        n_visible=6,
        n_hidden=(4, 3),
        steps=(1, 1),
        learning_rate=(0.001, 0.001),
        momentum=(0.0, 0.0),
        decay=(0.0, 0.0),
        temperature=(1.0, 1.0),
    )
    s = model.sample(n_samples=2, n_steps=3, gibbs_steps=10)
    assert s.shape == (2, 3, 6)
    assert torch.isfinite(s).all()
    print("RTDBN: multi-layer sample() generates correctly-shaped, finite output: PASS")


def test_trained_model_samples_resemble_training_structure():
    """Train on structured synthetic data and confirm sample() reproduces it."""
    from learnergy.models.temporal.rtrbm import RTRBM

    torch.manual_seed(3)
    np.random.seed(3)

    n_visible = 6
    seq_len = 6
    n_seqs = 200
    data = np.zeros((n_seqs, seq_len, n_visible), dtype=np.float32)
    for t in range(seq_len):
        if t % 2 == 0:
            data[:, t, :3] = 1.0
        else:
            data[:, t, 3:] = 1.0
    flip_mask = np.random.rand(*data.shape) < 0.03
    data[flip_mask] = 1.0 - data[flip_mask]

    ds = _SimpleSeqDataset(data)

    model = RTRBM(n_visible=n_visible, n_hidden=12, learning_rate=0.05)
    model.fit(ds, batch_size=32, epochs=40)

    samples = model.sample(n_samples=100, n_steps=seq_len, gibbs_steps=50)
    even_first_half = samples[:, 0::2, :3].mean().item()
    even_second_half = samples[:, 0::2, 3:].mean().item()
    odd_first_half = samples[:, 1::2, :3].mean().item()
    odd_second_half = samples[:, 1::2, 3:].mean().item()
    print(f"Trained RTRBM sample() stats: even-t first-half ON rate={even_first_half:.2f} "
          f"(expect high), even-t second-half ON rate={even_second_half:.2f} (expect low)")
    print(f"                              odd-t first-half ON rate={odd_first_half:.2f} (expect low), "
          f"odd-t second-half ON rate={odd_second_half:.2f} (expect high)")
    assert even_first_half > even_second_half, "sample() did not learn the even-timestep pattern"
    assert odd_second_half > odd_first_half, "sample() did not learn the odd-timestep pattern"
    print("RTRBM: trained-model sample() reproduces learned temporal structure: PASS")


def test_gaussian_gibbs_sampling_bugfix_visible_states_not_probs():
    """Regression test: gibbs_sampling() must return raw states, not sigmoid(probs)."""
    from learnergy.models.temporal.rt_gaussian_rbm import RTGaussianRBM
    torch.manual_seed(4)
    model = RTGaussianRBM(n_visible=6, n_hidden=4, normalize=False, input_normalize=False)
    with torch.no_grad():
        model.a.copy_(torch.tensor([5.0, -5.0, 8.0, -8.0, 10.0, -10.0]))
    v = torch.randn(4, 6) * 3
    h_prev = model.h0.unsqueeze(0).expand(4, -1)
    _, _, _, _, visible_states = model.gibbs_sampling(v, h_prev)
    frac_outside_unit_interval = ((visible_states < 0) | (visible_states > 1)).float().mean().item()
    print(f"RTGaussianRBM.gibbs_sampling: fraction of returned values outside (0,1)={frac_outside_unit_interval:.2f}")
    assert frac_outside_unit_interval > 0.3, (
        "gibbs_sampling's returned negative particle looks sigmoid-bounded -- "
        "the visible_probs bug may still be present"
    )
    print("RTGaussianRBM.gibbs_sampling: confirmed returns raw visible_states, not sigmoid(visible_probs): PASS")


if __name__ == "__main__":
    test_rtrbm_shape_and_binary()
    test_rtgaussian_shape_and_continuous()
    test_rtvariance_shape_and_continuous()
    test_rtvariance_respects_learned_sigma()
    test_rtgaussian_sample_is_not_degenerate_constant()
    test_gibbs_steps_actually_matters()
    test_rtdbn_sample_delegates()
    test_rtdbn_multilayer_sample_works()
    test_trained_model_samples_resemble_training_structure()
    test_gaussian_gibbs_sampling_bugfix_visible_states_not_probs()
    print("\nAll fork comprehensive sample() tests passed.")
