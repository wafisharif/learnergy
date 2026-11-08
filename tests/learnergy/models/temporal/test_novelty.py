import logging

import numpy as np
import pytest
import torch

import learnergy.utils.exception as e
from learnergy.models.temporal.novelty import NoveltyMonitor, StreamingNoveltyScorer, window_free_energy
from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from learnergy.models.temporal.rtdbn import RTDBN

logging.disable(logging.CRITICAL)


def _model(seed=0):
    torch.manual_seed(seed)
    model = RTVarianceGaussianRBM(n_visible=6, n_hidden=5)

    with torch.no_grad():
        model.W_prime.normal_(0.0, 0.5)
        model.h0.normal_(0.0, 0.5)

    return model


def _windows(n, seed=0, scale=1.0, shift=0.0, steps=10):
    return (np.random.RandomState(seed).randn(n, steps, 6) * scale + shift).astype(np.float32)


def _reference_free_energy(model, x):
    x = torch.as_tensor(x)
    h = model.h0.unsqueeze(0).expand(len(x), -1)
    out = []

    with torch.no_grad():
        for t in range(x.shape[1]):
            out.append(model.energy(x[:, t], h))
            h, _ = model.hidden_sampling(x[:, t], h)

    return torch.stack(out, dim=1).mean(dim=1).numpy()


def test_window_free_energy_matches_independent_computation():
    model = _model()
    x = _windows(30)

    assert np.allclose(window_free_energy(model, x), _reference_free_energy(model, x), atol=1e-5)
    assert np.allclose(window_free_energy(model, x, batch_size=7), window_free_energy(model, x), atol=1e-6)


def test_window_free_energy_uses_first_layer_of_a_stack():
    torch.manual_seed(0)
    stack = RTDBN(
        model=("variance_gaussian", "variance_gaussian"),
        n_visible=6,
        n_hidden=(5, 4),
        steps=(1, 1),
        learning_rate=(0.01, 0.01),
        momentum=(0.0, 0.0),
        decay=(0.0, 0.0),
        temperature=(1.0, 1.0),
    )
    x = _windows(8)

    assert np.allclose(window_free_energy(stack, x), window_free_energy(stack.models[0], x))


def test_window_free_energy_rejects_bad_shapes():
    model = _model()

    with pytest.raises(e.SizeError):
        window_free_energy(model, np.zeros((4, 6)))

    with pytest.raises(e.SizeError):
        window_free_energy(model, np.zeros((0, 5, 6)))

    with pytest.raises(e.ValueError):
        window_free_energy(model, _windows(3), batch_size=0)


def test_shifted_windows_score_higher_than_familiar_ones():
    model = _model()

    assert (
        window_free_energy(model, _windows(50, seed=1, scale=4.0, shift=6.0)).min()
        > window_free_energy(model, _windows(50, seed=2)).max()
    )


def test_calibrated_alarm_rate_on_reference_equals_one_minus_quantile():
    monitor = NoveltyMonitor(_model(), quantile=0.95)
    reference = _windows(200, seed=3)
    threshold = monitor.calibrate(reference)

    assert threshold == monitor.threshold
    assert monitor.n_reference == 200
    assert monitor.flag(reference).mean() == pytest.approx(0.05)


def test_monitor_flags_shifted_windows_and_few_familiar_ones():
    monitor = NoveltyMonitor(_model(), quantile=0.95)
    monitor.calibrate(_windows(400, seed=4))

    assert monitor.flag(_windows(200, seed=5, scale=4.0, shift=6.0)).mean() > 0.95
    assert monitor.flag(_windows(400, seed=6)).mean() < 0.12


def test_monitor_requires_calibration_and_valid_quantile():
    monitor = NoveltyMonitor(_model())

    with pytest.raises(e.BuildError):
        monitor.flag(_windows(3))

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(e.ValueError):
            NoveltyMonitor(_model(), quantile=bad)


def test_monitor_state_round_trip():
    monitor = NoveltyMonitor(_model(), quantile=0.9)
    monitor.calibrate(_windows(100, seed=7))
    restored = NoveltyMonitor(_model()).load_state(monitor.state())
    x = _windows(20, seed=8)

    assert restored.quantile == 0.9 and restored.threshold == monitor.threshold
    assert np.array_equal(restored.flag(x), monitor.flag(x))


def test_streaming_scorer_with_state_resets_reproduces_window_scores():
    model = _model()
    x = _windows(4, seed=9, steps=40)
    scorer = StreamingNoveltyScorer(model, window=10, reset_every=10)
    scorer.reset(4)
    scores = np.stack([scorer.step(x[:, t])[0] for t in range(40)], axis=1)
    expected = np.stack([window_free_energy(model, x[:, s : s + 10]) for s in range(0, 40, 10)], axis=1)

    assert np.allclose(scores[:, 9::10], expected, atol=1e-5)


def test_streaming_scorer_carrying_state_differs_from_resetting():
    model = _model()
    x = _windows(2, seed=10, steps=40)
    carried, reset = StreamingNoveltyScorer(model, window=10), StreamingNoveltyScorer(model, window=10, reset_every=10)

    last_carried = [carried.step(x[:, t])[0] for t in range(40)][-1]
    last_reset = [reset.step(x[:, t])[0] for t in range(40)][-1]

    assert not np.allclose(last_carried, last_reset, atol=1e-4)


def test_streaming_scorer_partial_window_average_and_alarm_rules():
    model = _model()
    x = _windows(1, seed=11, steps=12)
    scorer = StreamingNoveltyScorer(model, window=5, threshold=-1e9)
    out = [scorer.step(x[:, t]) for t in range(12)]
    energy = StreamingNoveltyScorer(model, window=1)
    raw = [float(energy.step(x[:, t])[0][0]) for t in range(12)]

    assert out[2][0][0] == pytest.approx(np.mean(raw[:3]), abs=1e-5)
    assert out[11][0][0] == pytest.approx(np.mean(raw[7:12]), abs=1e-5)
    assert not any(a.any() for _, a in out[:4]) and all(a.all() for _, a in out[4:])
    assert not StreamingNoveltyScorer(model, window=5).step(x[:, 0])[1].any()


def test_streaming_scorer_reset_clears_history_and_validates_arguments():
    model = _model()
    x = _windows(1, seed=12, steps=6)
    scorer = StreamingNoveltyScorer(model, window=3)
    first = [scorer.step(x[:, t])[0] for t in range(6)]
    scorer.reset(1)
    again = [scorer.step(x[:, t])[0] for t in range(6)]

    assert np.allclose(first, again) and scorer.n_frames == 6

    with pytest.raises(e.ValueError):
        StreamingNoveltyScorer(model, window=0)

    with pytest.raises(e.ValueError):
        StreamingNoveltyScorer(model, reset_every=0)
