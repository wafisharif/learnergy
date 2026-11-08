import logging

import numpy as np
import pytest
import torch

import learnergy.utils.exception as e
from learnergy.models.temporal.rt_gated_variance_gaussian_rbm import RTGatedVarianceGaussianRBM
from learnergy.models.temporal.rt_gaussian_rbm import RTGaussianRBM
from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from learnergy.models.temporal.rtdbn import RTDBN
from learnergy.models.temporal.rtrbm import RTRBM
from learnergy.models.temporal.streaming import (
    RidgeReadout,
    StreamingEncoder,
    StreamingPhaseEstimator,
    circular_error,
    phase_to_unit_circle,
    unit_circle_to_phase,
)

logging.disable(logging.CRITICAL)


def _excite(model, scale=0.5, seed=0):
    torch.manual_seed(seed)

    with torch.no_grad():
        model.W_prime.normal_(0.0, scale)
        model.h0.normal_(0.0, scale)

    return model


def _gaussian(n_visible=6, n_hidden=5):
    torch.manual_seed(0)

    return _excite(RTVarianceGaussianRBM(n_visible=n_visible, n_hidden=n_hidden))


def _stack():
    torch.manual_seed(0)
    model = RTDBN(
        model=("variance_gaussian", "variance_gaussian"),
        n_visible=6,
        n_hidden=(5, 4),
        steps=(1, 1),
        learning_rate=(0.01, 0.01),
        momentum=(0.0, 0.0),
        decay=(0.0, 0.0),
        temperature=(1.0, 1.0),
    )

    for i, layer in enumerate(model.models):
        _excite(layer, seed=i)

    return model


def test_stepping_equals_forward_for_variance_gaussian():
    model = _gaussian()
    x = torch.randn(4, 12, 6)
    stream = StreamingEncoder(model)
    ref = model.forward(x)
    stream.reset(4)

    for t in range(12):
        assert torch.allclose(stream.step(x[:, t]), ref[:, t], atol=1e-6)


def test_stepping_equals_forward_for_bernoulli_rtrbm():
    torch.manual_seed(0)
    model = _excite(RTRBM(n_visible=6, n_hidden=5))
    x = (torch.rand(3, 10, 6) > 0.5).float()
    stream = StreamingEncoder(model)
    stream.reset(3)
    h_prev = model.h0.unsqueeze(0).expand(3, -1)

    for t in range(10):
        ref, _ = model.hidden_sampling(x[:, t], h_prev)
        h_prev = ref
        assert torch.allclose(stream.step(x[:, t]), ref, atol=1e-6)


def test_stepping_equals_forward_for_gaussian_without_input_normalization():
    torch.manual_seed(0)
    model = _excite(RTGaussianRBM(n_visible=6, n_hidden=5, input_normalize=False))

    assert StreamingEncoder(model).max_abs_difference(torch.randn(3, 10, 6)) < 1e-6


def test_stacked_rtdbn_matches_layer_by_layer_forward():
    model = _stack()
    x = torch.randn(5, 14, 6)

    assert StreamingEncoder(model).max_abs_difference(x) < 1e-6
    assert StreamingEncoder(model).run(x).shape == (5, 14, 4)


def test_one_dimensional_frame_returns_one_dimensional_output():
    model = _gaussian()
    x = torch.randn(1, 8, 6)
    stream = StreamingEncoder(model)
    stream.reset()
    out = [stream.step(x[0, t]) for t in range(8)]

    assert out[0].shape == (5,)
    assert torch.allclose(torch.stack(out), model.forward(x)[0], atol=1e-6)


def test_step_without_reset_starts_from_h0():
    model = _gaussian()
    frame = torch.randn(2, 6)
    first = StreamingEncoder(model).step(frame)
    expected = model.forward(frame.unsqueeze(1))[:, 0]

    assert torch.allclose(first, expected, atol=1e-6)


def test_state_is_carried_between_steps_and_reset_restores_h0():
    model = _gaussian()
    x = torch.randn(2, 6, 6)
    stream = StreamingEncoder(model)
    stream.reset(2)
    first = [stream.step(x[:, t]) for t in range(6)]
    carried = stream.step(x[:, 0])
    stream.reset(2)
    fresh = stream.step(x[:, 0])

    assert not torch.allclose(carried, fresh, atol=1e-4)
    assert torch.allclose(fresh, first[0], atol=1e-6)
    assert stream.n_steps == 1


def test_without_recurrence_history_does_not_matter():
    model = _gaussian()

    with torch.no_grad():
        model.W_prime.zero_()

    x = torch.randn(2, 6, 6)
    stream = StreamingEncoder(model)
    stream.reset(2)
    [stream.step(x[:, t]) for t in range(5)]
    carried = stream.step(x[:, 0])
    stream.reset(2)

    assert torch.allclose(carried, stream.step(x[:, 0]), atol=1e-6)


def test_track_energy_matches_direct_energy_before_state_update():
    model = _gaussian()
    x = torch.randn(3, 5, 6)
    stream = StreamingEncoder(model, track_energy=True)
    hidden, energy = stream.run(x, return_energy=True)
    h_prev = model.h0.unsqueeze(0).expand(3, -1)
    expected = []

    with torch.no_grad():
        for t in range(5):
            expected.append(model.energy(x[:, t], h_prev))
            h_prev = hidden[:, t]

    assert torch.allclose(energy, torch.stack(expected, dim=1), atol=1e-6)


def test_return_energy_requires_track_energy():
    with pytest.raises(e.ValueError):
        StreamingEncoder(_gaussian()).run(torch.randn(1, 3, 6), return_energy=True)


def test_wrong_feature_count_and_batch_are_rejected():
    stream = StreamingEncoder(_gaussian())
    stream.reset(2)

    with pytest.raises(e.SizeError):
        stream.step(torch.randn(2, 7))

    with pytest.raises(e.SizeError):
        stream.step(torch.randn(3, 6))

    with pytest.raises(e.SizeError):
        stream.run(torch.randn(3, 6))

    with pytest.raises(e.ValueError):
        stream.reset(0)


def test_unsupported_models_are_rejected():
    with pytest.raises(e.TypeError):
        StreamingEncoder(RTGatedVarianceGaussianRBM(n_visible=4, n_hidden=3))

    with pytest.raises(e.ValueError):
        StreamingEncoder(RTGaussianRBM(n_visible=4, n_hidden=3))

    with pytest.raises(e.TypeError):
        StreamingEncoder(torch.nn.Linear(3, 3))


def test_gradients_are_not_tracked():
    model = _gaussian()
    stream = StreamingEncoder(model)

    assert not stream.step(torch.randn(1, 6)).requires_grad


def test_ridge_readout_matches_sklearn():
    linear_model = pytest.importorskip("sklearn.linear_model")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    rng = np.random.RandomState(0)
    x = rng.randn(200, 7)
    x[:, 3] = 2.0
    y = x[:, :2] @ rng.randn(2, 2) + 0.1 * rng.randn(200, 2) + 1.0
    z = preprocessing.StandardScaler().fit(x)
    ref = linear_model.Ridge(alpha=3.0).fit(z.transform(x), y).predict(z.transform(x[:20]))

    assert np.allclose(RidgeReadout(alpha=3.0).fit(x, y).predict(x[:20]), ref, atol=1e-8)


def test_ridge_readout_state_round_trip_and_errors():
    rng = np.random.RandomState(1)
    x, y = rng.randn(50, 4), rng.randn(50, 2)
    readout = RidgeReadout(alpha=0.5).fit(x, y)
    restored = RidgeReadout().load_state(readout.state())

    assert np.allclose(restored.predict(x), readout.predict(x))

    with pytest.raises(e.BuildError):
        RidgeReadout().predict(x)

    with pytest.raises(e.ValueError):
        RidgeReadout(alpha=-1.0)

    with pytest.raises(e.SizeError):
        RidgeReadout().fit(x, y[:10])


def test_phase_conversions_round_trip_and_wrap():
    phase = np.array([0.0, 12.5, 50.0, 99.0])

    assert np.allclose(unit_circle_to_phase(phase_to_unit_circle(phase)), phase, atol=1e-9)
    assert np.allclose(circular_error(np.array([1.0, 99.0, 25.0]), np.array([99.0, 1.0, 75.0])), [2.0, 2.0, 50.0])
    assert unit_circle_to_phase(np.array([-1e-12, 1.0])) < 100.0


def _cycles(n, steps, period, n_channels=6, seed=0):
    rng = np.random.RandomState(seed)
    offsets = rng.uniform(0, 2 * np.pi, n_channels)
    start = rng.randint(0, period, n)
    t = start[:, None] + np.arange(steps)[None, :]
    x = np.sin(2 * np.pi * t[:, :, None] / period + offsets[None, None, :]).astype(np.float32)
    phase = (t % period) / period * 100.0

    return x, phase


def _estimator():
    torch.manual_seed(0)
    model = RTVarianceGaussianRBM(n_visible=6, n_hidden=32)

    with torch.no_grad():
        model.W.normal_(0.0, 1.0)
        model.W_prime.normal_(0.0, 0.3)

    return StreamingPhaseEstimator(StreamingEncoder(model))


def test_phase_estimator_recovers_a_noiseless_cycle():
    x, phase = _cycles(400, 24, period=20)
    est = _estimator().fit(x, phase)
    xt, pt = _cycles(100, 24, period=20, seed=0)
    xt, pt = xt[::-1].copy(), pt[::-1].copy()
    pred, conf = est.predict_sequence(xt)

    assert np.median(circular_error(pred[:, 4:], pt[:, 4:])) < 5.0
    assert pred.shape == pt.shape == conf.shape
    assert pred.min() >= 0.0 and pred.max() < 100.0


def test_phase_estimator_step_equals_sequence_prediction():
    x, phase = _cycles(300, 20, period=20)
    est = _estimator().fit(x, phase)
    xt, _ = _cycles(5, 20, period=20, seed=0)
    seq_pred, seq_conf = est.predict_sequence(xt)
    est.reset(5)
    stepped = [est.step(xt[:, t]) for t in range(20)]

    assert np.allclose(np.stack([p for p, _ in stepped], axis=1), seq_pred, atol=1e-6)
    assert np.allclose(np.stack([c for _, c in stepped], axis=1), seq_conf, atol=1e-6)


def test_phase_estimator_ignores_nan_phase_and_burn_in_frames():
    x, phase = _cycles(300, 20, period=20)
    clean = _estimator().fit(x, phase)
    dirty_phase = phase.copy()
    dirty_phase[:, :2] = np.nan
    dirty_phase[::7, 9] = np.nan
    est = _estimator().fit(x, dirty_phase, burn_in=2)
    probe, _ = _cycles(10, 20, period=20, seed=0)

    assert np.isfinite(est.predict_sequence(probe)[0]).all()
    assert est.readout.is_fitted and clean.readout.is_fitted


def test_phase_estimator_rejects_bad_inputs():
    x, phase = _cycles(50, 10, period=10)

    with pytest.raises(e.SizeError):
        _estimator().fit(x, phase[:, :5])

    with pytest.raises(e.ValueError):
        _estimator().fit(x, phase, burn_in=10)

    with pytest.raises(e.BuildError):
        _estimator().step(x[:, 0])
