# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Causal frame-by-frame inference for trained RTRBM-family encoders, with a ridge phase readout."""

import numpy as np
import torch

import learnergy.utils.exception as e
from learnergy.utils import logging

logger = logging.get_logger(__name__)


def _check_layer(layer: torch.nn.Module) -> None:
    for name in ("h0", "W_prime", "hidden_sampling"):
        if not hasattr(layer, name):
            raise e.TypeError(f"`layer={type(layer).__name__}` has no `{name}`, so it is not an RTRBM-family layer.")

    if hasattr(layer, "c0") or hasattr(layer, "dynamic_hidden_bias"):
        raise e.TypeError(
            f"`layer={type(layer).__name__}` carries extra recurrent state or a history window, "
            "which streaming does not support."
        )

    if getattr(layer, "input_normalize", False):
        raise e.ValueError(
            f"`layer={type(layer).__name__}` normalizes each batch with its own statistics, "
            "so frame-by-frame inference would not match `forward`."
        )


class StreamingEncoder:
    """Carries the recurrent hidden state of an RTRBM-family model or RTDBN across single frames."""

    def __init__(self, model: torch.nn.Module, track_energy: bool = False) -> None:
        """Wrap a single layer or an RTDBN-like object exposing `models`.

        Args:
            model: Trained RTRBM-family layer, or an object whose `models` attribute lists such layers.
            track_energy: Whether `step` also records the first layer's free energy in `last_energy`.

        """

        layers = list(model.models) if hasattr(model, "models") else [model]

        if len(layers) == 0:
            raise e.SizeError("`model.models` should hold at least one layer.")

        for layer in layers:
            _check_layer(layer)

        self.layers = layers
        self.track_energy = track_energy
        self.last_energy = None
        self.n_steps = 0
        self._h = None

    @property
    def n_visible(self) -> int:
        """Number of visible units expected per frame."""
        return self.layers[0].n_visible

    @property
    def n_hidden(self) -> int:
        """Number of hidden units in the output (last) layer."""
        return self.layers[-1].n_hidden

    @property
    def batch_size(self) -> int | None:
        """Number of parallel streams currently held, or None before the first `reset` or `step`."""
        return None if self._h is None else self._h[0].shape[0]

    def reset(self, batch_size: int = 1) -> None:
        """Restart the stream(s) from each layer's learned initial hidden state.

        Args:
            batch_size: Number of parallel streams.

        """

        if batch_size < 1:
            raise e.ValueError(f"`batch_size` should be >= 1, but got {batch_size}.")

        with torch.no_grad():
            self._h = [layer.h0.detach().unsqueeze(0).repeat(batch_size, 1) for layer in self.layers]

        self.n_steps = 0
        self.last_energy = None

    def step(self, frame: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Consume one frame per stream and return the output layer's hidden probabilities.

        Each call applies the recurrence of `forward` (posterior probabilities, no sampling), so the output equals
        the matching slice of `forward` on the same stream since `reset`. Layers of an RTDBN are chained per frame.

        Args:
            frame: Tensor of shape (n_visible,) for a single stream or (batch, n_visible).

        Returns:
            Tensor of shape (n_hidden,) for a 1-D frame, otherwise (batch, n_hidden).

        """

        ref = self.layers[0].h0
        x = torch.as_tensor(frame, dtype=ref.dtype, device=ref.device)
        single = x.dim() == 1

        if single:
            x = x.unsqueeze(0)

        if x.dim() != 2 or x.shape[1] != self.n_visible:
            raise e.SizeError(f"`frame` should have {self.n_visible} features, but got shape {tuple(x.shape)}.")

        if self._h is None:
            self.reset(x.shape[0])

        if x.shape[0] != self._h[0].shape[0]:
            raise e.SizeError(f"`frame` has batch {x.shape[0]}, but the stream holds {self._h[0].shape[0]}.")

        with torch.no_grad():
            if self.track_energy:
                self.last_energy = self.layers[0].energy(x, self._h[0])

            for i, layer in enumerate(self.layers):
                probs, _ = layer.hidden_sampling(x, self._h[i])
                self._h[i] = probs
                x = probs

        self.n_steps += 1

        return x[0] if single else x

    def run(self, x: torch.Tensor | np.ndarray, return_energy: bool = False):
        """Reset, then stream whole sequences frame by frame.

        Args:
            x: Tensor of shape (batch, seq_len, n_visible).
            return_energy: Whether to also return the first layer's per-frame free energy.

        Returns:
            Hidden probabilities of shape (batch, seq_len, n_hidden), plus free energies of shape
            (batch, seq_len) when `return_energy` is True. Free energies require `track_energy`.

        """

        if return_energy and not self.track_energy:
            raise e.ValueError("`return_energy` is True, but `track_energy` is not.")

        x = torch.as_tensor(x, dtype=self.layers[0].h0.dtype, device=self.layers[0].h0.device)

        if x.dim() != 3:
            raise e.SizeError(f"`x` should have shape (batch, seq_len, n_visible), but got {tuple(x.shape)}.")

        self.reset(x.shape[0])
        out, energy = [], []

        for t in range(x.shape[1]):
            out.append(self.step(x[:, t, :]).unsqueeze(1))
            if return_energy:
                energy.append(self.last_energy.unsqueeze(1))

        hidden = torch.cat(out, dim=1)

        return (hidden, torch.cat(energy, dim=1)) if return_energy else hidden

    def max_abs_difference(self, x: torch.Tensor | np.ndarray) -> float:
        """Largest absolute gap between stepped output and the model's own batch `forward` on `x`.

        Args:
            x: Tensor of shape (batch, seq_len, n_visible).

        Returns:
            The maximum absolute difference over all elements.

        """

        x = torch.as_tensor(x, dtype=self.layers[0].h0.dtype, device=self.layers[0].h0.device)

        with torch.no_grad():
            ref = x
            for layer in self.layers:
                ref = layer.forward(ref)

            return float((self.run(x) - ref).abs().max())


class RidgeReadout:
    """Standardised ridge regression with an intercept, solved in closed form."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Create an unfitted readout.

        Args:
            alpha: Non-negative L2 penalty on the standardised features.

        """

        if alpha < 0:
            raise e.ValueError(f"`alpha` should be >= 0, but got {alpha}.")

        self.alpha = alpha
        self.mean = None
        self.scale = None
        self.coef = None
        self.intercept = None

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` or `load_state` has provided coefficients."""
        return self.coef is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        """Fit coefficients on features `x` of shape (n, d) and targets `y` of shape (n, k).

        Args:
            x: Feature matrix.
            y: Target matrix.

        Returns:
            The fitted readout.

        """

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise e.SizeError(f"`x` and `y` should be 2-D with equal rows >= 2, but got {x.shape} and {y.shape}.")

        self.mean = x.mean(0)
        scale = x.std(0)
        self.scale = np.where(scale > 0, scale, 1.0)

        z = (x - self.mean) / self.scale
        y_mean = y.mean(0)
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])

        self.coef = np.linalg.solve(gram, z.T @ (y - y_mean))
        self.intercept = y_mean

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict targets for features `x` of shape (n, d)."""
        if not self.is_fitted:
            raise e.BuildError("`readout` has not been fitted.")

        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale

        return z @ self.coef + self.intercept

    def state(self) -> dict:
        """Return the fitted parameters as a dictionary of arrays and floats."""
        if not self.is_fitted:
            raise e.BuildError("`readout` has not been fitted.")

        return {
            "alpha": self.alpha,
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
        }

    def load_state(self, state: dict) -> "RidgeReadout":
        """Restore parameters produced by `state`."""
        self.alpha = float(state["alpha"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.scale = np.asarray(state["scale"], dtype=np.float64)
        self.coef = np.asarray(state["coef"], dtype=np.float64)
        self.intercept = np.asarray(state["intercept"], dtype=np.float64)

        return self


def phase_to_unit_circle(phase_pct: np.ndarray) -> np.ndarray:
    """Map cycle positions in percent (0-100) to (sin, cos) targets of shape (..., 2)."""
    angle = 2.0 * np.pi * np.asarray(phase_pct, dtype=np.float64) / 100.0

    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


def unit_circle_to_phase(sincos: np.ndarray) -> np.ndarray:
    """Map (sin, cos) pairs of shape (..., 2) back to cycle positions in percent, in [0, 100)."""
    angle = np.arctan2(sincos[..., 0], sincos[..., 1])

    return np.mod(angle / (2.0 * np.pi) * 100.0, 100.0)


def circular_error(pred_pct: np.ndarray, true_pct: np.ndarray) -> np.ndarray:
    """Absolute circular distance in percent of the cycle (0 to 50) between two phase arrays."""
    d = np.abs(np.asarray(pred_pct, dtype=np.float64) - np.asarray(true_pct, dtype=np.float64)) % 100.0

    return np.minimum(d, 100.0 - d)


class StreamingPhaseEstimator:
    """Estimates the position in a periodic cycle (e.g. gait phase) from one frame at a time."""

    def __init__(self, encoder: StreamingEncoder, alpha: float = 1.0) -> None:
        """Pair a streaming encoder with a ridge readout to the unit circle.

        Args:
            encoder: Streaming wrapper around a trained model.
            alpha: Ridge penalty on the standardised hidden units.

        """

        self.encoder = encoder
        self.readout = RidgeReadout(alpha)

    def fit(self, windows: np.ndarray, phase_pct: np.ndarray, burn_in: int = 2, chunk: int = 2048):
        """Fit the readout from labelled windows, each streamed from the learned initial state.

        Frames before `burn_in` and frames with a NaN phase are left out of the fit.

        Args:
            windows: Array of shape (n, seq_len, n_visible), preprocessed as the encoder was trained.
            phase_pct: Cycle position in percent per frame, shape (n, seq_len), NaN where unknown.
            burn_in: Number of leading frames of every window excluded from the fit.
            chunk: Number of windows encoded at once.

        Returns:
            The estimator.

        """

        windows = np.asarray(windows)
        phase_pct = np.asarray(phase_pct, dtype=np.float64)

        if windows.ndim != 3 or phase_pct.shape != windows.shape[:2]:
            raise e.SizeError(f"`phase_pct` should match the first two dims of `windows`, but got {phase_pct.shape}.")

        if not 0 <= burn_in < windows.shape[1]:
            raise e.ValueError(f"`burn_in` should be in [0, {windows.shape[1]}), but got {burn_in}.")

        feats, targets = [], []

        for s in range(0, len(windows), chunk):
            hidden = self.encoder.run(windows[s : s + chunk]).cpu().numpy()[:, burn_in:, :]
            ph = phase_pct[s : s + chunk, burn_in:]
            ok = np.isfinite(ph)
            feats.append(hidden[ok])
            targets.append(phase_to_unit_circle(ph[ok]))

        self.readout.fit(np.concatenate(feats), np.concatenate(targets))

        return self

    def decode(self, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert hidden probabilities of shape (..., n_hidden) to phase in percent and confidence.

        Confidence is the length of the predicted (sin, cos) vector: near 1 when the readout is sure.

        """

        flat = np.asarray(hidden, dtype=np.float64).reshape(-1, hidden.shape[-1])
        pred = self.readout.predict(flat).reshape(*hidden.shape[:-1], 2)

        return unit_circle_to_phase(pred), np.linalg.norm(pred, axis=-1)

    def reset(self, batch_size: int = 1) -> None:
        """Restart the underlying stream(s)."""
        self.encoder.reset(batch_size)

    def step(self, frame: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Consume one frame per stream; return (phase in percent, confidence), each of shape (batch,)."""
        hidden = self.encoder.step(frame).cpu().numpy()

        if hidden.ndim == 1:
            hidden = hidden[None, :]

        return self.decode(hidden)

    def predict_sequence(self, x: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stream windows of shape (batch, seq_len, n_visible); return phase and confidence per frame."""
        return self.decode(self.encoder.run(x).cpu().numpy())
