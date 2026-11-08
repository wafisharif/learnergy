# Copyright (c) 2020-2026 Mateus Roder and Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Free-energy novelty scoring with a calibrated alarm threshold, for windows and for live streams."""

import numpy as np
import torch

import learnergy.utils.exception as e
from learnergy.models.temporal.streaming import StreamingEncoder
from learnergy.utils import logging

logger = logging.get_logger(__name__)


def _first_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.models[0] if hasattr(model, "models") else model


def window_free_energy(model: torch.nn.Module, windows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Score windows by their mean free energy under the first layer, restarting from `h0` at each window.

    Free energy is `energy(v_t, h_{t-1})` with the recurrent state propagated by posterior probabilities,
    averaged over the window. Higher means the window is less probable under the model, i.e. more novel.
    For an RTDBN only the first layer is used, since only it models the raw visible data.

    Args:
        model: Trained RTRBM-family layer, or an RTDBN-like object exposing `models`.
        windows: Array of shape (n, seq_len, n_visible), preprocessed as the model was trained.
        batch_size: Number of windows scored at once.

    Returns:
        Array of shape (n,) with one score per window.

    """

    windows = np.asarray(windows)

    if windows.ndim != 3 or len(windows) == 0:
        raise e.SizeError(f"`windows` should have shape (n > 0, seq_len, n_visible), but got {windows.shape}.")

    if batch_size < 1:
        raise e.ValueError(f"`batch_size` should be >= 1, but got {batch_size}.")

    encoder = StreamingEncoder(_first_layer(model), track_energy=True)
    scores = []

    for s in range(0, len(windows), batch_size):
        _, energy = encoder.run(windows[s : s + batch_size], return_energy=True)
        scores.append(energy.mean(dim=1).cpu().numpy())

    return np.concatenate(scores)


class NoveltyMonitor:
    """Flags windows whose free energy exceeds a threshold calibrated on familiar reference windows."""

    def __init__(self, model: torch.nn.Module, quantile: float = 0.95, batch_size: int = 2048) -> None:
        """Create an uncalibrated monitor.

        Args:
            model: Trained RTRBM-family layer, or an RTDBN-like object exposing `models`.
            quantile: Fraction of reference windows that should fall at or below the threshold.
            batch_size: Number of windows scored at once.

        """

        if not 0.0 < quantile < 1.0:
            raise e.ValueError(f"`quantile` should be in (0, 1), but got {quantile}.")

        self.model = model
        self.quantile = quantile
        self.batch_size = batch_size
        self.threshold = None
        self.n_reference = 0

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Return one free-energy score per window of shape (n, seq_len, n_visible)."""
        return window_free_energy(self.model, windows, self.batch_size)

    def calibrate(self, reference_windows: np.ndarray) -> float:
        """Set the threshold to the configured quantile of the reference scores.

        Calibrate on windows the model did not train on: scores of training windows are optimistically low, which
        would make the alarm fire too often on new data. On gait data a threshold from other subjects did not
        transfer (a nominal 5% false-alarm rate gave 10% to 25% on unseen subjects, and 0% to 100% per subject),
        whereas the subject's own familiar windows gave 5%, so calibrate on the person or system being monitored.

        Args:
            reference_windows: Array of shape (n, seq_len, n_visible) of familiar, non-training windows.

        Returns:
            The threshold.

        """

        scores = self.score(reference_windows)
        self.threshold = float(np.quantile(scores, self.quantile))
        self.n_reference = len(scores)
        logger.info("Novelty threshold %f from %d reference windows.", self.threshold, self.n_reference)

        return self.threshold

    def flag(self, windows: np.ndarray) -> np.ndarray:
        """Return a boolean array that is True where a window scores above the calibrated threshold."""
        if self.threshold is None:
            raise e.BuildError("`threshold` is None, so `calibrate` has to run first.")

        return self.score(windows) > self.threshold

    def state(self) -> dict:
        """Return the calibration as a dictionary that `load_state` accepts."""
        return {"quantile": self.quantile, "threshold": self.threshold, "n_reference": self.n_reference}

    def load_state(self, state: dict) -> "NoveltyMonitor":
        """Restore a calibration produced by `state`."""
        self.quantile = float(state["quantile"])
        self.threshold = None if state["threshold"] is None else float(state["threshold"])
        self.n_reference = int(state["n_reference"])

        return self


class StreamingNoveltyScorer:
    """Moving-average free energy over the last `window` frames of a live stream, with an optional alarm."""

    def __init__(
        self,
        model: torch.nn.Module,
        window: int = 30,
        threshold: float | None = None,
        reset_every: int | None = None,
    ) -> None:
        """Create a scorer around the first layer of `model`.

        Args:
            model: Trained RTRBM-family layer, or an RTDBN-like object exposing `models`.
            window: Number of recent frames averaged into the score.
            threshold: Alarm level on the moving average, for example from `NoveltyMonitor.calibrate`.
            reset_every: If set, restart the recurrent state every this many frames, which reproduces
                `window_free_energy` on consecutive blocks when it equals `window`.

        """

        if window < 1:
            raise e.ValueError(f"`window` should be >= 1, but got {window}.")

        if reset_every is not None and reset_every < 1:
            raise e.ValueError(f"`reset_every` should be >= 1, but got {reset_every}.")

        self.encoder = StreamingEncoder(_first_layer(model), track_energy=True)
        self.window = window
        self.threshold = threshold
        self.reset_every = reset_every
        self.n_frames = 0
        self._buffer = None

    def reset(self, batch_size: int = 1) -> None:
        """Restart the stream(s) and clear the score history."""
        self.encoder.reset(batch_size)
        ref = self.encoder.layers[0].h0
        self._buffer = torch.zeros(batch_size, self.window, dtype=ref.dtype, device=ref.device)
        self.n_frames = 0

    def step(self, frame: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Consume one frame per stream; return (moving-average score, alarm), each of shape (batch,).

        The alarm is False until `window` frames have been seen and whenever `threshold` is None.

        """

        x = torch.as_tensor(frame, dtype=self.encoder.layers[0].h0.dtype)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if self._buffer is None or x.shape[0] != self._buffer.shape[0]:
            self.reset(x.shape[0])

        if self.reset_every is not None and self.n_frames > 0 and self.n_frames % self.reset_every == 0:
            self.encoder.reset(x.shape[0])

        self.encoder.step(x)
        self._buffer[:, self.n_frames % self.window] = self.encoder.last_energy
        self.n_frames += 1

        n_seen = min(self.n_frames, self.window)
        score = (self._buffer.sum(dim=1) / n_seen).cpu().numpy()
        alarm = np.zeros(len(score), dtype=bool)

        if self.threshold is not None and self.n_frames >= self.window:
            alarm = score > self.threshold

        return score, alarm
