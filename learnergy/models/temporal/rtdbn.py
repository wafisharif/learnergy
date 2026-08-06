"""Recurrent Temporal Deep Belief Network (RTDBN).

- Each layer is an RTRBM (specifically RTVarianceGaussianRBM by default)
- forward() returns temporal embeddings via mean pooling over the time
  axis, collapsing (batch, seq_len, n_hidden) -> (batch, n_hidden)
- fit() trains each layer on sequences

Clustering head and training wrapper live in SIT-FUSE:
  sit_fuse.models.encoders.rtdbn_pl (encoder wrapper)
  sit_fuse.models.deep_cluster.rtdbn_dc (clustering head + IIC loss)
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Model
from learnergy.utils import logging

from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM

logger = logging.get_logger(__name__)


RT_MODELS = {
    "variance_gaussian": RTVarianceGaussianRBM,
}


class RTDBN(Model):

    def __init__(
        self,
        model: Tuple[str, ...] = ("variance_gaussian",),
        n_visible: int = 78,
        n_hidden: Tuple[int, ...] = (64,),
        steps: Tuple[int, ...] = (1,),
        learning_rate: Tuple[float, ...] = (0.001,),
        momentum: Tuple[float, ...] = (0.0,),
        decay: Tuple[float, ...] = (0.0,),
        temperature: Tuple[float, ...] = (1.0,),
        use_gpu: bool = False,
    ) -> None:
        logger.info("Overriding class: Model -> RTDBN.")

        super(RTDBN, self).__init__(use_gpu=use_gpu)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_layers = len(n_hidden)

        self.steps = steps
        self.lr = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.T = temperature

        if not isinstance(model, tuple):
            model = (model,)

        self.models = nn.ModuleList([])
        for i in range(self.n_layers):
            n_input = self.n_visible if i == 0 else self.n_hidden[i - 1]

            if model[i] not in RT_MODELS:
                raise e.ValueError(
                    f"Model '{model[i]}' not supported. "
                    f"Choose from: {list(RT_MODELS.keys())}"
                )

            m = RT_MODELS[model[i]](
                n_visible=n_input,
                n_hidden=self.n_hidden[i],
                steps=self.steps[i],
                learning_rate=self.lr[i],
                momentum=self.momentum[i],
                decay=self.decay[i],
                temperature=self.T[i],
                use_gpu=use_gpu,
            )
            self.models.append(m)

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")
        logger.debug("Number of layers: %d.", self.n_layers)

    @property
    def n_visible(self) -> int:
        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_visible: int) -> None:
        if n_visible <= 0:
            raise e.ValueError("`n_visible` should be > 0")
        self._n_visible = n_visible

    @property
    def n_hidden(self) -> Tuple[int, ...]:
        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden: Tuple[int, ...]) -> None:
        self._n_hidden = n_hidden

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise e.ValueError("`n_layers` should be > 0")
        self._n_layers = n_layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes sequences through all RTRBM layers and returns
        temporal embeddings via mean pooling over the time axis.

        Args:
            x: Input sequences, shape (batch, seq_len, n_visible).

        Returns:
            Temporal embeddings, shape (batch, n_hidden[-1]).
        """
        h = x
        for model in self.models:
            h = model.forward(h)  # (batch, seq_len, n_hidden_i)

        # Mean pool over time: (batch, seq_len, n_hidden) -> (batch, n_hidden)
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning temporal embeddings.

        Args:
            x: Input sequences, shape (batch, seq_len, n_visible).

        Returns:
            Temporal embeddings, shape (batch, n_hidden[-1]).
        """
        return self.encode(x)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: Tuple[int, ...] = (30,),
        warmup_epochs: Tuple[int, ...] = (15,),
    ) -> List[torch.Tensor]:
        """Trains each RTRBM layer sequentially.

        Args:
            dataset: Dataset where each sample is (seq_len, n_visible).
            batch_size: Batch size.
            epochs: Training epochs per layer.
            warmup_epochs: Sigma warmup epochs per layer.

        Returns:
            List of final MSE per layer.
        """
        if len(epochs) != self.n_layers:
            raise e.SizeError(
                f"`epochs` should have size equal to {self.n_layers}"
            )

        mse_per_layer = []

        for i, model in enumerate(self.models):
            logger.info("Fitting RTDBN layer %d/%d ...", i + 1, self.n_layers)

            if i == 0:
                warmup = warmup_epochs[i] if i < len(warmup_epochs) else 0
                full = epochs[i] - warmup

                if warmup > 0:
                    model.sigma.requires_grad_(False)
                    model.fit(dataset, batch_size=batch_size, epochs=warmup)

                model.sigma.requires_grad_(True)
                model.fit(dataset, batch_size=batch_size, epochs=full)

                mse_per_layer.append(model.history["mse"][-1])

            else:
                raise NotImplementedError(
                    "Multi-layer RTDBN training not yet implemented. "
                    "Use n_hidden=(64,) for the single-layer configuration."
                )

        return mse_per_layer
