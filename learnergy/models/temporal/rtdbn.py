"""Recurrent Temporal Deep Belief Network: stacked RTRBM layers with mean-pooled temporal embeddings."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Dataset, Model
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

    def sample(
        self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100
    ) -> torch.Tensor:
        """Delegates to the single trained RTRBM layer; multi-layer sampling isn't implemented."""
        if self.n_layers != 1:
            raise NotImplementedError("Multi-layer RTDBN sampling not implemented.")
        return self.models[0].sample(
            n_samples=n_samples, n_steps=n_steps, gibbs_steps=gibbs_steps
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes sequences through all RTRBM layers, mean-pooled over time."""
        h = x
        for model in self.models:
            h = model.forward(h)  # (batch, seq_len, n_hidden_i)

        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning temporal embeddings."""
        return self.encode(x)

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: Tuple[int, ...] = (30,),
        warmup_epochs: Tuple[int, ...] = (15,),
    ) -> List[torch.Tensor]:
        """Trains each RTRBM layer via greedy layer-wise pre-training: each
        layer trains to convergence, freezes, then its hidden output becomes
        the next layer's training data.
        """
        if len(epochs) != self.n_layers:
            raise e.SizeError(
                f"`epochs` should have size equal to {self.n_layers}"
            )

        mse_per_layer = []
        current_dataset = dataset

        for i, model in enumerate(self.models):
            logger.info("Fitting RTDBN layer %d/%d ...", i + 1, self.n_layers)

            has_sigma = hasattr(model, "sigma")
            warmup = (warmup_epochs[i] if i < len(warmup_epochs) else 0) if has_sigma else 0
            full = epochs[i] - warmup

            if warmup > 0:
                model.sigma.requires_grad_(False)
                model.fit(current_dataset, batch_size=batch_size, epochs=warmup)
                model.sigma.requires_grad_(True)
            elif has_sigma:
                model.sigma.requires_grad_(True)

            model.fit(current_dataset, batch_size=batch_size, epochs=full)

            mse_per_layer.append(model.history["mse"][-1])

            if i < self.n_layers - 1:
                for param in model.parameters():
                    param.requires_grad_(False)
                current_dataset = self._encode_dataset(current_dataset, model, batch_size)

        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(True)

        return mse_per_layer

    def _encode_dataset(
        self, dataset: torch.utils.data.Dataset, model: torch.nn.Module, batch_size: int
    ) -> Dataset:
        """Encodes a dataset once through a frozen layer to build the next
        layer's training data; torch.no_grad() avoids an autograd graph
        through the frozen params.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()
        all_encoded = []
        all_targets = []
        with torch.no_grad():
            for samples, targets in loader:
                if self.device == "cuda":
                    samples = samples.cuda()
                encoded = model.forward(samples)  # (batch, seq_len, n_hidden_i)
                all_encoded.append(encoded.cpu())
                all_targets.append(targets)
        model.train()

        encoded_data = torch.cat(all_encoded, dim=0)
        targets_data = torch.cat(all_targets, dim=0)

        return Dataset(encoded_data, targets_data, None, show_log=False)
