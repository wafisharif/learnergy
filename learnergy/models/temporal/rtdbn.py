"""Recurrent Temporal Deep Belief Network (RTDBN).

- Each layer is an RTRBM (specifically RTVarianceGaussianRBM by default)
- forward() returns temporal embeddings via mean pooling over the time
  axis, collapsing (batch, seq_len, n_hidden) -> (batch, n_hidden)
- fit() trains each layer on sequences rather than flat vectors
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


# Registry of supported RTRBM types -- mirrors DBN's MODELS dict (dbn.py)
# so additional RTRBM variants can be added later without changing RTDBN.
RT_MODELS = {
    "variance_gaussian": RTVarianceGaussianRBM,
}


class IICClusteringHead(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_clusters: int,
        n_hidden: int = 256,
        noise_std: float = 0.1,
    ) -> None:
        super(IICClusteringHead, self).__init__()

        self.noise_std = noise_std

        # Two fully connected layers -- same pattern as SIT-FUSE's
        # clustering head architecture from the paper
        self.fc = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_clusters),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std

    @staticmethod
    def iic_loss(p: torch.Tensor, p_perturbed: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Computes the IIC loss (negative mutual information).
        """
        # Joint distribution P(c, c') -- outer product averaged over batch
        # Shape: (n_clusters, n_clusters)
        p_joint = torch.einsum("bi,bj->ij", p, p_perturbed) / p.shape[0]
        p_joint = (p_joint + p_joint.t()) / 2  # symmetrize
        p_joint = torch.clamp(p_joint, min=eps)

        # Marginal distributions
        p_i = p_joint.sum(dim=1, keepdim=True)  # (n_clusters, 1)
        p_j = p_joint.sum(dim=0, keepdim=True)  # (1, n_clusters)

        # Mutual information (negative, since we minimize loss)
        mi = (p_joint * (torch.log(p_joint) -
              torch.log(p_i) - torch.log(p_j))).sum()

        return -mi


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
        n_clusters: int = 10,
        cluster_hidden: int = 256,
        noise_std: float = 0.1,
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

        # Build RTRBM layers -- mirrors DBN's nn.ModuleList pattern
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

        # IIC clustering head -- takes the temporal embedding from the
        # last RTRBM layer and produces cluster assignments
        self.clustering_head = IICClusteringHead(
            n_input=self.n_hidden[-1],
            n_clusters=n_clusters,
            n_hidden=cluster_hidden,
            noise_std=noise_std,
        )

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
        # Pass through each RTRBM layer sequentially --
        # mirrors DBN.forward()'s layer-by-layer pattern (dbn.py line 404)
        # but adapted for temporal (batch, seq_len, n_features) shape
        h = x
        for model in self.models:
            h = model.forward(h)  # (batch, seq_len, n_hidden_i)

        # Mean pool over time axis: (batch, seq_len, n_hidden) -> (batch, n_hidden)
        # Standard approach for collapsing sequence embeddings to fixed-size vectors
        embedding = h.mean(dim=1)

        return embedding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x)
        cluster_probs = self.clustering_head(embeddings)
        return embeddings, cluster_probs

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: Tuple[int, ...] = (30,),
        warmup_epochs: Tuple[int, ...] = (15,),
    ) -> List[torch.Tensor]:
        if len(epochs) != self.n_layers:
            raise e.SizeError(
                f"`epochs` should have size equal to {self.n_layers}"
            )

        mse_per_layer = []

        for i, model in enumerate(self.models):
            logger.info("Fitting RTDBN layer %d/%d ...", i + 1, self.n_layers)

            if i == 0:
                # First layer trains on raw sequences
                warmup = warmup_epochs[i] if i < len(warmup_epochs) else 0
                full = epochs[i] - warmup

                if warmup > 0:
                    model.sigma.requires_grad_(False)
                    model.fit(dataset, batch_size=batch_size, epochs=warmup)

                model.sigma.requires_grad_(True)
                model.fit(dataset, batch_size=batch_size, epochs=full)

                mse_per_layer.append(model.history["mse"][-1])

            else:
                # Subsequent layers train on hidden output of previous layers.
                # We need to transform the dataset by passing it through
                # all previous layers first -- same principle as DBN.fit()
                # lines 325-326 which pass samples through previous models.
                # TODO: implement multi-layer training when n_layers > 1.
                # For now, Nick said start with one layer -- this path won't
                # be hit with the default single-layer config.
                raise NotImplementedError(
                    "Multi-layer RTDBN training not yet implemented. "
                    "(n_hidden=(64,)) -- this error should not appear "
                    "in the single-layer configuration."
                )

        return mse_per_layer

    def fit_clustering_head(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.001,
    ) -> List[float]:
        """Trains the IIC clustering head on top of the frozen RTRBM encoder.

        Mirrors SIT-FUSE: encoder pre-trained first, then frozen,
        then clustering head trained with IIC loss. Perturbations are Gaussian
        noise added to encoder outputs.
        """
        # Freeze the RTRBM encoder -- only train the clustering head
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(False)

        optimizer = torch.optim.Adam(
            self.clustering_head.parameters(), lr=learning_rate
        )

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for samples, _ in tqdm(batches, desc=f"IIC epoch {epoch+1}/{epochs}"):
                if self.device == "cuda":
                    samples = samples.cuda()

                # Compute embeddings FRESH per batch -- NOT cached.
                # Encoder is frozen so no grad needed through it,
                # but embeddings must be computed here (not pre-cached)
                # so each batch draws from the actual data distribution.
                with torch.no_grad():
                    embeddings = self.encode(samples)

                # Scale noise relative to embedding magnitude --
                # fixed noise_std can be too large or too small depending
                # on the encoder's output scale. Using 10% of per-batch
                # std keeps perturbations meaningful without destroying signal.
                emb_std = embeddings.std().item()
                adaptive_noise = max(emb_std * 0.1, 1e-4)

                # Perturb embeddings (adaptive Gaussian noise)
                perturbed = embeddings + \
                    torch.randn_like(embeddings) * adaptive_noise

                # Forward pass through clustering head
                # embeddings.detach() so gradients only flow through
                # the clustering head, not back to the frozen encoder
                p = self.clustering_head(embeddings.detach())
                p_perturbed = self.clustering_head(perturbed.detach())

                # IIC loss
                loss = IICClusteringHead.iic_loss(p, p_perturbed)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            self.dump(iic_loss=avg_loss)

            logger.info("IIC Epoch %d/%d | Loss: %.4f",
                        epoch + 1, epochs, avg_loss)

        # Unfreeze encoder after clustering head training
        for model in self.models:
            for param in model.parameters():
                param.requires_grad_(True)

        return loss_history

    def fit_clustering_kmeans(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        n_init: int = 10,
    ) -> torch.Tensor:
        """Clusters temporal embeddings using k-means.

        Practical alternative to IIC for initial end-to-end analysis.
        IIC requires training the encoder jointly with the clustering head
        to learn discriminative features -- training a clustering head on
        frozen RTRBM embeddings alone leads to collapse when the embeddings
        lack sufficient discriminability.
        """
        from sklearn.cluster import KMeans

        # Extract all embeddings
        embeddings, _ = self.get_cluster_assignments(dataset, batch_size)
        emb_np = embeddings.numpy()

        # Fit k-means
        km = KMeans(
            n_clusters=self.clustering_head.fc[-2].out_features,
            n_init=n_init,
            random_state=42,
        )
        assignments = km.fit_predict(emb_np)

        return torch.from_numpy(assignments)

    def get_cluster_assignments(
        self, dataset: torch.utils.data.Dataset, batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns cluster assignments and embeddings for a full dataset.
        """
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_embeddings = []
        all_assignments = []

        with torch.no_grad():
            for samples, _ in tqdm(batches):
                if self.device == "cuda":
                    samples = samples.cuda()

                embeddings, cluster_probs = self.forward(samples)
                assignments = torch.argmax(cluster_probs, dim=1)

                all_embeddings.append(embeddings)
                all_assignments.append(assignments)

        return (
            torch.cat(all_embeddings, dim=0),
            torch.cat(all_assignments, dim=0),
        )
