"""Gaussian-Bernoulli Recurrent Temporal Restricted Boltzmann Machine.

Extends RTRBM (Bernoulli visible) with Gaussian visible units, making it
suitable for continuous-valued data like biomechanical time series.

The recurrent mechanism (W_prime, h0, hidden_sampling, fit_subseries,
fit, forward, reconstruct, sample) is inherited unchanged from RTRBM --
recurrence only touches the hidden layer, not the visible one.

Only four things change vs. Bernoulli RTRBM:
  1. energy()           -- quadratic visible term (v - a)^2 instead of -v*a
  2. visible_sampling() -- linear activations, not Bernoulli samples
  3. normalize/input_normalize -- per-batch normalization flags
  4. fit_subseries()    -- applies per-batch normalization before training,
                          mirroring GaussianRBM.fit()'s normalize step
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnergy.utils import logging
from learnergy.models.temporal.rtrbm import RTRBM

logger = logging.get_logger(__name__)


class RTGaussianRBM(RTRBM):
    """Gaussian-Bernoulli Recurrent Temporal RBM.

    Extends RTRBM by replacing Bernoulli visible units with Gaussian
    visible units (variance fixed to 1, same as GaussianRBM in
    learnergy's gaussian_rbm.py). All recurrent machinery inherited
    from RTRBM unchanged.

    Use this instead of RTRBM for continuous-valued input data
    (e.g. biomechanical joint angles, velocities, muscle activations).
    """

    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
        normalize: bool = True,
        input_normalize: bool = True,
    ) -> None:
        """Initialization method.

        Args mirror GaussianRBM.__init__ (gaussian_rbm.py) with the
        addition of all RTRBM recurrent parameters (W_prime, h0) added
        by the parent RTRBM.__init__.

        Args:
            n_visible: Amount of visible units.
            n_hidden: Amount of hidden units.
            steps: Number of Gibbs' sampling steps.
            learning_rate: Learning rate.
            momentum: Momentum parameter.
            decay: Weight decay used for penalization.
            temperature: Temperature factor.
            use_gpu: Whether GPU should be used or not.
            normalize: Whether or not to use batch normalization during
                fit(). Mirrors GaussianRBM's normalize parameter.
            input_normalize: Whether or not to normalize inputs during
                forward(). Mirrors GaussianRBM's input_normalize parameter.
        """
        # Set normalize flags BEFORE calling super().__init__() --
        # matches the order GaussianRBM.__init__ uses in gaussian_rbm.py.
        self._normalize = normalize
        self._input_normalize = input_normalize

        logger.info("Overriding class: RTRBM -> RTGaussianRBM.")

        super(RTGaussianRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        logger.info("Class overrided.")

    @property
    def normalize(self) -> bool:
        """Whether or not to use batch normalization during fit()."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    @property
    def input_normalize(self) -> bool:
        """Whether or not to normalize inputs during forward()."""
        return self._input_normalize

    @input_normalize.setter
    def input_normalize(self, input_normalize: bool) -> None:
        self._input_normalize = input_normalize

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Calculates the system's energy for Gaussian visible units.

        Overrides RTRBM.energy() to use the Gaussian visible energy term:
            E = 0.5 * sum((v - a)^2) - sum(softplus(W^T v + W'h + b))

        The quadratic visible term 0.5*(v-a)^2 replaces the linear -v*a
        term used in the Bernoulli case, matching GaussianRBM.energy()
        in gaussian_rbm.py -- the only change for Gaussian visibles.
        The recurrent bias term (W_prime @ h_prev + b) is inherited from
        RTRBM.energy().

        Args:
            samples: Visible samples, shape (batch, n_visible).
            h_prev: Previous hidden probabilities, shape (batch, n_hidden).

        Returns:
            System energy per sample, shape (batch,).
        """
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(samples, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)

        # Gaussian visible term -- replaces Bernoulli's -v*a
        # Mirrors GaussianRBM.energy() in gaussian_rbm.py exactly
        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        energy = v - h

        return energy

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor
    ):
        """Overrides RTRBM.gibbs_sampling for Gaussian visible units.

        KEY FIX: For Gaussian visible units, uses the mean field value
        (linear activation) directly during the Gibbs loop instead of
        sampling noisy visible states. This is standard practice for
        Gaussian-Bernoulli RBMs -- see:

        Hinton & Salakhutdinov (2006): "rather than sampling from the
        distribution, the visible units can be set equal to their means"

        Using sampled values introduces noise that causes activation
        explosion and NaN during CD-k with continuous visible units,
        especially when combined with per-batch normalization.
        """
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, h_prev)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            # Use visible_probs (mean field) not visible_states (samples)
            # for Gaussian visible units -- prevents activation explosion
            visible_probs, visible_states = self.visible_sampling(
                neg_hidden_states, True
            )

            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_probs, h_prev, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_probs,
        )

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ):
        """Overrides RTRBM.hidden_sampling to clamp probabilities and
        guard against NaN values in h_prev.

        After per-batch normalization, large activations during Gibbs
        sampling can produce NaN hidden states that propagate through
        the recurrent chain. Clamping h_prev and probs prevents this.
        """
        # Guard against NaN in h_prev -- can occur during Gibbs sampling
        # after normalization makes activations large
        h_prev = torch.nan_to_num(h_prev, nan=0.0)
        h_prev = torch.clamp(h_prev, 0.0, 1.0)

        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        # Clamp to avoid numerical issues with torch.bernoulli
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        states = torch.bernoulli(probs)

        return probs, states

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Overrides RTRBM.fit_subseries to apply per-batch normalization
        before training, mirroring GaussianRBM.fit()'s normalize step
        (gaussian_rbm.py lines 184-188).

        Per-batch normalization (zero mean, unit std per feature) is
        essential for Gaussian visible units -- without it, the unbounded
        linear activations cause the model to collapse to outputting a
        constant value (the data mean) immediately, ignoring the data
        structure entirely. This is a known issue with Gaussian RBMs
        documented in Cho, Ilin & Raiko (2011).

        The normalization is applied per-subseries batch, not globally --
        same as GaussianRBM.fit() does per-batch. This is separate from
        the external MinMaxScaler preprocessing, which scales across the
        full dataset.

        Args:
            sequence: One subseries, shape (batch, seq_len, n_visible).

        Returns:
            MSE for this subseries.
        """
        if self.normalize:
            batch_size, seq_len, n_visible = sequence.shape
            flat = sequence.reshape(-1, n_visible)
            flat = (
                (flat - torch.mean(flat, 0, True))
                / (torch.std(flat, 0, True) + 1e-6)
            ).detach()
            sequence = flat.reshape(batch_size, seq_len, n_visible)

        # Run the full subseries training loop with gradient clipping.
        # We can't call super().fit_subseries() and add clipping after
        # since the optimizer.step() is inside that method. Instead we
        # replicate the loop here with clipping added -- matches the
        # same pattern but adds stability for Gaussian training.
        batch_size, seq_len, n_visible = sequence.shape
        h_prev = self.h0.unsqueeze(0).expand(batch_size, -1)
        self.optimizer.zero_grad()

        total_cost = torch.tensor(0.0)
        total_mse = torch.tensor(0.0)

        for t in range(seq_len):
            v_t = sequence[:, t, :]
            _, _, _, _, visible_states = self.gibbs_sampling(v_t, h_prev)
            visible_states = visible_states.detach()

            cost_t = torch.mean(self.energy(v_t, h_prev)) - torch.mean(
                self.energy(visible_states, h_prev)
            )
            total_cost = total_cost + cost_t

            batch_mse = torch.div(
                torch.sum(torch.pow(v_t - visible_states, 2)), batch_size
            ).detach()
            total_mse = total_mse + batch_mse

            h_prev, _ = self.hidden_sampling(v_t, h_prev)
            # Guard against NaN propagation through the recurrent chain
            h_prev = torch.nan_to_num(h_prev, nan=0.5)
            h_prev = torch.clamp(h_prev, 0.0, 1.0)

        total_cost.backward()

        # Gradient clipping -- prevents weight explosion when training
        # with per-batch normalized data, which can produce large gradients.
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        return total_mse

    def visible_sampling(
        self, h: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs visible layer sampling for Gaussian units, P(v|h).

        Overrides RBM.visible_sampling (inherited via RTRBM). For
        Gaussian visible units, the mean of P(v|h) is simply the linear
        activation W*h + a -- no sigmoid, no Bernoulli sampling. This
        gives continuous-valued outputs instead of binary ones, which is
        exactly what we need for biomechanical data.

        Mirrors GaussianRBM.visible_sampling() in gaussian_rbm.py exactly.

        Args:
            h: Hidden layer tensor, shape (batch, n_hidden).
            scale: Whether to divide by temperature T.

        Returns:
            (probs, states): both are the linear activation for Gaussian
            units (probs = sigmoid of states, states = linear activation).
        """
        activations = F.linear(h, self.W, self.a)

        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)

        return probs, states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overrides RTRBM.forward() to apply input normalization.

        Mirrors GaussianRBM.forward() (gaussian_rbm.py lines 285-288),
        adapted for temporal (batch, seq_len, n_visible) input shape.
        Without this, forward() sees un-normalized data but the model
        was trained on normalized data -- causing poor hidden embeddings.
        """
        if self.input_normalize:
            batch_size, seq_len, n_visible = x.shape
            flat = x.reshape(-1, n_visible)
            flat = (
                (flat - torch.mean(flat, 0, True))
                / (torch.std(flat, 0, True) + 1e-6)
            ).detach()
            x = flat.reshape(batch_size, seq_len, n_visible)

        batch_size, seq_len, n_visible = x.shape
        h_prev = self.h0.unsqueeze(0).expand(batch_size, -1)
        all_probs = []
        for t in range(seq_len):
            v_t = x[:, t, :]
            probs, _ = self.hidden_sampling(v_t, h_prev)
            all_probs.append(probs.unsqueeze(1))
            h_prev = probs
        return torch.cat(all_probs, dim=1)

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        """Overrides RTRBM.reconstruct() to normalize during reconstruction.

        Mirrors GaussianRBM.reconstruct() (gaussian_rbm.py lines 249-254).
        The model was trained on per-batch normalized data -- without
        applying the same normalization during reconstruction, the model
        sees a completely different data distribution and reconstructions
        are meaningless.
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        logger.info("Reconstructing new samples ...")

        mse = torch.tensor(0.0)
        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        visible_probs_all = []

        for samples, _ in tqdm(batches):
            if self.device == "cuda":
                samples = samples.cuda()

            if self.normalize:
                b, s, n = samples.shape
                flat = samples.reshape(-1, n)
                flat = (
                    (flat - torch.mean(flat, 0, True))
                    / (torch.std(flat, 0, True) + 1e-6)
                ).detach()
                samples = flat.reshape(b, s, n)

            batch_size_actual = samples.size(0)
            seq_len = samples.size(1)
            h_prev = self.h0.unsqueeze(0).expand(batch_size_actual, -1)
            recon_probs = []
            recon_states = []

            for t in range(seq_len):
                v_t = samples[:, t, :]
                pos_hidden_probs, pos_hidden_states = self.hidden_sampling(
                    v_t, h_prev
                )
                visible_prob, visible_state = self.visible_sampling(
                    pos_hidden_states
                )
                recon_probs.append(visible_prob.unsqueeze(1))
                recon_states.append(visible_state.unsqueeze(1))
                h_prev = pos_hidden_probs

            recon_probs_seq = torch.cat(recon_probs, dim=1)
            recon_states_seq = torch.cat(recon_states, dim=1)

            batch_mse = torch.div(
                torch.sum(torch.pow(samples - recon_states_seq, 2)),
                batch_size_actual
            ).detach()
            mse += batch_mse
            visible_probs_all.append(recon_probs_seq)

        mse /= len(batches)
        logger.info("MSE: %f", mse)
        return mse, torch.cat(visible_probs_all, dim=0)
