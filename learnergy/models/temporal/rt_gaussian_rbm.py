"""Gaussian-Bernoulli Recurrent Temporal RBM (continuous visible units)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from learnergy.utils import logging
from learnergy.models.temporal.rtrbm import RTRBM

logger = logging.get_logger(__name__)


class RTGaussianRBM(RTRBM):
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
        """Gaussian visible energy: 0.5*sum((v-a)^2) - sum(softplus(W^Tv + W'h + b))."""
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(samples, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)

        v = 0.5 * torch.sum((samples - self.a) ** 2, dim=1)

        energy = v - h

        return energy

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor
    ):
        """One Gibbs step; uses raw visible_states (not sigmoid) as the CD negative particle."""
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, h_prev)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            visible_probs, visible_states = self.visible_sampling(
                neg_hidden_states, True
            )

            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_states, h_prev, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_states,
        )

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ):
        h_prev = torch.nan_to_num(h_prev, nan=0.0)
        h_prev = torch.clamp(h_prev, 0.0, 1.0)

        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        states = torch.bernoulli(probs)

        return probs, states

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:

        if self.normalize:
            batch_size, seq_len, n_visible = sequence.shape
            flat = sequence.reshape(-1, n_visible)
            flat = (
                (flat - torch.mean(flat, 0, True))
                / (torch.std(flat, 0, True) + 1e-6)
            ).detach()
            sequence = flat.reshape(batch_size, seq_len, n_visible)

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
            h_prev = torch.nan_to_num(h_prev, nan=0.5)
            h_prev = torch.clamp(h_prev, 0.0, 1.0)

        total_cost.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        return total_mse

    def visible_sampling(
        self, h: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Visible layer sampling for Gaussian units, P(v|h)."""
        activations = F.linear(h, self.W, self.a)

        if scale:
            states = torch.div(activations, self.T)
        else:
            states = activations

        probs = torch.sigmoid(states)

        return probs, states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies input normalization, then runs RTRBM.forward()."""
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
        """Reconstructs a dataset, normalizing per batch first."""
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

    def sample(
        self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100
    ) -> torch.Tensor:
        """Generates sequences; adds Gaussian noise to the mean since visible_sampling() doesn't."""
        with torch.no_grad():
            h_prev = self.h0.unsqueeze(0).expand(n_samples, -1)

            all_visible = []

            for t in range(n_steps):
                h = torch.bernoulli(
                    torch.full(
                        (n_samples, self.n_hidden), 0.5, device=h_prev.device
                    )
                )
                for _ in range(gibbs_steps):
                    _, mean = self.visible_sampling(h)
                    v = mean + torch.randn_like(mean)
                    _, h = self.hidden_sampling(v, h_prev)
                v_t = v

                all_visible.append(v_t.unsqueeze(1))

                h_prev, _ = self.hidden_sampling(v_t, h_prev)

            return torch.cat(all_visible, dim=1)
