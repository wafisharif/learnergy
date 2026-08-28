"""Recurrent Temporal RBM with learned per-feature variance (sigma)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.constants as c
from learnergy.utils import logging
from learnergy.models.temporal.rtrbm import RTRBM

logger = logging.get_logger(__name__)


class RTVarianceGaussianRBM(RTRBM):
    def __init__(
        self,
        n_visible: int = 128,
        n_hidden: int = 128,
        steps: int = 1,
        learning_rate: float = 0.001,
        momentum: float = 0.0,
        decay: float = 0.0,
        temperature: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        logger.info("Overriding class: RTRBM -> RTVarianceGaussianRBM.")

        super(RTVarianceGaussianRBM, self).__init__(
            n_visible,
            n_hidden,
            steps,
            learning_rate,
            momentum,
            decay,
            temperature,
            use_gpu,
        )

        self.sigma = nn.Parameter(torch.ones(n_visible))
        self.optimizer.add_param_group({"params": self.sigma})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def sigma(self) -> torch.nn.Parameter:
        """Per-feature learned standard deviation parameter."""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: torch.nn.Parameter) -> None:
        self._sigma = sigma

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_sq = torch.pow(self.sigma, 2) + c.EPSILON
        v_scaled = torch.div(v, sigma_sq)

        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v_scaled, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        states = torch.bernoulli(probs)

        return probs, states

    def visible_sampling(
        self, h: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = F.linear(h, self.W, self.a)

        if self.device == "cpu":
            sigma = self.sigma.unsqueeze(0).expand(activations.size(0), -1)
        else:
            sigma = self.sigma

        states = torch.normal(activations, torch.pow(sigma, 2))

        return states, activations

    def energy(
        self, samples: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        sigma_sq = torch.pow(self.sigma, 2) + c.EPSILON
        v_scaled = torch.div(samples, sigma_sq)

        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v_scaled, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)

        v = torch.sum(
            torch.div(torch.pow(samples - self.a, 2), 2 * sigma_sq), dim=1
        )

        energy = -v - h

        return energy

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gibbs sampling for one timestep with learned variance."""
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, h_prev)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            visible_states, visible_activations = self.visible_sampling(
                neg_hidden_states, True
            )
            neg_hidden_probs, neg_hidden_states = self.hidden_sampling(
                visible_activations, h_prev, True
            )

        return (
            pos_hidden_probs,
            pos_hidden_states,
            neg_hidden_probs,
            neg_hidden_states,
            visible_activations,
        )

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Trains on one subseries with learned variance."""
        batch_size, seq_len, n_visible = sequence.shape
        h_prev = self.h0.unsqueeze(0).expand(batch_size, -1)

        self.optimizer.zero_grad()

        total_cost = torch.tensor(0.0)
        total_mse = torch.tensor(0.0)

        for t in range(seq_len):
            v_t = sequence[:, t, :]
            _, _, _, _, visible_activations = self.gibbs_sampling(v_t, h_prev)
            visible_activations = visible_activations.detach()

            cost_t = torch.mean(self.energy(v_t, h_prev)) - torch.mean(
                self.energy(visible_activations, h_prev)
            )
            total_cost = total_cost + cost_t

            batch_mse = torch.div(
                torch.sum(torch.pow(v_t - visible_activations, 2)), batch_size
            ).detach()
            total_mse = total_mse + batch_mse

            h_prev, _ = self.hidden_sampling(v_t, h_prev)
            h_prev = torch.nan_to_num(h_prev, nan=0.5)
            h_prev = torch.clamp(h_prev, 0.0, 1.0)

        total_cost.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clamp sigma to prevent gradient-driven collapse toward 0.
        with torch.no_grad():
            self.sigma.data.clamp_(min=0.1, max=10.0)

        return total_mse

    def reconstruct(
        self, dataset: torch.utils.data.Dataset
    ) -> Tuple[float, torch.Tensor]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        logger.info("Reconstructing new samples ...")

        mse = torch.tensor(0.0, device=self.device)
        batch_size = len(dataset)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        visible_probs_all = []

        for samples, _ in tqdm(batches):
            if self.device == "cuda":
                samples = samples.cuda()

            batch_size_actual = samples.size(0)
            seq_len = samples.size(1)
            h_prev = self.h0.unsqueeze(0).expand(batch_size_actual, -1)

            recon_activations = []

            for t in range(seq_len):
                v_t = samples[:, t, :]
                pos_hidden_probs, pos_hidden_states = self.hidden_sampling(
                    v_t, h_prev
                )
                _, visible_activations = self.visible_sampling(
                    pos_hidden_states)
                recon_activations.append(visible_activations.unsqueeze(1))
                h_prev = pos_hidden_probs

            recon_seq = torch.cat(recon_activations, dim=1)

            batch_mse = torch.div(
                torch.sum(torch.pow(samples - recon_seq, 2)),
                batch_size_actual
            ).detach()
            mse += batch_mse
            visible_probs_all.append(recon_seq)

        mse /= len(batches)
        logger.info("MSE: %f", mse)

        return mse, torch.cat(visible_probs_all, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_visible = x.shape
        h_prev = self.h0.unsqueeze(0).expand(batch_size, -1)

        all_probs = []
        for t in range(seq_len):
            v_t = x[:, t, :]
            probs, _ = self.hidden_sampling(v_t, h_prev)
            all_probs.append(probs.unsqueeze(1))
            h_prev = probs

        return torch.cat(all_probs, dim=1)

    def sample(
        self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100
    ) -> torch.Tensor:
        """Generates sequences using the noisy visible draw (this class's tuple order is state-first)."""
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
                    v, _ = self.visible_sampling(h)
                    _, h = self.hidden_sampling(v, h_prev)
                v_t = v

                all_visible.append(v_t.unsqueeze(1))

                h_prev, _ = self.hidden_sampling(v_t, h_prev)

            return torch.cat(all_visible, dim=1)
