from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


class RTRBM(RBM):
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
    ) -> None:

        logger.info("Overriding class: RBM -> RTRBM.")

        super(RTRBM, self).__init__(
            n_visible, n_hidden, steps, learning_rate, momentum,
            decay, temperature, use_gpu,
        )

        # Recurrent hidden-to-hidden weights: W' in the paper.
        self.W_prime = nn.Parameter(torch.randn(n_hidden, n_hidden) * 0.01)

        # Learnable initial hidden state, used at t=0 (no h_{-1} exists).
        self.h0 = nn.Parameter(torch.zeros(n_hidden))

        self.optimizer.add_param_group({"params": [self.W_prime, self.h0]})

        if self.device == "cuda":
            self.cuda()

        logger.info("Class overrided.")

    @property
    def W_prime(self) -> torch.nn.Parameter:
        """Recurrent hidden-to-hidden weights matrix."""
        return self._W_prime

    @W_prime.setter
    def W_prime(self, W_prime: torch.nn.Parameter) -> None:
        self._W_prime = W_prime

    @property
    def h0(self) -> torch.nn.Parameter:
        """Learnable initial hidden state (used at the first timestep)."""
        return self._h0

    @h0.setter
    def h0(self, h0: torch.nn.Parameter) -> None:
        self._h0 = h0

    def hidden_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor, scale: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(v, self.W.t()) + recurrent_bias

        if scale:
            probs = torch.sigmoid(torch.div(activations, self.T))
        else:
            probs = torch.sigmoid(activations)

        states = torch.bernoulli(probs)

        return probs, states

    def energy(self, samples: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        recurrent_bias = F.linear(h_prev, self.W_prime, self.b)
        activations = F.linear(samples, self.W.t()) + recurrent_bias

        s = nn.Softplus()
        h = torch.sum(s(activations), dim=1)
        v = torch.mv(samples, self.a)

        energy = -v - h

        return energy

    def gibbs_sampling(
        self, v: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs one timestep of Gibbs sampling."""
        pos_hidden_probs, pos_hidden_states = self.hidden_sampling(v, h_prev)
        neg_hidden_states = pos_hidden_states

        for _ in range(self.steps):
            _, visible_states = self.visible_sampling(neg_hidden_states, True)
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

    def cd_step(self, v: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        _, _, _, _, visible_states = self.gibbs_sampling(v, h_prev)
        visible_states = visible_states.detach()

        cost = torch.mean(self.energy(v, h_prev)) - torch.mean(
            self.energy(visible_states, h_prev)
        )

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        batch_size = v.size(0)
        mse = torch.div(
            torch.sum(torch.pow(v - visible_states, 2)), batch_size
        ).detach()

        return mse

    def fit_subseries(self, sequence: torch.Tensor) -> torch.Tensor:
        """Trains on one subseries via BPTT (single backward pass across timesteps)."""
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

        total_cost.backward()
        self.optimizer.step()

        return total_mse

    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
        epochs: int = 10,
    ) -> torch.Tensor:
        import time
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        mse = torch.tensor(0.0, device=self.device)

        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            start = time.time()
            mse = torch.tensor(0.0, device=self.device)

            for samples, _ in tqdm(batches):
                # samples: (batch, seq_len, n_visible)
                if self.device == "cuda":
                    samples = samples.cuda()

                batch_mse = self.fit_subseries(samples)
                mse += batch_mse

            mse /= len(batches)

            end = time.time()

            self.dump(mse=mse.item(), time=end - start)

            logger.info("MSE: %f", mse)

        return mse

    def sample(
        self, n_samples: int = 1, n_steps: int = 10, gibbs_steps: int = 100
    ) -> torch.Tensor:
        """Generates sequences via per-timestep Gibbs burn-in (Algorithm 3, Sutskever et al. 2008)."""
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
                    _, v = self.visible_sampling(h)
                    _, h = self.hidden_sampling(v, h_prev)
                v_t = v

                all_visible.append(v_t.unsqueeze(1))

                h_prev, _ = self.hidden_sampling(v_t, h_prev)

            return torch.cat(all_visible, dim=1)
