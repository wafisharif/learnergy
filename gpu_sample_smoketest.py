"""
Manual GPU smoke test for multi-layer RTDBN.sample() (commit 34df8b1).

Not part of the pytest suite on purpose (no "test_" prefix, so pytest won't
auto-collect it) -- this needs a real CUDA device and does real training, so
it's meant to be run once by hand: `python gpu_sample_smoketest.py`.

Why this exists: the CPU-only validation that shipped with 34df8b1 flagged
one specific unverified risk -- RTVarianceGaussianRBM.visible_sampling()
handles the `sigma` broadcast differently by device:

    if self.device == "cpu":
        sigma = self.sigma.unsqueeze(0).expand(activations.size(0), -1)
    else:
        sigma = self.sigma

On CPU, sigma is expanded to match the batch dim before torch.normal(mean, std).
On GPU, it's left as shape (n_visible,) unexpanded. torch.normal(mean, std)
needs mean/std to either match exactly or be broadcastable, so this script
checks whether that difference actually breaks anything in practice on real
hardware, since it couldn't be checked in the sandbox (no GPU there).

Delete this file (or leave it untracked) once you've run it -- it's a
one-off diagnostic, not meant to be a permanent part of the repo.
"""
import sys

import numpy as np
import torch

from learnergy.core import Dataset as LDataset
from learnergy.models.temporal.rtdbn import RTDBN

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        import traceback
        traceback.print_exc()
        FAILURES.append(name)


def make_dataset(n_samples, seq_len, n_visible, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_samples, seq_len, n_visible)).astype(np.float32)
    targets = np.stack([np.arange(n_samples), np.zeros(n_samples)], axis=1).astype(np.int64)
    return LDataset(torch.from_numpy(data), torch.from_numpy(targets), None, show_log=False)


def main():
    if not torch.cuda.is_available():
        print("No CUDA device visible to torch -- nothing to test. Exiting.")
        sys.exit(1)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    seq_len, n_visible, n_samples = 6, 5, 24
    holder = {}

    def _train_2layer_gpu():
        ds = make_dataset(n_samples, seq_len, n_visible, seed=1)
        rtdbn = RTDBN(
            model=("variance_gaussian", "variance_gaussian"),
            n_visible=n_visible,
            n_hidden=(8, 4),
            steps=(1, 1),
            learning_rate=(0.001, 0.001),
            momentum=(0.0, 0.0),
            decay=(0.0, 0.0),
            temperature=(1.0, 1.0),
            use_gpu=True,
        )
        assert rtdbn.device == "cuda"
        rtdbn.fit(ds, batch_size=8, epochs=(2, 2), warmup_epochs=(1, 1))
        holder["model"] = rtdbn

    check("2-layer RTDBN trains on GPU (use_gpu=True)", _train_2layer_gpu)

    def _sample_on_gpu():
        rtdbn = holder["model"]
        out = rtdbn.sample(n_samples=3, n_steps=4, gibbs_steps=10)
        assert out.is_cuda, "sample() output is not on GPU"
        assert out.shape == (3, 4, n_visible), out.shape
        assert torch.isfinite(out).all(), "non-finite values in GPU-generated sequence"

    check("multi-layer sample() runs on GPU, correct shape, finite (tests the sigma-broadcast risk directly)", _sample_on_gpu)

    def _sample_statistical_sanity_gpu():
        rtdbn = holder["model"]
        out = rtdbn.sample(n_samples=8, n_steps=6, gibbs_steps=30)
        std = out.std().item()
        assert std > 1e-4, f"degenerate output on GPU, std={std}"
        assert std < 1e4, f"exploding output on GPU, std={std}"
        print(f"    (GPU generated std={std:.4f}, training data std~1.0)")

    check("GPU-generated output is statistically sane", _sample_statistical_sanity_gpu)

    def _cpu_gpu_consistency():
        # Not a numerical-equality check (Gibbs sampling is stochastic) --
        # just confirms a CPU-trained model of the same architecture produces
        # comparably-scaled output, i.e. the device-dependent sigma branch
        # isn't silently producing garbage on one side.
        ds = make_dataset(n_samples, seq_len, n_visible, seed=1)
        rtdbn_cpu = RTDBN(
            model=("variance_gaussian", "variance_gaussian"),
            n_visible=n_visible,
            n_hidden=(8, 4),
            steps=(1, 1),
            learning_rate=(0.001, 0.001),
            momentum=(0.0, 0.0),
            decay=(0.0, 0.0),
            temperature=(1.0, 1.0),
            use_gpu=False,
        )
        rtdbn_cpu.fit(ds, batch_size=8, epochs=(2, 2), warmup_epochs=(1, 1))
        out_cpu = rtdbn_cpu.sample(n_samples=8, n_steps=6, gibbs_steps=30)
        out_gpu = holder["model"].sample(n_samples=8, n_steps=6, gibbs_steps=30)
        std_cpu, std_gpu = out_cpu.std().item(), out_gpu.std().item()
        print(f"    (CPU std={std_cpu:.4f}, GPU std={std_gpu:.4f})")
        assert abs(std_cpu - std_gpu) < 5.0, "CPU and GPU generated scales diverge wildly"

    check("CPU vs GPU generated output are comparably scaled (no device-specific corruption)", _cpu_gpu_consistency)

    def _single_layer_gpu():
        ds = make_dataset(16, seq_len, n_visible, seed=2)
        rtdbn = RTDBN(
            model=("variance_gaussian",),
            n_visible=n_visible,
            n_hidden=(6,),
            steps=(1,),
            learning_rate=(0.001,),
            momentum=(0.0,),
            decay=(0.0,),
            temperature=(1.0,),
            use_gpu=True,
        )
        rtdbn.fit(ds, batch_size=8, epochs=(2,), warmup_epochs=(1,))
        out = rtdbn.sample(n_samples=2, n_steps=3, gibbs_steps=10)
        assert out.is_cuda
        assert out.shape == (2, 3, n_visible)

    check("single-layer RTDBN.sample() still works on GPU (regression)", _single_layer_gpu)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("RESULT: ALL GPU CHECKS PASSED")


if __name__ == "__main__":
    main()
