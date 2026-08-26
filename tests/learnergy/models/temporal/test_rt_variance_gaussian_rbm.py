import torch
from learnergy.models.temporal import rt_variance_gaussian_rbm


def test_rt_variance_gaussian_rbm_n_visible():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert model.n_visible == 128


def test_rt_variance_gaussian_rbm_n_hidden():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert model.n_hidden == 128


def test_rt_variance_gaussian_rbm_sigma_shape():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert model.sigma.size(0) == 128


def test_rt_variance_gaussian_rbm_sigma_initialized_to_ones():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert torch.all(model.sigma == 1.0)


def test_rt_variance_gaussian_rbm_sigma_is_parameter():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert isinstance(model.sigma, torch.nn.Parameter)


def test_rt_variance_gaussian_rbm_W_prime():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert model.W_prime.size(0) == 128
    assert model.W_prime.size(1) == 128


def test_rt_variance_gaussian_rbm_h0():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    assert model.h0.size(0) == 128


def test_rt_variance_gaussian_rbm_hidden_sampling():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    v = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    probs, states = model.hidden_sampling(v, h_prev)
    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rt_variance_gaussian_rbm_visible_sampling_continuous():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    h = torch.ones(1, 128)
    states, activations = model.visible_sampling(h)
    assert states.size(1) == 128
    assert activations.size(1) == 128
    assert not torch.all((states == 0) | (states == 1))


def test_rt_variance_gaussian_rbm_energy():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    samples = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    energy = model.energy(samples, h_prev)
    assert energy.size(0) == 1
    assert torch.isfinite(energy).all()


def test_rt_variance_gaussian_rbm_energy_uses_sigma():
    from learnergy.models.temporal import rtrbm
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    plain = rtrbm.RTRBM()
    samples = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    e_var = model.energy(samples, h_prev)
    e_plain = plain.energy(samples, h_prev)
    assert not torch.allclose(e_var, e_plain)


def test_rt_variance_gaussian_rbm_sigma_receives_gradient():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    x = torch.ones(2, 5, 128)
    h_prev = model.h0.unsqueeze(0).expand(2, -1)
    model.optimizer.zero_grad()
    total_cost = torch.tensor(0.0)
    for t in range(5):
        v_t = x[:, t, :]
        _, _, _, _, vis_act = model.gibbs_sampling(v_t, h_prev)
        vis_act = vis_act.detach()
        cost_t = torch.mean(model.energy(v_t, h_prev)) - \
                 torch.mean(model.energy(vis_act, h_prev))
        total_cost = total_cost + cost_t
        h_prev, _ = model.hidden_sampling(v_t, h_prev)
    total_cost.backward()
    assert model.sigma.grad is not None
    assert torch.any(model.sigma.grad != 0)


def test_rt_variance_gaussian_rbm_forward():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    x = torch.ones(2, 5, 128)
    out = model.forward(x)
    assert out.size(0) == 2
    assert out.size(1) == 5
    assert out.size(2) == 128


def test_rt_variance_gaussian_rbm_recurrence():
    model = rt_variance_gaussian_rbm.RTVarianceGaussianRBM()
    v = torch.ones(1, 128)
    h_prev_zero = torch.zeros(1, 128)
    h_prev_one = torch.ones(1, 128)
    probs_zero, _ = model.hidden_sampling(v, h_prev_zero)
    probs_one, _ = model.hidden_sampling(v, h_prev_one)
    assert not torch.allclose(probs_zero, probs_one)