import torch
from learnergy.models.temporal import rt_gaussian_rbm


def test_rt_gaussian_rbm_n_visible():
    model = rt_gaussian_rbm.RTGaussianRBM()
    assert model.n_visible == 128


def test_rt_gaussian_rbm_n_hidden():
    model = rt_gaussian_rbm.RTGaussianRBM()
    assert model.n_hidden == 128


def test_rt_gaussian_rbm_normalize():
    model = rt_gaussian_rbm.RTGaussianRBM()
    assert isinstance(model.normalize, bool)


def test_rt_gaussian_rbm_W_prime():
    model = rt_gaussian_rbm.RTGaussianRBM()
    assert model.W_prime.size(0) == 128
    assert model.W_prime.size(1) == 128


def test_rt_gaussian_rbm_h0():
    model = rt_gaussian_rbm.RTGaussianRBM()
    assert model.h0.size(0) == 128


def test_rt_gaussian_rbm_hidden_sampling():
    model = rt_gaussian_rbm.RTGaussianRBM()
    v = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    probs, states = model.hidden_sampling(v, h_prev)
    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rt_gaussian_rbm_visible_sampling_continuous():
    # Gaussian visible sampling must return continuous values, not binary
    model = rt_gaussian_rbm.RTGaussianRBM()
    h = torch.ones(1, 128)
    probs, states = model.visible_sampling(h)
    assert probs.size(1) == 128
    # Continuous outputs should not all be 0 or 1
    assert not torch.all((probs == 0) | (probs == 1))


def test_rt_gaussian_rbm_gibbs_sampling():
    model = rt_gaussian_rbm.RTGaussianRBM()
    v = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    pos_h_probs, pos_h_states, neg_h_probs, neg_h_states, vis = \
        model.gibbs_sampling(v, h_prev)
    assert pos_h_probs.size(1) == 128
    assert vis.size(1) == 128


def test_rt_gaussian_rbm_forward():
    model = rt_gaussian_rbm.RTGaussianRBM()
    x = torch.ones(2, 5, 128)
    out = model.forward(x)
    assert out.size(0) == 2
    assert out.size(1) == 5
    assert out.size(2) == 128