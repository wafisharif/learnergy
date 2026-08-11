import torch
from learnergy.models.temporal import rtrbm


def test_rtrbm_n_visible():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.n_visible == 128


def test_rtrbm_n_visible_setter():
    new_rtrbm = rtrbm.RTRBM()
    try:
        new_rtrbm.n_visible = "a"
    except:
        new_rtrbm.n_visible = 1
    assert new_rtrbm.n_visible == 1
    try:
        new_rtrbm.n_visible = 0
    except:
        new_rtrbm.n_visible = 1
    assert new_rtrbm.n_visible == 1


def test_rtrbm_n_hidden():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.n_hidden == 128


def test_rtrbm_n_hidden_setter():
    new_rtrbm = rtrbm.RTRBM()
    try:
        new_rtrbm.n_hidden = "a"
    except:
        new_rtrbm.n_hidden = 1
    assert new_rtrbm.n_hidden == 1
    try:
        new_rtrbm.n_hidden = 0
    except:
        new_rtrbm.n_hidden = 1
    assert new_rtrbm.n_hidden == 1


def test_rtrbm_steps():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.steps == 1


def test_rtrbm_steps_setter():
    new_rtrbm = rtrbm.RTRBM()
    try:
        new_rtrbm.steps = "a"
    except:
        new_rtrbm.steps = 1
    assert new_rtrbm.steps == 1
    try:
        new_rtrbm.steps = 0
    except:
        new_rtrbm.steps = 1
    assert new_rtrbm.steps == 1


def test_rtrbm_lr():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.lr == 0.1


def test_rtrbm_lr_setter():
    new_rtrbm = rtrbm.RTRBM()
    try:
        new_rtrbm.lr = "a"
    except:
        new_rtrbm.lr = 0.1
    assert new_rtrbm.lr == 0.1
    try:
        new_rtrbm.lr = -1
    except:
        new_rtrbm.lr = 0.1
    assert new_rtrbm.lr == 0.1


def test_rtrbm_W():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.W.size(0) == 128
    assert new_rtrbm.W.size(1) == 128


def test_rtrbm_W_prime():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.W_prime.size(0) == 128
    assert new_rtrbm.W_prime.size(1) == 128


def test_rtrbm_h0():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.h0.size(0) == 128


def test_rtrbm_a():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.a.size(0) == 128


def test_rtrbm_b():
    new_rtrbm = rtrbm.RTRBM()
    assert new_rtrbm.b.size(0) == 128


def test_rtrbm_hidden_sampling():
    new_rtrbm = rtrbm.RTRBM()
    v = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    probs, states = new_rtrbm.hidden_sampling(v, h_prev)
    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rtrbm_visible_sampling():
    new_rtrbm = rtrbm.RTRBM()
    h = torch.ones(1, 128)
    probs, states = new_rtrbm.visible_sampling(h)
    assert probs.size(1) == 128
    assert states.size(1) == 128


def test_rtrbm_energy():
    new_rtrbm = rtrbm.RTRBM()
    samples = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    energy = new_rtrbm.energy(samples, h_prev)
    assert energy.detach().numpy()[0] < 0


def test_rtrbm_gibbs_sampling():
    new_rtrbm = rtrbm.RTRBM()
    v = torch.ones(1, 128)
    h_prev = torch.zeros(1, 128)
    pos_h_probs, pos_h_states, neg_h_probs, neg_h_states, vis = \
        new_rtrbm.gibbs_sampling(v, h_prev)
    assert pos_h_probs.size(1) == 128
    assert vis.size(1) == 128


def test_rtrbm_fit_subseries():
    new_rtrbm = rtrbm.RTRBM()
    x = torch.rand(2, 5, 128)
    mse = new_rtrbm.fit_subseries(x)
    assert torch.isfinite(mse)


def test_rtrbm_recurrence():
    new_rtrbm = rtrbm.RTRBM()
    v = torch.ones(1, 128)
    h_prev_zero = torch.zeros(1, 128)
    h_prev_one = torch.ones(1, 128)
    probs_zero, _ = new_rtrbm.hidden_sampling(v, h_prev_zero)
    probs_one, _ = new_rtrbm.hidden_sampling(v, h_prev_one)
    assert not torch.allclose(probs_zero, probs_one)
