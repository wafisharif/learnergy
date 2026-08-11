import torch
from learnergy.models.temporal import rtdbn


def test_rtdbn_n_visible():
    model = rtdbn.RTDBN()
    assert model.n_visible == 78


def test_rtdbn_n_visible_setter():
    model = rtdbn.RTDBN()
    try:
        model.n_visible = 0
    except:
        model.n_visible = 78
    assert model.n_visible == 78


def test_rtdbn_n_hidden():
    model = rtdbn.RTDBN()
    assert model.n_hidden == (64,)


def test_rtdbn_n_layers():
    model = rtdbn.RTDBN()
    assert model.n_layers == 1


def test_rtdbn_n_layers_setter():
    model = rtdbn.RTDBN()
    try:
        model.n_layers = 0
    except:
        model.n_layers = 1
    assert model.n_layers == 1


def test_rtdbn_models_length():
    model = rtdbn.RTDBN()
    assert len(model.models) == 1


def test_rtdbn_models_type():
    from learnergy.models.temporal.rt_variance_gaussian_rbm import \
        RTVarianceGaussianRBM
    model = rtdbn.RTDBN()
    assert isinstance(model.models[0], RTVarianceGaussianRBM)


def test_rtdbn_encode_shape():
    model = rtdbn.RTDBN()
    x = torch.ones(2, 20, 78)
    emb = model.encode(x)
    assert emb.size(0) == 2
    assert emb.size(1) == 64


def test_rtdbn_forward_shape():
    model = rtdbn.RTDBN()
    x = torch.ones(2, 20, 78)
    out = model.forward(x)
    assert out.size(0) == 2
    assert out.size(1) == 64


def test_rtdbn_encode_collapses_time():
    # encode must collapse (batch, seq_len, n_hidden) -> (batch, n_hidden)
    model = rtdbn.RTDBN()
    x = torch.ones(4, 10, 78)
    emb = model.encode(x)
    assert emb.dim() == 2
    assert emb.size(0) == 4
    assert emb.size(1) == 64


def test_rtdbn_multilayer():
    # Two-layer RTDBN should work with matching n_hidden sizes
    model = rtdbn.RTDBN(
        model=("variance_gaussian", "variance_gaussian"),
        n_visible=78,
        n_hidden=(64, 32),
        steps=(1, 1),
        learning_rate=(0.001, 0.001),
        momentum=(0.0, 0.0),
        decay=(0.0, 0.0),
        temperature=(1.0, 1.0),
    )
    assert model.n_layers == 2
    assert len(model.models) == 2
    assert model.models[0].n_visible == 78
    assert model.models[1].n_visible == 64


def test_rtdbn_invalid_model_type():
    try:
        model = rtdbn.RTDBN(model=("invalid_type",))
        assert False, "Should have raised an error"
    except:
        pass