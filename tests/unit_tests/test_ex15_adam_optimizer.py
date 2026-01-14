import numpy as np
from si.neural_networks.optimizers import Adam

def test_adam_initialization():
    opt = Adam(learning_rate=0.01)

    assert opt.learning_rate == 0.01
    assert opt.beta_1 == 0.9
    assert opt.beta_2 == 0.999
    assert opt.epsilon == 1e-8
    assert opt.t == 0

def test_adam_updates_weights():
    opt = Adam(learning_rate=0.01)

    weights = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.1, 0.1])

    new_weights = opt.update(weights, grads)

    assert not np.allclose(weights, new_weights)

def test_adam_shape_preservation():
    opt = Adam()

    weights = np.random.rand(4, 3)
    grads = np.random.rand(4, 3)

    new_weights = opt.update(weights, grads)

    assert new_weights.shape == weights.shape

def test_adam_time_step_increment():
    opt = Adam()

    weights = np.array([1.0, 2.0])
    grads = np.array([0.1, 0.2])

    opt.update(weights, grads)
    assert opt.t == 1

    opt.update(weights, grads)
    assert opt.t == 2

def test_adam_numerical_stability():
    opt = Adam()

    weights = np.array([1.0, 1.0])
    grads = np.array([0.0, 0.0])

    new_weights = opt.update(weights, grads)

    assert not np.any(np.isnan(new_weights))
    assert not np.any(np.isinf(new_weights))
