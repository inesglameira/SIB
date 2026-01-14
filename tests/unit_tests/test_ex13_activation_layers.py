import numpy as np

from si.neural_networks.activation import TanhActivation, SoftmaxActivation

def test_tanh_forward():
    tanh = TanhActivation()

    X = np.array([[-1.0, 0.0, 1.0]])
    out = tanh.forward_propagation(X, training=True)

    # Valores do tanh estão no intervalo [-1, 1]
    assert np.all(out >= -1)
    assert np.all(out <= 1)

    # tanh(0) == 0
    assert np.isclose(out[0, 1], 0.0)

def test_tanh_backward():
    tanh = TanhActivation()

    X = np.random.randn(3, 4)
    _ = tanh.forward_propagation(X, training=True)

    output_error = np.ones_like(X)
    input_error = tanh.backward_propagation(output_error)

    # Shape preservada
    assert input_error.shape == X.shape

def test_softmax_forward():
    softmax = SoftmaxActivation()

    X = np.array([[1.0, 2.0, 3.0]])
    out = softmax.forward_propagation(X, training=True)

    # Todas as probabilidades são >= 0
    assert np.all(out >= 0)

    # A soma das probabilidades é 1
    assert np.isclose(np.sum(out), 1.0)

def test_softmax_numerical_stability():
    softmax = SoftmaxActivation()

    # Valores grandes para testar estabilidade
    X = np.array([[1000.0, 1001.0, 1002.0]])
    out = softmax.forward_propagation(X, training=True)

    assert np.isclose(np.sum(out), 1.0)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))

def test_softmax_backward():
    softmax = SoftmaxActivation()

    X = np.random.rand(2, 3)
    _ = softmax.forward_propagation(X, training=True)

    output_error = np.ones_like(X)
    input_error = softmax.backward_propagation(output_error)

    assert input_error.shape == X.shape

def test_activation_layer_parameters_and_shape():
    tanh = TanhActivation()
    softmax = SoftmaxActivation()

    tanh.set_input_shape((5,))
    softmax.set_input_shape((5,))

    assert tanh.parameters() == 0
    assert softmax.parameters() == 0

    assert tanh.output_shape() == (5,)
    assert softmax.output_shape() == (5,)
