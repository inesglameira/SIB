import numpy as np

from si.neural_networks.layers import Dropout


def test_dropout_forward_training():
    """
    Testa o comportamento do Dropout em modo de treino.
    Verifica:
    - shape preservada
    - existência de valores a zero
    - aplicação do scaling factor
    """

    np.random.seed(42)

    dropout = Dropout(p=0.5)

    X = np.random.rand(4, 5)

    out = dropout.forward_propagation(X, training=True)

    # Shape não muda
    assert out.shape == X.shape

    # Deve haver valores a zero (neurónios desligados)
    assert np.any(out == 0)

    # Deve haver valores diferentes do input original (scaling)
    assert not np.allclose(out, X)


def test_dropout_forward_inference():
    """
    Testa o comportamento do Dropout em modo de inferência.
    O output deve ser exatamente igual ao input.
    """

    dropout = Dropout(p=0.3)

    X = np.random.rand(3, 4)

    out = dropout.forward_propagation(X, training=False)

    assert np.allclose(out, X)


def test_dropout_backward():
    """
    Testa o backward propagation.
    O erro deve ser propagado apenas pelos neurónios ativos.
    """

    np.random.seed(0)

    dropout = Dropout(p=0.5)

    X = np.random.rand(2, 3)
    _ = dropout.forward_propagation(X, training=True)

    output_error = np.ones_like(X)

    input_error = dropout.backward_propagation(output_error)

    # Shape preservada
    assert input_error.shape == X.shape

    # Onde a máscara é zero, o erro também deve ser zero
    assert np.all(input_error[dropout.mask == 0] == 0)


def test_dropout_parameters_and_output_shape():
    """
    Testa métodos auxiliares:
    - parameters deve devolver 0 ou {}
    - output_shape deve devolver input_shape
    """

    dropout = Dropout(p=0.4)
    dropout.set_input_shape((5,))

    assert dropout.parameters() == {}
    assert dropout.output_shape() == (5,)
