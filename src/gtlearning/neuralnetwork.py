import math, numpy as np, random
from functools import reduce


class SequentialNetwork(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self):
        reduce(lambda l, r: l.link(r), self.layers)
        return self

    def fit(self, x_train, y_train, batch_size=None, epochs=1, verbose=0):
        # todo: use the batch_size

        for i_epochs in range(epochs):
            if verbose > 0:
                print('Epoche ({}) {}'.format(i_epochs, '{'))
            for inputs, targets in zip(x_train, y_train):
                outputs = self.layers[0].feed_forward(inputs)
                self.layers[-1].back_propagate(outputs - targets)
            if verbose > 0:
                print('}')
        return self

    def predict(self, x, batch_size=None):
        return self.layers[0].feed_forward(x)


class Layer(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.biases = None
        self.shape = None
        self.prev_layer = None
        self.next_layer = None

    def link(self, next_layer):
        self.next_layer = next_layer
        return next_layer.link_to(self)

    def link_to(self, prev_layer):
        self.prev_layer = prev_layer
        return self

    def feed_forward(self, inputs):
        pass

    def back_propagate(self, targets):
        pass


class DenseLayer(Layer):
    def __init__(self, units, input_shape=None, activation="sigmoid"):
        super(DenseLayer, self).__init__()

        activation = activation.lower()
        if activation == "sigmoid":
            self.activation = SigmoidActivation()
        else:
            self.activation = activation  # todo: allow other activations

        self.shape = (units,)

        self.input_shape = input_shape
        if input_shape is not None:
            self._init_weights_and_biases()

    def link_to(self, prev_layer):
        if self.input_shape is None:
            self.input_shape = prev_layer.shape
            self._init_weights_and_biases()
        return super(DenseLayer, self).link_to(prev_layer)

    def _init_weights_and_biases(self):
        self.weights = np.random.rand(self.shape[0], self.input_shape[0])
        self.biases = np.zeros(self.shape[0])

    def feed_forward(self, inputs):
        self.inputs = inputs
        outputs = [self.activation.activate(np.dot(weights, inputs) + bias)
                   for weights, bias in zip(self.weights, self.biases)]
        self.outputs = outputs = np.array(outputs)
        return self.next_layer.feed_forward(outputs) if self.next_layer else outputs

    def back_propagate(self, output_deltas):
        input_deltas = self.activation.derivate(self.outputs, output_deltas)
        n, m = self.weights.shape;
        for i in range(n):
            input_delta = input_deltas[i]
            self.weights[i] -= input_delta * self.inputs
            self.biases[i] -= input_delta

        propagated_deltas = [np.dot(input_deltas, self.weights[:, i])
                             for i in range(m)]
        propagated_deltas = np.array(propagated_deltas)
        return self.prev_layer.back_propagate(propagated_deltas) if self.prev_layer else propagated_deltas


class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()
        self.input_shape = None

    def link_to(self, prev_layer):
        if self.input_shape is None:
            input_shape = self.input_shape = prev_layer.shape
            self.shape = (reduce(lambda l, r: l * r, input_shape),)
        return super(FlattenLayer, self).link_to(prev_layer)

    def feed_forward(self, inputs):
        self.inputs = inputs
        outputs = self.outputs = np.reshape(inputs, self.shape)
        return self.next_layer.feed_forward(outputs) if self.next_layer else outputs

    def back_propagate(self, targets):
        propagated_targets = np.reshape(targets, self.input_shape)
        return self.prev_layer.back_propagate(propagated_targets) if self.prev_layer else propagated_targets


class Conv2DLayer(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', input_shape=None, activation="sigmoid"):
        super(Conv2DLayer, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.lower()
        self.input_shape = input_shape[:3] if len(input_shape) >= 3 else (*input_shape[:2], 1)
        if input_shape is not None:
            self._init_shape()
            self._init_weights_and_biases()

        activation = activation.lower()
        if activation == "sigmoid":
            self.activation = SigmoidActivation()
        else:
            self.activation = activation  # todo: allow other activations

    def link_to(self, prev_layer):
        if self.input_shape is None:
            self.input_shape = prev_layer.shape
            self._init_shape()
            self._init_weights_and_biases()
        return super(Conv2DLayer, self).link_to(prev_layer)

    def _init_shape(self):
        shape_xy = [self._round_to_int((self.input_shape[i] - (self.kernel_size[i] - 1)) / self.strides[i])
                    for i in range(2)]
        self.shape = (*shape_xy, self.filters)

    def _round_to_int(self, x):
        round_fun = math.floor if self.padding == 'valid' else math.ceil
        return int(round_fun(x))

    def _init_weights_and_biases(self):
        shape = self.shape
        self.weights = np.random.rand(*shape, *self.kernel_size, self.input_shape[2])
        self.biases = np.zeros(reduce(lambda l, r: l * r, shape)).reshape(shape)

    def feed_forward(self, inputs):
        if inputs.shape != self.input_shape:
            inputs = inputs.reshape(self.input_shape)
        if self.padding == 'same':
            pass  # todo: allow the 'same' padding
        self.inputs = inputs
        stride_x, stride_y = self.strides
        kernel_x, kernel_y = self.kernel_size
        n_x, n_y, n_f, m_x, m_y, m_f = self.weights.shape
        m_size = m_x * m_y * m_f
        outputs = [
            self.activation.activate(
                np.dot(
                    self.weights[i_x, i_y, i_f, :, :, :].reshape((m_size,)),
                    inputs[i_x * stride_x:i_x * stride_x + kernel_x,
                    i_y * stride_y:i_y * stride_y + kernel_y,
                    :].reshape((m_size,))
                ) +
                self.biases[i_x, i_y, i_f])
            for i_x in range(n_x)
            for i_y in range(n_y)
            for i_f in range(n_f)]
        self.outputs = outputs = np.array(outputs).reshape(self.shape)
        return self.next_layer.feed_forward(outputs) if self.next_layer else outputs

    def back_propagate(self, output_deltas):
        input_deltas = self.activation.derivate(self.outputs, output_deltas)
        stride_x, stride_y = self.strides
        kernel_x, kernel_y = self.kernel_size
        n_x, n_y, n_f, m_x, m_y, m_f = self.weights.shape
        n_size = n_x * n_y * n_f
        for i_x, i_y, i_f in [(x, y, f) for x in range(n_x) for y in range(n_y) for f in range(n_f)]:
            input_delta = input_deltas[i_x, i_y, i_f]
            self.biases[i_x, i_y, i_f] -= input_delta
            self.weights[i_x, i_y, i_f] -= input_delta * self.inputs[i_x * stride_x:i_x * stride_x + kernel_x,
                                                         i_y * stride_y:i_y * stride_y + kernel_y, :]
        propagated_deltas = [np.dot(input_deltas.reshape((n_size,)),
                                    self.weights[:, :, :, i_x, i_y, i_f].reshape((n_size,)))
                             for i_x in range(m_x)
                             for i_y in range(m_y)
                             for i_f in range(m_f)]
        propagated_deltas = np.array(propagated_deltas)
        return self.prev_layer.back_propagate(propagated_deltas) if self.prev_layer else propagated_deltas


class Activation(object):
    def __init__(self):
        pass

    def activate(self, input):
        return input

    def derivatie(self, x, delta_x):
        return 1.


class SigmoidActivation(Activation):
    def __init__(self):
        super(SigmoidActivation, self).__init__()

    def activate(self, input):
        return 1. / (1. + np.exp(-input))

    def derivate(self, x, delta_x):
        return x * (1 - x) * delta_x
