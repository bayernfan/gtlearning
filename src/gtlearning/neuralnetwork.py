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
            random.shuffle(x_train)
            for i, inputs in enumerate(x_train):
                self.layers[0].feed_forward(inputs)
                self.layers[-1].backpropagate(y_train[i])
            if verbose > 0:
                print('Epoche {} finished.'.format(i_epochs))
        return self

    def predict(self, x, batch_size=None):
        return self.layers[0].feed_forward(x)


class Layer(object):
    def __init__(self):
        self.weights = []
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

    def backpropagate(self, targets):
        pass


class DenseLayer(Layer):
    def __init__(self, units, input_shape=None, activation="sigmoid"):
        super(DenseLayer, self).__init__()
        if activation == "sigmoid":
            self.activation = SigmoidActivation()
        else:
            self.activation = activation
        self.shape = (units,)
        self.input_shape = input_shape
        if input_shape is not None:
            self._fill_weights_and_biases()

    def link_to(self, prev_layer):
        if self.input_shape is None:
            self.input_shape = prev_layer.shape
            self._fill_weights_and_biases()
        return super(DenseLayer, self).link_to(prev_layer)

    def _fill_weights_and_biases(self):
        self.weights = [[random.random() for j in range(self.input_shape[0])]
                        for i in range(self.shape[0])]
        self.biases = [0 for i in range(self.shape[0])]

    def feed_forward(self, inputs):
        self.inputs = inputs
        outputs = self.outputs = [self.activation.activate(np.dot(weights_per_neuron, inputs) + self.biases[i])
                                  for i, weights_per_neuron in enumerate(self.weights)]
        return self.next_layer.feed_forward(outputs) if self.next_layer else outputs

    def backpropagate(self, targets):
        output_deltas = [self.activation.derivate(output, output - target)
                         for output, target in zip(self.outputs, targets)]
        for i, (weights_per_neuron, output_delta) in enumerate(zip(self.weights, output_deltas)):
            for j, prev_output in enumerate(self.inputs):
                self.biases[i] -= output_delta
                weights_per_neuron[j] -= output_delta * prev_output
        propagated_targets = [prev_output - np.dot(output_deltas, [n[i] for n in self.weights])
                              for i, prev_output in enumerate(self.inputs)]
        return self.prev_layer.backpropagate(propagated_targets) if self.prev_layer else propagated_targets


class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()
        self.shape = None
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

    def backpropagate(self, targets):
        propagated_targets = np.reshape(targets, self.input_shape)
        return self.prev_layer.backpropagate(propagated_targets) if self.prev_layer else propagated_targets


class Activation(object):
    def __init__(self):
        pass

    def activate(self, input):
        return input

    def derivate(self, x, delta_x):
        return 1.


class SigmoidActivation(Activation):
    def __init__(self):
        super(SigmoidActivation, self).__init__()

    def activate(self, input):
        return 1. / (1. + np.exp(-input))

    def derivate(self, x, delta_x):
        return x * (1. - x) * delta_x
