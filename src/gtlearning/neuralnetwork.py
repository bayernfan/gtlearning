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

    def fit(self, x_train, y_train, batch_size=None, epochs=1):
        # todo: use the batch_size

        for _ in range(epochs):
            for i, inputs in enumerate(x_train):
                self.layers[0].feed_forward(inputs)
                self.layers[-1].backpropagate(y_train[i])
        return self

    def predict(self, x, batch_size=None):
        return self.layers[0].feed_forward(x)


class Layer(object):
    def __init__(self, activation="sigmoid"):
        self.weights = []
        self.shape = None
        self.prev_layer = None
        self.next_layer = None
        if activation == "sigmoid":
            self.activation = SigmoidActivation()
        else:
            self.activation = activation

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
        super(DenseLayer, self).__init__(activation=activation)
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
        outputs = [self.activation.activate(np.dot(weights_per_neuron, inputs) + self.biases[i])
                   for i, weights_per_neuron in enumerate(self.weights)]
        self.outputs = outputs
        return self.next_layer.feed_forward(outputs) if self.next_layer else outputs

    def backpropagate(self, targets):
        # todo: adjusstments for biaaes
        output_deltas = [self.activation.derivate(output, output - target)
                         for output, target in zip(self.outputs, targets)]
        for i, weights_per_neuron in enumerate(self.weights):
            for j, prev_output in enumerate(self.inputs):
                weights_per_neuron[j] -= output_deltas[i] * prev_output
        propagated_targets = [prev_output - np.dot(output_deltas, [n[i] for n in self.weights])
                for i, prev_output in enumerate(self.inputs)]
        return self.prev_layer.backpropagate(propagated_targets) if  self.prev_layer else propagated_targets


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
        return 1. / (1. + math.exp(-input))

    def derivate(self, x, delta_x):
        return x * (1. - x) * delta_x
