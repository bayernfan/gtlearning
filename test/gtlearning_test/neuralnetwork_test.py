import unittest, numpy as np
from gtlearning.neuralnetwork import *


class SequentialNetworkTestCase(unittest.TestCase):
    def setUp(self):
        small_inputs = [
            [1, 1, 1,
             1, 0, 1,
             1, 0, 1,
             1, 0, 1,
             1, 1, 1],
            [0, 1, 0,
             0, 1, 0,
             0, 1, 0,
             0, 1, 0,
             0, 1, 0],
            [1, 1, 1,
             0, 0, 1,
             1, 1, 1,
             1, 0, 0,
             1, 1, 1],
            [1, 1, 1,
             0, 0, 1,
             1, 1, 1,
             0, 0, 1,
             1, 1, 1],
            [1, 0, 1,
             1, 0, 1,
             1, 1, 1,
             0, 0, 1,
             0, 0, 1],
            [1, 1, 1,
             1, 0, 0,
             1, 1, 1,
             0, 0, 1,
             1, 1, 1],
            [1, 1, 1,
             1, 0, 0,
             1, 1, 1,
             1, 0, 1,
             1, 1, 1],
            [1, 1, 1,
             0, 0, 1,
             0, 0, 1,
             0, 0, 1,
             0, 0, 1],
            [1, 1, 1,
             1, 0, 1,
             1, 1, 1,
             1, 0, 1,
             1, 1, 1],
            [1, 1, 1,
             1, 0, 1,
             1, 1, 1,
             0, 0, 1,
             1, 1, 1],
        ]
        small_shape = (5, 3)

        mid_inputs = [[small[i]
                       for i in [0, 1, 1, 1, 2,
                                 3, 4, 4, 4, 5,
                                 3, 4, 4, 4, 5,
                                 6, 7, 7, 7, 8,
                                 9, 10, 10, 10, 11,
                                 9, 10, 10, 10, 11,
                                 12, 13, 13, 13, 14]]
                      for small in small_inputs]
        mid_shape = (7, 5)

        big_inputs = [[small[i]
                       for i in [0, 1, 1, 1, 1, 1, 2,
                                 3, 4, 4, 4, 4, 4, 5,
                                 3, 4, 4, 4, 4, 4, 5,
                                 3, 4, 4, 4, 4, 4, 5,
                                 6, 7, 7, 7, 7, 7, 8,
                                 9, 10, 10, 10, 10, 10, 11,
                                 9, 10, 10, 10, 10, 10, 11,
                                 9, 10, 10, 10, 10, 10, 11,
                                 12, 13, 13, 13, 13, 13, 14]]
                      for small in small_inputs]
        big_shape = (9, 7)
        self.output_size = output_size = 10
        inputs = [np.array(l)
                  for l in mid_inputs][:output_size]
        shape = mid_shape
        self.inputs = inputs
        self.targets = [np.array([1 if i == j else 0
                                  for i in range(output_size)])
                        for j in range(output_size)]
        self.inputs2D = [np.reshape(arr, shape) for arr in inputs]

    def test_denselayers(self):
        output_size = self.output_size
        network = SequentialNetwork().add(
            DenseLayer(12, input_shape=self.inputs[0].shape)).add(
            DenseLayer(output_size)).compile().fit(
            self.inputs, self.targets, epochs=4999, verbose=0)
        for i in range(output_size):
            print((network.predict(self.inputs[i]) * 100).tolist())
        self.assertTrue(True)

    def test_conv2dlayers(self):
        output_size = self.output_size
        network = SequentialNetwork().add(
            Conv2DLayer(2, kernel_size=(3,3), input_shape=self.inputs2D[0].shape)).add(
            FlattenLayer()).add(
            DenseLayer(output_size)).compile().fit(
            self.inputs2D, self.targets, epochs=2999, verbose=0)
        for i in range(output_size):
            print((network.predict(self.inputs2D[i]) * 100).tolist())
        self.assertTrue(True)


    def test_reluActivation(self):
        output_size = self.output_size
        network = SequentialNetwork().add(
            Conv2DLayer(2, kernel_size=(3,3), activation='relu', input_shape=self.inputs2D[0].shape)).add(
            FlattenLayer()).add(
            DenseLayer(output_size)).compile().fit(
            self.inputs2D, self.targets, epochs=999, verbose=0)
        for i in range(output_size):
            print((network.predict(self.inputs2D[i]) * 100).tolist())
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
