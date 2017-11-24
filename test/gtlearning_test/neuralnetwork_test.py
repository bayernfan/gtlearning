import unittest
from gtlearning.neuralnetwork import *

inputs = [
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
]

targets = [[1 if i == j else 0 for i in range(10)]
           for j in range(10)]


class SequentialNetworkTestCase(unittest.TestCase):
    def test_train_and_pred_with_nparray(self):
        network = SequentialNetwork().add(
            DenseLayer(9, input_shape=(25,))).add(
            DenseLayer(6)).add(
            DenseLayer(10)).compile().fit(
            inputs, targets, epochs=50000)
        x_index = 7
        y_pred = network.predict(inputs[x_index])
        for y in y_pred:
            print(y)
        self.assertEqual(sorted(y_pred, reverse=True)[0], y_pred[x_index])


if __name__ == '__main__':
    unittest.main()
