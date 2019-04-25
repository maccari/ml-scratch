from typing import Iterable, Any
import numpy as np
import logging
from abc import abstractmethod

from ml_scratch.algorithms import Algorithm
from ml_scratch.algorithms.activation import step, sigmoid
from ml_scratch.algorithms.optimizer import compute_gradient

logger = logging.getLogger(__name__)


class NeuralNetwork(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ """
        return self.forward(inputs)

    @abstractmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        """ Args:
                inputs: shape [n_inputs, n_features]
            Retuns:
                outputs of shape [n_inputs]
        """
        pass

    @abstractmethod
    def backward(
        inputs: np.ndarray, outputs: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """ Args:
                inputs: input matrix
                outputs: output matrix
                targets: target matrix
            Returns:
                Number of errors
        """
        pass


class SingleLayerFFNN(NeuralNetwork):
    """ Single-layer Feed-Forward Neural Network """

    def __init__(
        self,
        num_inputs: int,
        init_range: float = 0.1,
        learning_rate: float = 0.01,
    ) -> None:
        """ """
        super().__init__()
        self.num_inputs = num_inputs
        self.init_range = init_range
        self.learning_rate = learning_rate
        self._init_weights()

    def reset(self) -> None:
        self._init_weights()

    def _init_weights(self) -> None:
        self.weights = np.random.uniform(
            -self.init_range, self.init_range, self.num_inputs
        )
        self.bias = np.random.uniform(-self.init_range, self.init_range)

    def forward(self, inputs: Iterable[Any]) -> Iterable[float]:
        """ """
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    @abstractmethod
    def activation(self, outputs: np.ndarray) -> np.ndarray:
        """ """
        pass

    def _format_outputs(
        self, outputs: np.ndarray, copy: bool = True
    ) -> np.ndarray:
        """ """
        if copy:
            outputs = np.copy(outputs)
        return outputs

    def backward(
        self, inputs: np.ndarray, outputs: np.ndarray, targets: np.ndarray
    ) -> float:
        """ """
        outputs = self._format_outputs(outputs, copy=False)
        self.update_weights(inputs, outputs, targets)
        return (targets != self.outputs2pred(outputs)).sum()

    def train(
        self, inputs: np.ndarray, targets: np.ndarray, num_epochs: int
    ) -> None:
        """ """
        preds = self.outputs2pred(
            self._format_outputs(self.forward(inputs), copy=False)
        )
        num_errors = (targets != preds).sum()
        logger.debug(
            f"Ratio of error before training: {num_errors}/{len(inputs)}"
        )
        logger.debug("Start training")
        targets = self._format_outputs(targets)
        for epoch in range(num_epochs):
            outputs = self.forward(inputs)
            num_errors = self.backward(inputs, outputs, targets)
            logger.debug(
                f"EPOCH {epoch} -- "
                f"ratio of errors: {num_errors}/{len(inputs)}"
            )
        logger.debug("End training")

    @abstractmethod
    def update_weights(
        self, inputs: np.ndarray, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """ """
        pass

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """ """
        return self.outputs2pred(self.forward(inputs))

    @abstractmethod
    def outputs2pred(self, outputs: np.ndarray) -> np.ndarray:
        """ """
        pass

    @property
    def num_parameters(self):
        return len(self.weights) + 1  # + 1 for bias

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.num_inputs}, {self.init_range}, {self.learning_rate})"
        )

    __str__ = __repr__


class Perceptron(SingleLayerFFNN):
    def __init__(self, num_inputs: int, init_range: float = 0.1) -> None:
        """ The learning rate does not influence the perceptron convergence
        """
        super().__init__(num_inputs, init_range, 1)

    def activation(self, outputs: np.ndarray) -> np.ndarray:
        return step(outputs)

    def update_weights(
        self, inputs: np.ndarray, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """ """
        errors = targets - predictions
        for input_, error in zip(inputs, errors):
            self.weights += error * input_
            self.bias += error

    def _format_outputs(
        self, outputs: np.ndarray, copy: bool = True
    ) -> np.ndarray:
        """ """
        if copy:
            outputs = np.copy(outputs)
        outputs[outputs == 0] = -1
        return outputs

    def outputs2pred(self, outputs: np.ndarray) -> np.ndarray:
        """ """
        return outputs


class LogisticRegression(SingleLayerFFNN):
    def __init__(
        self,
        num_inputs: int,
        init_range: float = 0.1,
        learning_rate: float = 0.01,
    ) -> None:
        """ """
        super().__init__(num_inputs, init_range, learning_rate)

    def activation(self, outputs: np.ndarray) -> np.ndarray:
        return sigmoid(outputs)

    def update_weights(
        self, inputs: np.ndarray, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """ """
        # add dimension for bias
        inputs = np.append(inputs, np.ones((len(inputs), 1)), axis=1)
        gradient = compute_gradient(
            inputs, predictions, targets, "cross_entropy"
        )
        self.weights -= gradient[:-1] * self.learning_rate
        self.bias -= gradient[-1] * self.learning_rate

    def outputs2pred(self, outputs: np.ndarray) -> np.ndarray:
        """ """
        return np.where(outputs > 0.5, 1, 0)
