from abc import ABC, abstractproperty, abstractmethod
import numpy as np


class Algorithm(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractproperty
    def num_parameters(self) -> int:
        """ Total number of parameters of the model """
        pass


class SupervisedAlgorithm(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(
        self, inputs: np.ndarray, targets: np.ndarray, *args, **kwargs
    ) -> None:
        pass
