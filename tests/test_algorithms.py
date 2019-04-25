import logging

from base import BaseTest
from ml_scratch.datasets import HeartDisease
from ml_scratch.algorithms import Perceptron, LogisticRegression

logger = logging.getLogger(__name__)


class TestNeuralNetworks(BaseTest):
    def test_perceptron(self):
        dataset = HeartDisease()
        perceptron = Perceptron(num_inputs=len(dataset.input_columns))
        accuracy_per_run = self._run_classifier(dataset, perceptron)
        self._test_classifier_score(accuracy_per_run, 0.7, 0.1)

    def test_logistic_regression(self):
        dataset = HeartDisease()
        logreg = LogisticRegression(
            num_inputs=len(dataset.input_columns), learning_rate=0.5
        )
        logger.info(f"Test model: {logreg}")
        accuracy_per_run = self._run_classifier(dataset, logreg)
        self._test_classifier_score(accuracy_per_run, 0.7, 0.1)
