import os
import unittest
import logging
import numpy as np

from ml_scratch.algorithms import SupervisedAlgorithm
from ml_scratch.datasets import Dataset
from ml_scratch.metrics import accuracy_score

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


class BaseTest(unittest.TestCase):
    def _run_classifier(
        self,
        dataset: Dataset,
        algorithm: SupervisedAlgorithm,
        num_runs: int = 10,
        num_epochs: int = 10,
    ):
        accuracy_per_run = []
        for _ in range(num_runs):
            dataset.shuffle()
            dataset.normalize()
            algorithm.reset()
            algorithm.train(
                dataset.inputs, dataset.targets, num_epochs=num_epochs
            )
            # replace with CV instead of overfit
            predictions = algorithm.predict(dataset.inputs)
            accuracy = accuracy_score(predictions, dataset.targets)
            accuracy_per_run.append(accuracy)
        return accuracy_per_run

    def _test_classifier_score(self, scores, min_mean, max_std):
        self.assertTrue(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        logger.info(f"mean:{mean}, std: {std}")
        self.assertGreaterEqual(mean, min_mean)
        self.assertLessEqual(std, max_std)
