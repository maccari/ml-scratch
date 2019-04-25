from base import BaseTest
from ml_scratch.datasets import HeartDisease


class TestHeartDiseaseDataset(BaseTest):
    def test_load_dataset(self):
        dataset = HeartDisease()
        self.assertEqual(len(dataset), 303)
