import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

from . import Dataset

logger = logging.getLogger(__name__)


class HeartDisease(Dataset):
    """ https://www.kaggle.com/ronitf/heart-disease-uci
    """

    name: str = "HeartDisease"

    def __init__(
        self,
        filepath: str = "heart-disease-uci/heart.csv",
        datasets_dir: str = None,
    ) -> None:
        """
        """
        super().__init__(filepath, datasets_dir)
        self._dataset: pd.DataFrame = pd.read_csv(self.fullpath)
        logger.info(f"Loaded dataset {self.name} from {self.fullpath}")

    @property
    def inputs(self) -> np.ndarray:
        return self.dataset[self.input_columns].values

    @property
    def targets(self) -> np.ndarray:
        return self.dataset[self.target_column].values

    @property
    def input_columns(self) -> List[str]:
        return self.dataset.columns[self.dataset.columns != "target"].values

    @property
    def target_column(self) -> str:
        return "target"

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def shuffle(self) -> None:
        self._dataset = self._dataset.sample(frac=1).reset_index(drop=True)

    def normalize(self) -> None:
        for input_column in self.input_columns:
            values = self.dataset[input_column]
            range_ = values.max() - values.min()
            self.dataset[input_column] = (values - values.min()) / range_

    def __next__(self):
        for index, row in self.dataset.iterrows():
            yield row[self.input_columns], row[self.target_column]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        row = self.dataset[index]
        return row[self.input_columns].values, row[self.target_column].values

    def __len__(self) -> int:
        return self.dataset.shape[0]
