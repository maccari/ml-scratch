import os
from abc import ABC, abstractmethod, abstractproperty
from typing import Any
import pandas as pd


class Dataset(ABC):

    DATASETS_DIR_ENV: str = "DATASETS_DIR"

    def __init__(self, filepath: str, datasets_dir: str = None) -> None:
        if datasets_dir is None:
            datasets_dir: str = self.datasets_dir
        self.fullpath: str = os.path.join(datasets_dir, filepath)

    @property
    def datasets_dir(self) -> str:
        if self.DATASETS_DIR_ENV not in os.environ:
            raise KeyError(
                f"Missing environment variable for datasets "
                f"{self.DATASETS_DIR_ENV}"
            )
        return os.environ[self.DATASETS_DIR_ENV]

    @abstractproperty
    def dataset(self) -> pd.DataFrame:
        pass

    def __iter__(self) -> "Dataset":
        return self

    @abstractmethod
    def __next__(self) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return str(self.dataset)

    def __repr__(self) -> str:
        return repr(self.dataset)
