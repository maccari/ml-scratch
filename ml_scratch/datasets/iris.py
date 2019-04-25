import os
import logging
import pandas as pd
from . import Dataset

logger = logging.getLogger(__name__)


class Iris(Dataset):
    """ https://www.kaggle.com/uciml/iris
    """

    name: str = "Iris"

    def __init__(
        self, filepath: str = "iris/Iris.csv", datasets_dir: str = None
    ) -> None:
        """
        """
        super().__init__()
        if datasets_dir is None:
            datasets_dir: str = self.get_datasets_dir()
        fullpath: str = os.path.join(datasets_dir, filepath)
        self.dataset: pd.DataFrame = pd.read_csv(fullpath)
        logger.info(f"Loaded dataset {self.name} from {fullpath}")
