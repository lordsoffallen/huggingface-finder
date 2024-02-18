from typing import Any
from kedro.io.core import AbstractDataset
from huggingface_hub.utils import disable_progress_bars
from pathlib import Path
from datasets import Dataset
from .api_tools import update_text_and_metadata_in_parallel, get_dataset_ids, \
    get_model_ids, find_missing_ids, compute_partition_number
from .fs import mkdir
import logging


logger = logging.getLogger(__name__)


class HFScrapper(AbstractDataset):
    def __init__(self, cachepath: str, cache_interval: int = 1000, n_jobs: int = 10):
        self.cachepath = cachepath
        self.cache_interval = cache_interval

        self.datasets_cachepath = mkdir(Path(cachepath).joinpath('datasets'))
        self.dataset_ids_path = self.datasets_cachepath.joinpath("ids.pkl")

        self.models_cachepath = mkdir(Path(cachepath).joinpath('models'))
        self.model_ids_path = self.models_cachepath.joinpath("ids.pkl")
        self.n_jobs = n_jobs

        disable_progress_bars()     # no tqdm in api call

    def _describe(self) -> dict[str, Any]:
        return {
            "cachepath": self.cachepath,
            "cache_interval": self.cache_interval,
            "datasets_cachepath": self.datasets_cachepath,
            "dataset_ids_path": self.dataset_ids_path,
            "models_cachepath": self.models_cachepath,
            "model_ids_path": self.model_ids_path,
            "n_jobs": self.n_jobs,
        }

    def _load(self):
        # Populate cache
        self.get_models()
        self.get_datasets()

        # data_ds = Dataset.from_json(str(self.datasets_cachepath.joinpath("*.json")))
        # model_ds = Dataset.from_json(str(self.models_cachepath.joinpath("*.json")))
        #
        # return data_ds, model_ds

    def _save(self, data) -> None:
        raise NotImplementedError

    def get_datasets(self):
        ids = get_dataset_ids(self.dataset_ids_path)
        missing_ids = find_missing_ids(ids, self.datasets_cachepath)

        if missing_ids is not None:
            ids = missing_ids

        if len(ids) > 0:
            partition_start = compute_partition_number(self.datasets_cachepath)

            update_text_and_metadata_in_parallel(
                ids,
                path=self.datasets_cachepath,
                card_type="dataset",
                partition_start=partition_start,
                cache_interval=self.cache_interval,
                n_jobs=self.n_jobs,
                padding_size=4,
            )
        else:
            logger.info("No missing ids, all computation is done for datasets")

    def get_models(self):
        ids = get_model_ids(self.model_ids_path)
        missing_ids = find_missing_ids(ids, self.models_cachepath)

        if missing_ids is not None:
            ids = missing_ids

        if len(ids) > 0:
            partition_start = compute_partition_number(self.models_cachepath)

            update_text_and_metadata_in_parallel(
                ids,
                path=self.models_cachepath,
                card_type="model",
                partition_start=partition_start,
                cache_interval=self.cache_interval,
                n_jobs=self.n_jobs,
                padding_size=4,
            )
        else:
            logger.info("No missing ids, all computation is done for models")
