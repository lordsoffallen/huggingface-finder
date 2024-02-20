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


class HFDatasetScrapper(AbstractDataset):
    def __init__(self, cachepath: str, cache_interval: int = 1000, n_jobs: int = 10):
        self.cache_interval = cache_interval
        self.cachepath = mkdir(Path(cachepath))
        self.ids_path = self.cachepath.joinpath("ids.pkl")
        self.n_jobs = n_jobs

        disable_progress_bars()     # no tqdm in api call

    def _describe(self) -> dict[str, Any]:
        return {
            "cachepath": self.cachepath,
            "cache_interval": self.cache_interval,
            "ids_path": self.ids_path,
            "n_jobs": self.n_jobs,
        }

    def _load(self):
        # Populate cache
        self.get_datasets()

        return Dataset.from_json(str(self.cachepath.joinpath("*.json")))

    def _save(self, data) -> None:
        raise NotImplementedError

    def get_datasets(self):
        ids = get_dataset_ids(self.ids_path)
        missing_ids = find_missing_ids(ids, self.cachepath)

        if missing_ids is not None:
            ids = missing_ids

        if len(ids) > 0:
            partition_start = compute_partition_number(self.cachepath)

            update_text_and_metadata_in_parallel(
                ids,
                path=self.cachepath,
                card_type="dataset",
                partition_start=partition_start,
                cache_interval=self.cache_interval,
                n_jobs=self.n_jobs,
                padding_size=4,
            )
        else:
            logger.info("No missing ids, all computation is done for datasets")


class HFModelScrapper(AbstractDataset):
    def __init__(self, cachepath: str, cache_interval: int = 1000, n_jobs: int = 10):
        self.cache_interval = cache_interval
        self.cachepath = mkdir(Path(cachepath))
        self.ids_path = self.cachepath.joinpath("ids.pkl")
        self.n_jobs = n_jobs

        disable_progress_bars()     # no tqdm in api call

    def _describe(self) -> dict[str, Any]:
        return {
            "cachepath": self.cachepath,
            "cache_interval": self.cache_interval,
            "ids_path": self.ids_path,
            "n_jobs": self.n_jobs,
        }

    def _load(self):
        # Populate cache
        self.get_models()

        return Dataset.from_json(str(self.cachepath.joinpath("*.json")))

    def _save(self, data) -> None:
        raise NotImplementedError

    def get_models(self):
        ids = get_model_ids(self.ids_path)
        missing_ids = find_missing_ids(ids, self.cachepath)

        if missing_ids is not None:
            ids = missing_ids

        if len(ids) > 0:
            partition_start = compute_partition_number(self.cachepath)

            update_text_and_metadata_in_parallel(
                ids,
                path=self.cachepath,
                card_type="model",
                partition_start=partition_start,
                cache_interval=self.cache_interval,
                n_jobs=self.n_jobs,
                padding_size=4,
            )
        else:
            logger.info("No missing ids, all computation is done for models")
