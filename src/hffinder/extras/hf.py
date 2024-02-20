from typing import Any
from datasets import Dataset, load_dataset
from kedro.io import AbstractDataset
from .fs import mkdir
from pathlib import Path
import logging


logger = logging.getLogger(__file__)


class HFDataset(AbstractDataset):
    def __init__(
        self,
        filepath: str = None,
        dataset_name: str = None,
        credentials: dict[Any] = None,
        raise_error_if_push_hub_fails: bool = True
    ):
        self._filepath = str(mkdir(Path(filepath)))
        self._dataset_name = dataset_name
        self._credentials = credentials
        self._raise_error_if_push_hub_fails = raise_error_if_push_hub_fails

    def _load(self) -> Dataset:
        try:
            ds = Dataset.load_from_disk(self._filepath)
        except FileNotFoundError:
            ds = load_dataset(self._dataset_name)
        return ds

    def _save(self, data: Dataset) -> None:
        logger.info("Saving to local disk.")
        data.save_to_disk(self._filepath)

        logger.info("Saving to HuggingFace Hub")

        if isinstance(self._credentials, dict):
            token = self._credentials.get('write')
        elif isinstance(self._credentials, str):
            token = self._credentials
        else:
            token = None

        try:
            data.push_to_hub(self._dataset_name, token=token)
        except BaseException as e:
            logger.warning(f"Push to hub failed, provide credentials maybe?")
            if self._raise_error_if_push_hub_fails:
                raise e

    def _describe(self) -> dict[str, Any]:
        return {
            "file_path": self._filepath,
            "dataset_name": self._dataset_name,
            "raise_error_if_push_hub_fails": self._raise_error_if_push_hub_fails,
        }
