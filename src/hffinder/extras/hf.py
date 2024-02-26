from typing import Any
from datasets import Dataset, load_dataset
from kedro.io import AbstractDataset
from .fs import mkdir
from pathlib import Path
from .. import TransformerModel, device
from transformers import AutoTokenizer, AutoModel

import logging
import importlib


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


class HFTransformer(AbstractDataset):
    def __init__(
        self,
        checkpoint: str,
        model_type: str = None,
        move_to_device: bool = True
    ):
        self.checkpoint = checkpoint

        if model_type is not None:
            try:
                self.model = importlib.import_module(model_type, package='transformers')
            except ImportError as e:
                logger.info(
                    f"Given model type={model_type} doesn't exist in transformers"
                )
                raise e
        else:
            self.model = AutoModel

        self.move_to_device = move_to_device

    def _load(self) -> TransformerModel:
        model = self.model.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        if self.move_to_device:
            model.to(device)

        return TransformerModel(model=model, tokenizer=tokenizer)

    def _save(self, data) -> None:
        raise NotImplementedError("Pretrained models don't support saving for now")

    def _describe(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "model_type": self.model,
            "move_to_device": self.move_to_device,
        }
