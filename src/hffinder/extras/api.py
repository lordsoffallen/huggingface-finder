from typing import Any, Optional
from bs4 import BeautifulSoup
from kedro.io.core import AbstractDataset
from tqdm.auto import tqdm
from huggingface_hub import HfApi, DatasetCard
from huggingface_hub.utils import disable_progress_bars
from pathlib import Path
from datasets import Dataset
from multiprocessing import Pool
from .fs import save_json, save_partitioned_json, read_pickle, save_pickle, \
    file_or_dir_exists, mkdir, get_last_partition_file

import requests
import logging
import json


logger = logging.getLogger(__name__)
BASE_URL = "https://huggingface.co"


def check_if_http_call_successful(response: requests.Response):
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Failed to retrieve the webpage={response.url}. "
            f"Status code: {response.status_code}"
        )


def get_total_count(_type: str):
    url = BASE_URL + '/' + _type
    response = requests.get(url)
    check_if_http_call_successful(response)

    content = BeautifulSoup(response.content)
    # Find the header and extract the next element where the
    # total count dataset/model lies.
    count = content.find("h1").find_next().text

    try:
        count = int(count.replace(",", ""))
    except ValueError:
        raise ValueError(f"Could not parse to int. Extracted value is={count}, url={url}")

    return count


def update_text_and_metadata(
    ids: list[dict],
    path: Path,
    partition: int = None,
    progress_bar: bool = False,
    padding_size: int = 4,
):
    data = []

    if progress_bar:
        ids = tqdm(ids)

    for i, dataset in enumerate(ids, start=1):
        try:
            card = DatasetCard.load(dataset["id"], ignore_metadata_errors=True)
        except BaseException as e:  # noqa: Catch all errors
            logger.debug(
                f"DatasetCard Loading Exception is caught for {dataset['id']}",
                exc_info=e
            )
            data.append(dataset)
            continue

        dataset['text'] = card.text

        try:
            dataset['metadata'] = json.dumps(card.data.to_dict())
        except BaseException as e:  # noqa: Catch all errors
            logger.warning(
                f"JSON Dumps Exception: is caught for {dataset['id']}", exc_info=e
            )
            data.append(dataset)
            continue

        data.append(dataset)

    if partition is not None:
        save_partitioned_json(
            data,
            path=path,
            as_binary=False,
            partition=partition,
            padding_size=padding_size,
        )
    else:
        save_json(data, path=path, as_binary=False)


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

    @property
    def api(self):
        return HfApi()

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
        ids = self.get_dataset_ids()

        if file_or_dir_exists(self.datasets_cachepath):
            ds = Dataset.from_json(
                str(self.datasets_cachepath.joinpath("*.json")),
                num_proc=self.n_jobs,
            )
            cache = ds['id'][:]
            # Compute remaining ids
            all_ids = [i['id'] for i in ids]
            remaining = [i for i in all_ids if i not in cache]
            logger.info(f"There are {len(remaining)} IDs remaining")

            ids = [i for i in ids if i['id'] in remaining]

        file = get_last_partition_file(self.datasets_cachepath)
        if file is not None:
            partition_start = int(
                file.replace(".json", "").split('-')[-1]
            ) + 1
            logger.info(
                f"Found previously cached files. Starting partition: {partition_start}"
            )
        else:
            partition_start = 0

        self.update_text_and_metadata_in_parallel(ids, partition_start)

    def _save(self, data) -> None:
        raise NotImplementedError

    def get_dataset_ids(self) -> list[dict]:
        if file_or_dir_exists(self.dataset_ids_path):
            ids = read_pickle(self.dataset_ids_path)
        else:
            datasets = self.api.list_datasets(sort="createdAt", direction=1)    # noqa
            ids = [
                dict(
                    id=d.id,
                    sha=d.sha,
                    created_at=d.created_at.isoformat(),
                    last_modified=d.last_modified.isoformat(),
                    tags=d.tags,
                    # Set them for default values for now and update it later
                    metadata="{}",
                    text=""
                )
                for d in tqdm(datasets, total=get_total_count('datasets'))
            ]
            save_pickle(ids, path=self.dataset_ids_path)

        return ids

    def update_text_and_metadata_in_parallel(
        self,
        ids: list[dict],
        partition_start: int = 0,
    ):
        # Split the data into smaller chunks
        if len(ids) < (self.cache_interval * self.n_jobs):
            logger.info(f"Less data= {len(ids)}, disabling parallelization.")
            update_text_and_metadata(
                ids,
                self.datasets_cachepath,
                partition=partition_start,
                progress_bar=True
            )
        else:
            id_chunks = [
                ids[i:i + self.cache_interval]
                for i in range(0, len(ids), self.cache_interval)
            ]

            with Pool(self.n_jobs) as p:
                p.starmap(
                    update_text_and_metadata,
                    tqdm(
                        [
                            # (chunk, path, partition, total_partition_size)
                            (chunk, self.datasets_cachepath, partition)
                            for partition, chunk in enumerate(id_chunks,
                                                              start=partition_start)
                        ],
                        total=len(id_chunks)
                    )
                )
