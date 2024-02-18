from typing import Any, Optional
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from huggingface_hub import HfApi, DatasetCard, ModelCard
from pathlib import Path
from datasets import Dataset
from multiprocessing import Pool
from datetime import datetime
from .fs import save_json, save_partitioned_json, read_pickle, save_pickle, \
    file_or_dir_exists, get_last_partition_file, listdir

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


def _get_cached_ids(path: Path) -> Optional[Any]:
    if file_or_dir_exists(path):
        return read_pickle(path)


def _datetime_to_str(d: str | datetime | Any):
    if d is None or isinstance(d, str):
        return d
    elif isinstance(d, datetime):
        return d.isoformat()
    else:
        raise ValueError(f"Got unexpected type: {type(d)}")


def get_dataset_ids(path: Path) -> list[dict]:
    ids = _get_cached_ids(path)

    if ids is None:
        api = HfApi()
        datasets = api.list_datasets(sort="createdAt", direction=1)    # noqa
        ids = [
            dict(
                id=d.id,
                sha=d.sha,
                created_at=_datetime_to_str(d.created_at),
                last_modified=_datetime_to_str(d.last_modified),
                tags=d.tags,
                # Set them for default values for now and update it later
                metadata="{}",
                text=""
            )
            for d in tqdm(datasets, total=get_total_count('datasets'))
        ]
        save_pickle(ids, path=path)

    return ids


def get_model_ids(path: Path) -> list[dict]:
    ids = _get_cached_ids(path)

    if ids is None:
        api = HfApi()
        models = api.list_models(sort="createdAt", direction=1)    # noqa
        ids = [
            dict(
                id=m.id,
                sha=m.sha,
                created_at=_datetime_to_str(m.created_at),
                last_modified=_datetime_to_str(m.last_modified),
                tags=m.tags,
                pipeline_tag=m.pipeline_tag,
                library_name=m.library_name,
                # Set them for default values for now and update it later
                metadata="{}",
                text=""
            )
            for m in tqdm(models, total=get_total_count('models'))
        ]
        save_pickle(ids, path=path)

    ids = [i for i in ids if i['id'] != "userzyzz/test"]

    return ids


def _update_text_and_metadata(
    ids: list[dict],
    path: Path,
    card_type: str = 'dataset',
    partition: int = None,
    progress_bar: bool = False,
    padding_size: int = 4,
):
    card_data = []

    if progress_bar:
        ids = tqdm(ids)

    match card_type:
        case "dataset":
            card_cls = DatasetCard
        case "model":
            card_cls = ModelCard
        case _:
            raise ValueError(
                f"Expected model or dataset for card type but got {card_type}"
            )

    for i, data in enumerate(ids, start=1):
        try:
            card = card_cls.load(data["id"], ignore_metadata_errors=True)
        except BaseException as e:  # noqa: Catch all errors
            logger.debug(
                f"{card_cls.__name__} loading exception is caught: {data['id']}",
                exc_info=e
            )
            card_data.append(data)
            continue

        data['text'] = card.text

        try:
            data['metadata'] = json.dumps(card.data.to_dict())
        except BaseException as e:  # noqa: Catch all errors
            logger.warning(
                f"JSON Dumps Exception: is caught for {data['id']}", exc_info=e
            )
            card_data.append(data)
            continue

        card_data.append(data)

    if partition is not None:
        save_partitioned_json(
            card_data,
            path=path,
            as_binary=False,
            partition=partition,
            padding_size=padding_size,
        )
    else:
        save_json(card_data, path=path, as_binary=False)


def update_text_and_metadata_in_parallel(
    ids: list[dict],
    path: Path,
    card_type: str = 'dataset',
    partition_start: int = 0,
    cache_interval: int = 1000,
    n_jobs: int = 10,
    padding_size: int = 4,
):
    # Split the data into smaller chunks
    if len(ids) < (cache_interval * n_jobs):
        logger.info(f"Less data= {len(ids)}, disabling parallelization.")
        _update_text_and_metadata(
            ids,
            path=path,
            card_type=card_type,
            partition=partition_start,
            progress_bar=True,
            padding_size=padding_size,
        )
    else:
        id_chunks = [
            ids[i:i + cache_interval]
            for i in range(0, len(ids), cache_interval)
        ]

        with Pool(n_jobs) as p:
            p.starmap(
                _update_text_and_metadata,
                tqdm(
                    [
                        (chunk, path, card_type, partition, False, padding_size)
                        for partition, chunk in enumerate(id_chunks,
                                                          start=partition_start)
                    ],
                    total=len(id_chunks)
                )
            )


def find_missing_ids(
    ids: list[dict], path: Path, n_jobs: int = 10
) -> Optional[list[dict]]:
    if file_or_dir_exists(path):
        files = [f for f in listdir(path) if f.endswith(".json")]

        if len(files) > 0:
            ds = Dataset.from_json(str(path.joinpath("*.json")), num_proc=n_jobs)
            cache = set(ds['id'][:])
            logger.info("Computing missing ids..")
            # Compute remaining ids
            # Filter bad readme file (9gb) which crashes the process
            remaining = [
                i for i in ids if i['id'] not in cache and i['id'] != "userzyzz/test"
            ]
            # remaining = [i for i in all_ids if i not in cache]
            logger.info(f"There are {len(remaining)} IDs remaining")

            # ids = [i for i in ids if i['id'] in remaining]

            return remaining


def compute_partition_number(path: Path) -> int:
    file = get_last_partition_file(path)

    if file is not None:
        partition_start = int(
            file.replace(".json", "").split('-')[-1]
        ) + 1
        logger.info(
            f"Found previously cached files. Starting partition: {partition_start}"
        )
    else:
        partition_start = 0

    return partition_start


