import json
import pickle
import logging
import os

from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)
# Find main project directory
BASE_PATH = Path(__file__).parent.parent.parent.parent.resolve()


def _save_object(obj: Any, path: Path, func: Callable, **kwargs):
    if not path.is_absolute():
        path = BASE_PATH.joinpath(path).resolve()

    func(obj, path, **kwargs)
    logger.info(f"Saved file: {path}")


def _read_object(path: Path, func: Callable, **kwargs):
    if not path.is_absolute():
        path = BASE_PATH.joinpath(path).resolve()

    data = func(path, **kwargs)
    logger.info(f"Read file: {path}")

    return data


def file_or_dir_exists(path: Path) -> bool:
    if not path.is_absolute():
        path = BASE_PATH.joinpath(path).resolve()

    return path.exists()


def mkdir(path: Path, parents: bool = False, exist_ok: bool = True) -> Path:
    if not path.is_absolute():
        path = BASE_PATH.joinpath(path).resolve()

    path.mkdir(parents=parents, exist_ok=exist_ok)

    if not file_or_dir_exists(path):
        raise ValueError('Directory creation failed')

    return path


def listdir(path: Path) -> list[str]:
    if not path.is_absolute():
        path = BASE_PATH.joinpath(path).resolve()

    return os.listdir(path)


def get_last_partition_file(path: Path) -> Optional[str]:

    files = listdir(path)
    filtered_files = [
        file for file in files
        if file.startswith("data-partition-") and file.endswith(".json")
    ]

    # Sort the filtered files and get the last one
    if filtered_files:
        logger.info("Found cached files. Searching for the last file")
        sorted_files = sorted(filtered_files)
        last_file = sorted_files[-1]
        logger.info(f"The last file name is: {last_file}")
        return last_file


def save_json(obj: dict | Any, path: Path, as_binary: bool = False):
    def _save(d: dict, _path: str, binary: bool = False):
        if binary:
            with open(_path, mode="wb") as f:
                f.write(json.dumps(d))  # noqa
        else:
            with open(_path, mode="w") as f:
                json.dump(d, f)

    _save_object(obj, path=path, func=_save, binary=as_binary)


def save_partitioned_json(
    obj: dict | Any,
    path: Path,
    as_binary: bool = False,
    partition: int = None,
    padding_size: int = 4,
):

    if path.name.endswith("json"):
        raise ValueError("Partition files should not contain file name")

    partition_suffix = f"{str(partition).zfill(padding_size)}.json"
    file_name = "data-partition" + "-" + partition_suffix
    path = path.joinpath(file_name)

    save_json(obj=obj, path=path, as_binary=as_binary)


def read_json(path: Path, as_binary: bool = False) -> dict | str:
    def _read(_path: Path, binary: bool = False):
        if binary:
            with open(_path, "rb") as f:
                data = json.loads(f.read())
        else:
            with open(_path, "r") as f:
                data = json.load(f)
        return data

    return _read_object(path, func=_read, binary=as_binary)


def save_pickle(obj: Any, path: Path):
    def _save(d: Any, _path: str):
        with open(_path, mode="wb") as f:
            pickle.dump(d, f)

    _save_object(obj, path=path, func=_save)


def read_pickle(path: Path) -> Any:
    def _read(_path: Path):
        with open(_path, "rb") as f:
            data = pickle.load(f)
        return data

    return _read_object(path, func=_read)

