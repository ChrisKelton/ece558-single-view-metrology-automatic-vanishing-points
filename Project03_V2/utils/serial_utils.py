__all__ = ["save_pickle", "load_pickle"]
import pickle
from pathlib import Path
from typing import Any, Union


def save_pickle(obj: Any, pickle_path: Union[str, Path]) -> int:
    if isinstance(pickle_path, Path):
        pickle_path = str(pickle_path.absolute())

    with open(pickle_path, "wb") as p:
        pickle.dump(obj, p)

    return 0


def load_pickle(pickle_path: Union[str, Path]) -> Any:
    if isinstance(pickle_path, Path):
        pickle_path = str(pickle_path.absolute())

    with open(pickle_path, "rb") as p:
        obj = pickle.load(p)

    return obj
