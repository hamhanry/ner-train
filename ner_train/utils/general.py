import collections
import random
import subprocess
import warnings
from collections.abc import KeysView
from functools import wraps
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.distributed import get_rank


def rank_zero(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_homogenous_iterable(iterable_obj, type_=int):
    return all(isinstance(el, type_) for el in iterable_obj)

def get_head_commit():
    file_path = str(Path(__file__).parent.absolute())

    try:
        commit = (
            subprocess.check_output(["git", "-C", file_path, "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError as ex:
        warnings.warn(f"Failed to get HEAD commit. Stack trace are shown below\n{ex}")
        commit = ""

    return commit


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_key(keys: KeysView, keys_to_search: Iterable):
    try:
        return set(keys).intersection(keys_to_search).pop()
    except KeyError as e:
        raise RuntimeError(
            f"No keys in {tuple(keys_to_search)} exists. "
            f"Available keys are: {keys}"
        ) from e


def default_worker_init_fn(worker_id: int, seed: int=None):
    if seed is None:
        seed = torch.initial_seed() % (2 ** worker_id)

    import os

    os.environ["OMP_NUM_THREADS"] = str(1)

    import cv2
    import torch

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    
    seed_all(seed)


def get_torch_generator(seed: int=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_version_string_as_integer(version: str):
    version_numbers = version.replace(".", "").split("-")[0]
    return int(version_numbers)
