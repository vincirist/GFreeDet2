# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

import dataclasses
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

import torch
from utils import logging_util


logger = logging_util.get_logger(__name__)


def is_dictlike(obj: Any) -> bool:

    return dataclasses.is_dataclass(obj) or hasattr(obj, "_asdict") or isinstance(obj, Mapping)


def map_fields(func, obj, only_type=object):

    if is_dictlike(obj):
        ty = type(obj)
        if isinstance(obj, Mapping):
            return ty((k, map_fields(func, v, only_type)) for (k, v) in obj.items())
        else:
            return ty(**{k: map_fields(func, v, only_type) for (k, v) in asdict(obj).items()})
    elif isinstance(obj, tuple):
        return tuple(map_fields(func, v, only_type) for v in obj)
    elif isinstance(obj, list):
        return [map_fields(func, v, only_type) for v in obj]
    elif isinstance(obj, only_type):
        return func(obj)
    else:
        return obj


def array_to_tensor(array: np.ndarray, make_array_writeable: bool = True) -> torch.Tensor:

    if not array.flags.writeable:
        if make_array_writeable and array.flags.owndata:
            array.setflags(write=True)
        else:
            array = np.array(array)
    return torch.from_numpy(array)


def arrays_to_tensors(data: Any) -> Any:

    return map_fields(lambda x: array_to_tensor(x), data, only_type=np.ndarray)


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:

    return tensor.detach().cpu().numpy()


def tensors_to_arrays(data: Any) -> Any:

    return map_fields(lambda x: tensor_to_array(x), data, only_type=torch.Tensor)
