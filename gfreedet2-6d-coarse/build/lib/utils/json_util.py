import collections.abc as abc
import dataclasses
import json
import re
from enum import Enum
import inspect
from typing import (
    Any,
    Dict,
    FrozenSet,
    get_type_hints,
    List,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np


JsonValue = Union[None, str, bool, int, float, List[Any], Dict[str, Any]]

T = TypeVar("T")


def merge(a: JsonValue, b: JsonValue) -> JsonValue:

    if isinstance(a, dict) and isinstance(b, dict):
        a = dict(a)
        for k in b:
            if k in a:
                a[k] = merge(a[k], b[k])
            else:
                a[k] = b[k]
        return a

    return b


def merge_at(obj: JsonValue, path: str, rhs: JsonValue) -> JsonValue:

    keys = [k for k in path.split(".") if k]
    while keys:
        key = keys.pop()
        rhs = {key: rhs}
    return merge(obj, rhs)


def parse_json_or_str(s: str) -> JsonValue:
    s = s.strip()
    if re.match("""null$|true$|false$|^[0-9."'[{-]""", s):
        return json.loads(s)
    else:
        return s


def merge_from_arg(obj: JsonValue, arg: str):

    if re.match("[^\"']+=", arg):
        path, rhs = arg.split("=", 1)
    else:
        path = ""
        rhs = arg
    rhs_obj = parse_json_or_str(rhs)
    return merge_at(obj, path, rhs_obj)


def from_any(x: Any) -> JsonValue:

    if type(x) in (str, bool, int, float, type(None)):
        return x

    if isinstance(x, Enum):
        return from_any(x.value)

    if isinstance(x, abc.Mapping):
        return {str_from_any(k): from_any(v) for (k, v) in x.items()}

    if dataclasses.is_dataclass(x):
        return from_any(dataclasses.asdict(x))

    if hasattr(x, "_asdict"):
        return from_any(x._asdict())

    if isinstance(x, (abc.Sequence, abc.Set)):
        return [from_any(x) for x in x]

    if isinstance(x, np.floating):
        return float(x)

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    raise ValueError(f"conversion of {x!r} to json not supported")


def str_from_any(x: Any) -> str:
    k = from_any(x)
    if not isinstance(k, str):
        raise TypeError(f"use of {x!r} as a json key is not supported")
    return k


def save_json(path: str, content: Any) -> None:

    with open(path, "w", encoding="utf-8") as f:
        content_json = from_any(content)
        json.dump(content_json, f, indent=2)


def load_json(path: str, keys_to_int: bool = False) -> Any:

    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

    with open(path, "r") as f:
        if keys_to_int:
            return json.load(f, object_hook=convert_keys_to_int)
        else:
            return json.load(f)


def _is_optional(t: Type) -> bool:

    return getattr(t, "__origin__", None) is Union and t.__args__[1:] == (type(None),)


def _is_sequence(t: Type) -> bool:

    origin = getattr(t, "__origin__", None)

    if origin in (list, set, frozenset, List, Set, FrozenSet):
        return True
    if origin in (tuple, Tuple) and t.__args__[1:] == (Ellipsis,):
        return True
    return False


def get_real_type(t):

    if hasattr(t, "__orig_bases__"):
        return t.__orig_bases__[0]
    else:
        return t.__origin__


def _append_path(path: str, key: Union[str, int]) -> str:

    if path:
        return f"{path[:-1]}.{key}'"
    else:
        return f" at '{key}'"


def validate_json(j: JsonValue, t: Type[T], at: str = "") -> T:

    return _validate_json(j, t, at)


def _validate_json(j, t, at: str):

    if _is_optional(t):
        if j is None:
            return None
        else:
            t = t.__args__[0]

    tname = getattr(t, "__name__", str(t))

    if t in (bool, int, float):
        if type(j) not in (bool, int, float) or t(j) != j:
            raise ValueError(f"{j!r} is not a valid value of type {tname}{at}")
        return t(j)

    if t is str:
        if not isinstance(j, str):
            raise ValueError(f"expected json string but got {j}{at}")
        return j

    if t in (None, type(None)):
        if j is not None:
            raise ValueError(f"expected json null but got {j}{at}")
        return None

    origin = getattr(t, "__origin__", None)

    if _is_sequence(t):
        return _json_to_seq(j, t, at)

    if origin in (tuple, Tuple):
        return _json_to_tuple(j, t, at)

    if origin in (dict, Dict):
        return _json_to_dict(j, t, at)

    if origin is Union:
        for tt in t.__args__:
            try:
                return _validate_json(j, tt, at)
            except ValueError:
                pass
        raise ValueError(f"expected json Union but got {j}{at}")

    if inspect.isclass(t) and issubclass(t, Enum):
        return t(j)

    if hasattr(t, "__annotations__"):
        return _json_to_struct(j, t, at)

    if "numpy" in str(t):
        if np.issubdtype(t, np.ndarray):
            return np.array(j)
        else:
            return t(j)

    raise TypeError(f"don't know how to validate {t}{at}")


def _json_to_struct(j, t, at):

    if not isinstance(j, dict):
        raise ValueError(f"expected json dict but got {j!r}{at}")

    types = get_type_hints(t)
    args = {}
    for k, v in j.items():
        at_k = _append_path(at, k)

        if k not in types:
            raise ValueError(f"unknown field{at_k} for {t.__name__}")

        args[k] = _validate_json(v, types[k], at_k)

    try:
        return t(**args)
    except TypeError as e:
        raise ValueError(str(e) + at) from e


def _json_to_tuple(j, tuple_type, at: str):

    elem_types = tuple_type.__args__

    if elem_types[1:] == [Ellipsis]:
        # sequence-like tuple
        return tuple(_json_to_seq(j, List[elem_types[0]], at))

    if not isinstance(j, (list, tuple)):
        raise ValueError(f"expected json list/tuple but got {j!r}{at}")

    n = len(elem_types)
    if len(j) != n:
        # fixed-size tuple
        raise ValueError(f"expected {n}-tuple but got {len(j)}{at}")

    return tuple(_validate_json(j[i], elem_types[i], _append_path(at, i)) for i in range(n))


def _json_to_seq(j, seq_type, at):

    if not isinstance(j, list):
        raise ValueError(f"expected json list but got {j!r} for {seq_type}")
    elem_type = seq_type.__args__[0]
    elems = [_validate_json(elem, elem_type, _append_path(at, i)) for (i, elem) in enumerate(j)]

    return get_real_type(seq_type)(elems)


def _json_to_dict(j, t, at: str):

    kt, vt = t.__args__
    if not isinstance(j, dict):
        raise ValueError(f"expected json object but got {j!r} for {t}{at}")

    d = {}
    for k, v in j.items():
        at_k = _append_path(at, k)
        d[_validate_json(k, kt, at_k)] = _validate_json(v, vt, at_k)
    return d
