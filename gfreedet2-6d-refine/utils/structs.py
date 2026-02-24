# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3


from __future__ import annotations
import itertools
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Any,
)

import numpy as np
import torch

import abc
import math

from utils import torch_helpers, transform3d

ArrayData = Union[np.ndarray, torch.Tensor]
Color = Tuple[float, float, float]


class RigidTransform(NamedTuple):

    R: ArrayData
    t: ArrayData


ObjectPose = RigidTransform


class ObjectAnnotation(NamedTuple):

    dataset: str
    lid: int
    uid: Optional[str] = None
    pose: Optional[ObjectPose] = None
    masks_modal: Optional[ArrayData] = None
    masks_amodal: Optional[ArrayData] = None
    boxes_modal: Optional[ArrayData] = None
    boxes_amodal: Optional[ArrayData] = None
    active: Optional[bool] = None
    visibilities: Optional[ArrayData] = None
    score: Optional[float] = None


class ImageObjectAnnotations(NamedTuple):

    labels: Optional[ArrayData] = None
    poses: Optional[ArrayData] = None
    masks: Optional[ArrayData] = None
    boxes: Optional[ArrayData] = None
    active_labels: Optional[ArrayData] = None
    visibilities: Optional[ArrayData] = None


class SceneObservation(NamedTuple):

    scene_id: int
    im_id: int
    image: Optional[ArrayData] = None
    depth_image: Optional[ArrayData] = None
    camera: Optional[List[List[CameraModel]]] = None
    objects_anno: Optional[List[List[ObjectAnnotation]]] = None
    time: Optional[float] = None


class AlignedBox2f:

    def __init__(self, left: float, top: float, right: float, bottom: float):

        self._left: float = left
        self._top: float = top
        self._right: float = right
        self._bottom: float = bottom

    def __repr__(self):
        return f"AlignedBox2f(left: {self._left}, top: {self._top}, right: {self._right}, bottom: {self._bottom})"

    @property
    def left(self) -> float:

        return self._left

    @property
    def top(self) -> float:

        return self._top

    @property
    def right(self) -> float:

        return self._right

    @property
    def bottom(self) -> float:

        return self._bottom

    @property
    def width(self) -> float:

        return self.right - self.left

    @property
    def height(self) -> float:

        return self.bottom - self.top

    def pad(self, width: float, height: float) -> AlignedBox2f:

        return AlignedBox2f(
            self.left - width,
            self.top - height,
            self.right + width,
            self.bottom + height,
        )

    def array_ltrb(self) -> np.ndarray:

        return np.array([self.left, self.top, self.right, self.bottom])

    def array_ltwh(self) -> np.ndarray:

        return np.array([self.left, self.top, self.width, self.height])

    def int_array_ltrb(self) -> np.ndarray:

        return self.array_ltrb().astype(int)

    def int_array_ltwh(self) -> np.ndarray:

        return self.array_ltwh().astype(int)

    def round(self) -> AlignedBox2f:

        return AlignedBox2f(
            np.round(self.left),
            np.round(self.top),
            np.round(self.right),
            np.round(self.bottom),
        )

    def clip(self, boundary: AlignedBox2f) -> AlignedBox2f:

        return AlignedBox2f(
            min(max(self.left, boundary.left), boundary.right),
            min(max(self.top, boundary.top), boundary.bottom),
            min(max(self.right, boundary.left), boundary.right),
            min(max(self.bottom, boundary.top), boundary.bottom),
        )


class CameraModel(abc.ABC):

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    T_world_from_eye: np.ndarray

    _zmin: Optional[float]
    _max_angle: Optional[float]

    def __init__(
        self,
        width,
        height,
        f,
        c,
        T_world_from_eye=None,
        serial="",
    ):
        self.width = width
        self.height = height
        self.serial = serial

        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if T_world_from_eye is None:
            self.T_world_from_eye = np.eye(4)
        else:
            self.T_world_from_eye = transform3d.as_4x4(T_world_from_eye, copy=True)
            if np.abs((self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3] - np.eye(3)).max() >= 1.0e-5:
                info_str = "camera T_world_from_eye must be a rigid transform\n"
                info_str = info_str + "T\n{}\n".format(self.T_world_from_eye.T)
                info_str = info_str + "(T*T_t - I).max()\n{}\n".format(
                    np.abs((self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3] - np.eye(3)).max()
                )
                raise ValueError(info_str)

        self._zmin = None
        self._max_angle = None

    def __repr__(self):
        return f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"

    def to_json(self):
        js = {}
        js["ImageSizeX"] = self.width
        js["ImageSizeY"] = self.height
        js["T_WorldFromCamera"] = self.T_world_from_eye.tolist()

        js["ModelViewMatrix"] = np.linalg.inv(self.T_world_from_eye).tolist()

        js["fx"], js["fy"] = np.asarray(self.f).tolist()
        js["cx"], js["cy"] = np.asarray(self.c).tolist()

        return js

    def is_fisheye(self):
        raise NotImplementedError("You should implement this for specific camera")

    def copy(
        self,
        T_world_from_eye=None,
    ):

        return self.crop(
            0,
            0,
            self.width,
            self.height,
            T_world_from_eye=T_world_from_eye,
        )

    def compute_zmin(self):
        corners = np.array([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]]) - 0.5
        self._zmin = self.window_to_eye(corners)[:, 2].min()
        self._max_angle = np.arccos(self._zmin)

    def zmin(self):
        if self._zmin is None:
            self.compute_zmin()
        return self._zmin

    def max_angle(self):
        if self._max_angle is None:
            self.compute_zmin()
        return self._max_angle

    def world_to_window(self, v):

        return self.eye_to_window(self.world_to_eye(v))

    def world_to_window3(self, v):

        return self.eye_to_window3(self.world_to_eye(v))

    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):

        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = transform3d.normalized(v, axis=-1)
        return v

    @staticmethod
    def project3(v):

        x, y, z = np.moveaxis(v, -1, 0)
        return np.stack([x / z, y / z, z], axis=-1)

    @staticmethod
    def unproject3(p):

        x, y, z = np.moveaxis(p, -1, 0)
        return np.stack((x * z, y * z, z), axis=-1)

    def pos(self):

        return self.T_world_from_eye[:3, 3]

    def orient(self):

        return self.T_world_from_eye[:3, :3]

    def window_to_world_ray(self, w):

        v = transform3d.rotate_points(self.T_world_from_eye, self.window_to_eye(w))
        o = np.broadcast_to(self.pos(), v.shape)
        return (o, v)

    def window_to_world3(self, w):

        return self.eye_to_world(self.window_to_eye3(w))

    def world_visible(self, v):

        return self.eye_visible(self.world_to_eye(v))

    def world_to_eye(self, v):

        return transform3d.rotate_points(self.T_world_from_eye.T, v - self.T_world_from_eye[:3, 3])

    def eye_to_world(self, v):

        return transform3d.transform_points(self.T_world_from_eye, v)

    def eye_to_window(self, v):

        p = self.project(v)
        return p * self.f + self.c

    def window_to_eye(self, w):

        q = (np.asarray(w) - self.c) / self.f
        return self.unproject(q)

    def eye_to_window3(self, v):

        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p

    def window_to_eye3(self, w):

        # assert self.undistort is not None
        temp = np.array(w, dtype=np.float64)
        temp[..., :2] -= self.c
        temp[..., :2] /= self.f
        # temp[..., :2] = self.undistort.evaluate(temp[..., :2])
        return self.unproject3(temp)

    def visible(self, v):

        return self.eye_visible(self.world_to_eye(v))

    def eye_visible(self, v):

        v = transform3d.normalized(v)
        w = self.eye_to_window(v)
        return (v[..., 2] >= self.zmin()) & self.w_visible(w)

    def w_visible(self, w, *, margin=0):

        x, y = np.moveaxis(w, -1, 0)
        x0 = -margin - 0.5
        y0 = -margin - 0.5
        x1 = self.width + margin - 0.5
        y1 = self.height + margin - 0.5
        return (x > x0) & (x < x1) & (y >= y0) & (y < y1)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        T_world_from_eye=None,
        serial=None,
    ):

        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.T_world_from_eye if T_world_from_eye is None else T_world_from_eye,
            self.serial if serial is None else serial,
        )

    def subrect(self, transform, width, height, bypass_fit_undistort_coeffs=False):

        f = np.diag(transform[:-2, :-2])
        c = transform[:-2, 2]
        offdiag = np.diag(np.flip(transform[..., :2, :2], -1))
        if not np.all(offdiag == 0.0):
            raise NotImplementedError("transforms with rotation not yet supported")
        cam = type(self)(
            width,
            height,
            self.f * f,
            self.c * f + c,
            self.distort_coeffs,
            self.undistort_coeffs,
            self.T_world_from_eye,
            self.serial,
            bypass_fit_undistort_coeffs,
        )
        return cam


class PinholePlaneCameraModel(CameraModel):
    model_fov_limit = 50 * (math.pi / 180)

    def uv_to_window_matrix(self):

        return np.array([[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]])

    def is_fisheye(self):
        return False


class Collection:

    def __init__(self, **kwargs: Any) -> None:

        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:  # pyre-ignore
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Collection!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:  # pyre-ignore

        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, "Adding a field {} of length {} to a Collection of length {}".format(
                name, data_len, len(self)
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:

        return name in self._fields

    def remove(self, name: str) -> None:

        del self._fields[name]

    def get(self, name: str) -> Any:  # pyre-ignore

        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:

        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Collection":

        ret = Collection()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(
        self,
        # pyre-fixme[24]: Generic type `slice` expects 3 type parameters.
        item: Union[int, slice, torch.Tensor, torch.BoolTensor],
    ) -> "Collection":

        if type(item) is int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Collection index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Collection()
        for k, v in self._fields.items():
            if isinstance(v, List):
                ret.set(k, [v[i] for i in item])
            else:
                ret.set(k, v[item])

        return ret

    def __len__(self) -> int:
        for v in self._fields.values():

            return v.__len__()
        return 0

    def __iter__(self) -> None:
        raise NotImplementedError("`Collection` object is not iterable!")

    def tensors_to_arrays(self) -> None:
        for field in self._fields.keys():
            if torch.is_tensor(self._fields[field]):
                self._fields[field] = torch_helpers.tensor_to_array(self._fields[field])

    def arrays_to_tensors(self) -> None:
        for field in self._fields.keys():
            if isinstance(self._fields[field], np.ndarray):
                self._fields[field] = torch_helpers.array_to_tensor(self._fields[field])

    @staticmethod
    def cat(collection_list: List["Collection"]) -> "Collection":

        assert all(isinstance(i, Collection) for i in collection_list)
        assert len(collection_list) > 0
        if len(collection_list) == 1:
            return collection_list[0]

        ret = Collection()
        for k in collection_list[0]._fields.keys():
            values = [i.get(k) for i in collection_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_samples={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__  # pyre-ignore


class BatchedData:

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)
