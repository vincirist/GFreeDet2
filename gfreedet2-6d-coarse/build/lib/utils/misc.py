#!/usr/bin/env python3

import os
import dataclasses
from dataclasses import asdict
import math
import time
import cv2
import uuid

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping
from collections import namedtuple

import numpy as np
from PIL import Image
import torch

from utils import geometry, logging_util

import utils.structs as structs
from utils.structs import AlignedBox2f, CameraModel, PinholePlaneCameraModel

from utils.geometry import transform_3d_points_numpy, gen_look_at_matrix

logger: logging_util.Logger = logging_util.get_logger_v2(__name__)


class Timer:
    def __init__(self, enabled: bool = True, log=True) -> None:
        self.enabled = enabled
        self.start_time = None
        self.log = log

    def start(self):
        if self.enabled:
            self.start_time = time.perf_counter()

    def elapsed(self, msg="Elapsed") -> Optional[float]:
        if self.enabled:
            elapsed = time.perf_counter() - self.start_time

            if self.log:
                logger.info(f"{msg}: {elapsed:.5f}s")
            return elapsed
        else:
            return None


def fibonacci_sampling(n_pts: int, radius: float = 1.0) -> List[Tuple[float, float, float]]:

    assert n_pts % 2 == 1

    n_pts_half = int(n_pts / 2)

    phi = (math.sqrt(5.0) + 1.0) / 2.0
    phi_inv = phi - 1.0
    ga = 2.0 * math.pi * phi_inv

    pts = []
    for i in range(-n_pts_half, n_pts_half + 1):
        lat = math.asin((2 * i) / float(2 * n_pts_half + 1))
        lon = (ga * i) % (2 * math.pi)

        s = math.cos(lat) * radius
        x, y, z = math.cos(lon) * s, math.sin(lon) * s, math.tan(lat) * s
        pts.append([x, y, z])

    return pts


def sample_views(
    min_n_views: int,
    radius: float = 1.0,
    azimuth_range: Tuple[float, float] = (0, 2 * math.pi),
    elev_range: Tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi),
    mode: str = "fibonacci",
) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:

    if mode == "fibonacci":
        n_views = min_n_views
        if n_views % 2 != 1:
            n_views += 1

        pts = fibonacci_sampling(n_views, radius=radius)
        pts_level = [0 for _ in range(len(pts))]
    else:
        raise ValueError("Unknown view sampling mode.")

    views = []
    for pt in pts:

        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and elev_range[0] <= elev <= elev_range[1]):
            continue

        f = -np.array(pt)
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0])
        s = np.cross(f, u)
        if np.count_nonzero(s) == 0:

            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.array([[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]])

        R_yz_flip = geometry.rotation_matrix_numpy(math.pi, np.array([1, 0, 0]))[:3, :3]
        R = R_yz_flip.dot(R)

        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({"R": R, "t": t})

    return views, pts_level


def calc_crop_box(
    box: AlignedBox2f,
    box_scaling_factor: float = 1.0,
    make_square: bool = False,
) -> AlignedBox2f:

    crop_box_width = box.width * box_scaling_factor
    crop_box_height = box.height * box_scaling_factor

    if make_square:
        crop_box_side = max(crop_box_width, crop_box_height)
        crop_box_width = crop_box_side
        crop_box_height = crop_box_side

    x_pad = 0.5 * (crop_box_width - box.width)
    y_pad = 0.5 * (crop_box_height - box.height)

    return AlignedBox2f(
        left=box.left - x_pad,
        top=box.top - y_pad,
        right=box.right + x_pad,
        bottom=box.bottom + y_pad,
    )


def get_adaptive_crop_rel_pad(box_center, image_size, dmin=0.0, dmax=1.0, threshold=0.3, alpha=15.0):
    x_b, y_b = box_center
    H, W = image_size

    x_c, y_c = W / 2, H / 2
    dx = (x_b - x_c) / (W / 2)
    dy = (y_b - y_c) / (H / 2)
    norm_dist = np.sqrt(dx**2 + dy**2) / np.sqrt(2)
    norm_dist = np.clip(norm_dist, 0.0, 1.0)

    def sigmoid(x):
        return 1 / (1 + np.exp(-alpha * (x - threshold)))

    f_d = sigmoid(norm_dist)
    f_min = sigmoid(0)
    f_max = sigmoid(1)
    f_norm = (f_d - f_min) / (f_max - f_min)

    score = dmin + (dmax - dmin) * f_norm
    return score


def unproject_fisheye_equidistant(u, v, f, cx, cy):

    x = u - cx
    y = v - cy
    r = np.sqrt(x**2 + y**2)

    if r < 1e-8:
        return [0, 0, 1.0]

    theta = r / f
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return [x / r * sin_theta, y / r * sin_theta, cos_theta]


def construct_crop_camera(
    box: AlignedBox2f,
    camera_model_c2w: CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float,
    correct_fisheye: bool = True,
) -> CameraModel:

    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    if camera_model_c2w.is_fisheye() and correct_fisheye:
        box_corners_in_c = np.array(
            [
                unproject_fisheye_equidistant(box.left, box.top, f, cx, cy),
                unproject_fisheye_equidistant(box.right, box.top, f, cx, cy),
                unproject_fisheye_equidistant(box.left, box.bottom, f, cx, cy),
                unproject_fisheye_equidistant(box.right, box.bottom, f, cx, cy),
            ]
        )
    else:
        box_corners_in_c = np.array(
            [
                [box.left - cx, box.top - cy, f],
                [box.right - cx, box.top - cy, f],
                [box.left - cx, box.bottom - cy, f],
                [box.right - cx, box.bottom - cy, f],
            ]
        )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[:3, 0]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = gen_look_at_matrix(trans_w2c, centroid_in_w)

    centroid_in_vc = transform_3d_points_numpy(trans_w2vc, np.expand_dims(centroid_in_w, axis=0)).squeeze()

    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    fx_fy = fx_fy_orig * cx_cy / extent_2d

    return PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )


def calc_2d_box(
    xs: torch.Tensor,
    ys: torch.Tensor,
    im_size: Optional[torch.Tensor] = None,
    clip: bool = False,
) -> torch.Tensor:

    if len(xs) == 0 or len(ys) == 0:
        return torch.Tensor([0.0, 0.0, 0.0, 0.0])

    box_min = torch.as_tensor([xs.min(), ys.min()])
    box_max = torch.as_tensor([xs.max(), ys.max()])

    return torch.hstack([box_min, box_max])


def get_rigid_matrix(trans: structs.RigidTransform) -> np.ndarray:

    matrix = np.eye(4)
    matrix[:3, :3] = trans.R
    matrix[:3, 3:] = trans.t
    return matrix


def get_intrinsic_matrix(cam: CameraModel) -> np.ndarray:

    return np.array(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:

    if interpolation is None:
        interpolation = cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


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


def is_dictlike(obj: Any) -> bool:

    return dataclasses.is_dataclass(obj) or hasattr(obj, "_asdict") or isinstance(obj, Mapping)


def chw_to_hwc(data: np.ndarray) -> np.ndarray:

    return np.transpose(data, (1, 2, 0))


def slugify(string: str) -> str:

    return string.strip("/").replace("/", "-").replace(" ", "-").replace(".", "-")


def crop_image(image: np.ndarray, crop_box: AlignedBox2f) -> np.ndarray:

    return image[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]


def ensure_three_channels(im: np.ndarray) -> np.ndarray:

    if im.ndim == 3:
        return im
    elif im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        return np.dstack([im, im, im])
    else:
        raise ValueError("Unknown image format.")


def warp_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
    factor_to_downsample: int = 1,
) -> np.ndarray:

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def warp_depth_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:

    depth_image = np.array(src_depth_image)

    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):

        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)

        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )


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


def set_torch_cuda_arch_list(all_devices: bool = True):

    arches = []
    if all_devices:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arches.append(f"{major}.{minor}")
    else:
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        arches.append(f"{major}.{minor}")

    arch_list = ";".join(sorted(set(arches)))
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

    print(f"TORCH_CUDA_ARCH_LIST set to: {arch_list}")
    return arch_list
