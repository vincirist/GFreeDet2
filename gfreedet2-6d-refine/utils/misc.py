# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

import os
import math
from bop_toolkit_lib import inout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt

import torch
import time
from utils import structs, transform3d, logging_util


logger = logging_util.get_logger(__name__)


class Timer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_time = None

    def start(self):
        if self.enabled:
            self.start_time = time.time()

    def elapsed(self, msg="Elapsed") -> Optional[float]:
        if self.enabled:
            elapsed = time.time() - self.start_time
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

        f = -np.array(pt)  # Forward direction.
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0])  # Up direction.
        s = np.cross(f, u)  # Side direction.
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis.
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)  # Recompute up.
        R = np.array([[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]])

        # Convert from OpenGL to OpenCV coordinate system.
        R_yz_flip = transform3d.rotation_matrix_numpy(math.pi, np.array([1, 0, 0]))[:3, :3]
        R = R_yz_flip.dot(R)

        # Translation vector.
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({"R": R, "t": t})

    return views, pts_level


def get_rigid_matrix(trans: structs.RigidTransform) -> np.ndarray:

    matrix = np.eye(4)
    matrix[:3, :3] = trans.R
    matrix[:3, 3:] = trans.t
    return matrix


def get_intrinsic_matrix(cam: structs.CameraModel) -> np.ndarray:

    return np.array(
        [
            [cam.f[0], 0.0, cam.c[0]],
            [0.0, cam.f[1], cam.c[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def slugify(string: str) -> str:

    return string.strip("/").replace("/", "-").replace(" ", "-").replace(".", "-")


def ensure_three_channels(im: np.ndarray) -> np.ndarray:

    if im.ndim == 3:
        return im
    elif im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        return np.dstack([im, im, im])
    else:
        raise ValueError("Unknown image format.")


def check_var(
    var: Any,  # pyre-ignore
    dtype: Optional[Type[Any]] = None,  # pyre-ignore
    array_shape: Optional[List[int]] = None,
    array_dtype: Optional[Union[npt.DTypeLike, torch.dtype]] = None,  # pyre-ignore
) -> bool:

    valid = True
    if dtype is not None:
        valid = valid and isinstance(var, dtype)
    if array_shape is not None:
        valid = (
            valid
            and len(var.shape) == len(array_shape)
            and all(var.shape[i] == array_shape[i] or array_shape[i] == -1 for i in range(len(array_shape)))
        )
    if array_dtype is not None:
        valid = valid and var.dtype == array_dtype
    return valid


def save_per_frame_prediction(
    scene_ids: np.ndarray,
    im_ids: np.ndarray,
    obj_ids: np.ndarray,
    run_times: np.ndarray,
    scores: np.ndarray,
    save_path: Path,
    obj_xywh_boxes: Optional[np.ndarray] = None,
    obj_frame_ids: Optional[np.ndarray] = None,
    poses_cam_from_model: Optional[np.ndarray] = None,
    debug: bool = False,
) -> None:

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if obj_xywh_boxes is not None:
        assert poses_cam_from_model is None
        is_2d_detection_results = True
    else:
        assert poses_cam_from_model is not None
        is_2d_detection_results = False

    if len(np.unique(scene_ids)) > 1:
        obj_scene_ids = scene_ids[obj_frame_ids]
        obj_im_ids = im_ids[obj_frame_ids]
        run_times = run_times[obj_frame_ids]
    else:
        obj_scene_ids = np.repeat(scene_ids, len(scores))
        obj_im_ids = np.repeat(im_ids, len(scores))
        run_times = np.repeat(run_times, len(scores))

    predictions = []
    for idx, (scene_id, im_id, obj_id, run_time, score) in enumerate(
        zip(obj_scene_ids, obj_im_ids, obj_ids, run_times, scores)
    ):
        pred = {
            "scene_id": int(scene_id),
            "time": float(run_time),
            "score": float(score),
        }

        if is_2d_detection_results:
            pred["image_id"] = int(im_id)
            pred["category_id"] = int(obj_id)
            pred["bbox"] = obj_xywh_boxes[idx].tolist()
        else:
            pred["im_id"] = int(im_id)
            pred["obj_id"] = int(obj_id)
            pose = poses_cam_from_model[idx]
            assert pose.shape == (4, 4)
            pred["R"] = pose[:3, :3].flatten()
            pred["t"] = pose[:3, 3]
        predictions.append(pred)

    if is_2d_detection_results:
        inout.save_json(save_path, predictions)
    else:
        inout.save_bop_results(save_path, predictions)
    if debug:
        logger.info(f"Saved per-frame predictions to {save_path}")


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


def check_np_array_valid(arr: np.ndarray) -> bool:

    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    ok = not (has_nan or has_inf)
    return ok
