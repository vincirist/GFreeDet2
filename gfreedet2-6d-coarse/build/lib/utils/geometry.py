#!/usr/bin/env python3

import math
import numpy as np
import torch
from typing import Tuple, TypeVar
import scipy
from utils import geometry

from scipy.spatial.transform import Rotation

AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")


def transform_3d_points_numpy(trans: np.ndarray, points: np.ndarray) -> np.ndarray:

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return trans.dot(points_h.T)[:3, :].T


def transform_3d_points_torch(trans: torch.Tensor, points: torch.Tensor) -> torch.Tensor:

    assert trans.shape == (4, 4)
    assert points.shape[1] == 3
    assert trans.device == points.device
    assert trans.dtype == points.dtype
    points_h = torch.hstack([points, torch.ones((points.shape[0], 1), device=points.device)])
    return torch.matmul(trans, points_h.T)[:3, :].T


def gen_look_at_matrix(
    orig_camera_from_world: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
    return_camera_from_world: bool = True,
) -> np.ndarray:

    center_local = transform_points(orig_camera_from_world, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(np.array([0, 0, 1], dtype=center.dtype), z_dir_local)
    orig_world_from_camera = np.linalg.inv(orig_camera_from_world)

    world_from_aligned_camera = orig_world_from_camera.copy()
    world_from_aligned_camera[0:3, 0:3] = world_from_aligned_camera[0:3, 0:3] @ delta_r_local

    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    world_from_aligned_camera[0:3, 0:3] = world_from_aligned_camera[0:3, 0:3] @ z_local_rot

    if return_camera_from_world:
        return np.linalg.inv(world_from_aligned_camera)
    return world_from_aligned_camera


def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:

    return rotate_points(matrix, points) + matrix[..., :3, 3]


def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:

    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)


def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:

    a = geometry.normalized(a_orig)
    b = geometry.normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = np.eye(3, 3, dtype=a_orig.dtype) + v_mat + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))

    return rot


def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype)
    return res


def rotation_matrix_numpy(angle: float, direction: np.ndarray) -> np.ndarray:

    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = geometry.normalized(direction[:3])
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    return M


def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:

    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def normalized(v: AnyTensor, axis: int = -1, eps: float = 5.43e-20) -> AnyTensor:

    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d
