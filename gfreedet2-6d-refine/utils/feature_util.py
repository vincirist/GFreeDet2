# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Tuple

import kornia
import torch

from utils import logging_util, misc, transform3d
from utils.structs import PinholePlaneCameraModel

from utils import dinov2_util

logger = logging_util.get_logger(__name__)


def make_feature_extractor(model_name: str) -> torch.nn.Module:
    if model_name.startswith("dinov2_"):
        return dinov2_util.DinoFeatureExtractor(model_name=model_name)
    else:
        raise NotImplementedError(model_name)


def generate_grid_points(
    grid_size: Tuple[int, int],
    cell_size: float = 1.0,
) -> torch.Tensor:

    grid_cols = int(grid_size[0] / cell_size)
    grid_rows = int(grid_size[1] / cell_size)

    cell_half_size = cell_size / 2.0
    x = torch.linspace(cell_half_size, grid_size[0] - cell_half_size, grid_cols, dtype=torch.float)
    y = torch.linspace(cell_half_size, grid_size[1] - cell_half_size, grid_rows, dtype=torch.float)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    return torch.vstack((grid_x.flatten(), grid_y.flatten())).T


def filter_points_by_box(
    points: torch.Tensor, box: Tuple[float, float, float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:

    x1, y1, x2, y2 = box
    valid_mask = torch.logical_and(
        torch.logical_and(points[:, 0] > x1, points[:, 0] < x2),
        torch.logical_and(points[:, 1] > y1, points[:, 1] < y2),
    )
    valid_mask = valid_mask.to(points.device)
    return points[valid_mask], valid_mask


def filter_points_by_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    points_int = (points + 0.5).int()

    points_int, valid_mask = filter_points_by_box(points_int, (0, 0, mask.shape[1], mask.shape[0]))

    points = points[valid_mask][mask[points_int[:, 1], points_int[:, 0]].bool()]

    return points


def sample_feature_map_at_points(
    feature_map_chw: torch.Tensor, points: torch.Tensor, crop_size: Tuple[int, int]
) -> torch.Tensor:

    uv = torch.div(2.0, torch.as_tensor(crop_size)).to(points.device) * points - 1.0

    query_coords = uv.unsqueeze(0).unsqueeze(2)

    features = torch.nn.functional.grid_sample(
        feature_map_chw.unsqueeze(0),
        query_coords,
        align_corners=False,
    )

    features = features[0, :, :, 0].permute(1, 0)

    return features


def lift_2d_points_to_3d(
    points: torch.Tensor,
    depth_image: torch.Tensor,
    camera_model: PinholePlaneCameraModel,
) -> torch.Tensor:
    device = points.device

    focal = 0.5 * (camera_model.f[0] + camera_model.f[1])

    points_3d_in_cam = torch.hstack(
        [
            points - torch.as_tensor(camera_model.c).to(torch.float32).to(device),
            focal * torch.ones(points.shape[0], 1).to(torch.float32).to(device),
        ]
    )
    depths = depth_image[
        torch.floor(points[:, 1]).to(torch.int32),
        torch.floor(points[:, 0]).to(torch.int32),
    ].reshape(-1, 1)
    points_3d_in_cam *= depths / points_3d_in_cam[:, 2].reshape(-1, 1)

    return points_3d_in_cam


@torch.no_grad()
def get_visual_features_registered_in_3d(
    image_chw: torch.Tensor,
    depth_image_hw: torch.Tensor,
    object_mask: torch.Tensor,
    camera: PinholePlaneCameraModel,
    T_model_from_camera: torch.Tensor,
    extractor: torch.nn.Module,
    grid_cell_size: float,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = image_chw.device

    timer = misc.Timer(enabled=debug)
    timer.start()

    grid_points = generate_grid_points(
        grid_size=(image_chw.shape[2], image_chw.shape[1]),
        cell_size=grid_cell_size,
    ).to(device)

    kernel = torch.ones(5, 5).to(device)
    object_mask_eroded = (
        kornia.morphology.erosion(object_mask.reshape(1, 1, *object_mask.shape).to(torch.float32), kernel)
        .squeeze([0, 1])
        .to(object_mask.dtype)
    )

    query_points = filter_points_by_mask(grid_points, object_mask_eroded)

    vertices_in_cam = lift_2d_points_to_3d(
        points=query_points,
        depth_image=depth_image_hw,
        camera_model=camera,
    )

    vertices_in_model = transform3d.transform_points(T_model_from_camera.to(device), vertices_in_cam.to(device))
    vertex_ids = torch.arange(vertices_in_model.shape[0], dtype=torch.int32)

    timer.elapsed("Time for preparation")

    timer.start()
    image_bchw = image_chw.unsqueeze(0)

    timer.start()

    extractor_output = extractor(image_bchw)
    cls_feature = extractor_output["cls_tokens"][0]
    feature_map_chw = extractor_output["feature_maps"][0]
    feature_map_chw = feature_map_chw.to(device)

    timer.elapsed("Time for feature extraction")
    timer.start()

    feat_vectors = sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        crop_size=(image_chw.shape[-1], image_chw.shape[-2]),
    ).detach()

    timer.elapsed("Time for feature sampling.")

    return (
        cls_feature,
        feat_vectors,
        vertex_ids,
        vertices_in_model,
    )
