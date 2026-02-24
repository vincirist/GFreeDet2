# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import time
import warnings
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from kornia.geometry.conversions import rotation_matrix_to_axis_angle

from utils import (
    logging_util,
    misc,
    structs,
    transform3d,
    vis_base_util,
    torch_helpers,
    config,
)


warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")
logger = logging_util.get_logger(__name__)


def estimate_pose(
    corresp: Dict[str, Any],
    camera_c2w: structs.PinholePlaneCameraModel,
    pnp_type: str,
    pnp_ransac_iter: int,
    pnp_inlier_thresh: float,
    pnp_required_ransac_conf: float,
    pnp_refine_lm: bool,
    pnp_in_ransac_type: str = "iterative",  # NOTE: added; if "iterative", use iterative PnP; if "epnp", use EPnP
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:

    if pnp_type == "opencv":
        object_points = torch_helpers.tensor_to_array(corresp["coord_3d"]).astype(np.float32)
        image_points = torch_helpers.tensor_to_array(corresp["coord_2d"]).astype(np.float32)
        K = misc.get_intrinsic_matrix(camera_c2w)
        if pnp_in_ransac_type == "iterative":
            flags = cv2.SOLVEPNP_ITERATIVE
        elif pnp_in_ransac_type == "epnp":
            flags = cv2.SOLVEPNP_EPNP
        else:
            raise ValueError(f"Unknown PnP type {pnp_in_ransac_type}")
        try:
            pose_est_success, rvec_est_m2c, t_est_m2c, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=pnp_ransac_iter,
                reprojectionError=pnp_inlier_thresh,
                confidence=pnp_required_ransac_conf,
                flags=flags,
            )
        except Exception:

            pose_est_success = False
            r_est_m2c = None
            t_est_m2c = None
            inliers = None
            quality = None
        else:

            if pose_est_success and pnp_refine_lm:
                rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                    objectPoints=object_points[inliers],
                    imagePoints=image_points[inliers],
                    cameraMatrix=K,
                    distCoeffs=None,
                    rvec=rvec_est_m2c,
                    tvec=t_est_m2c,
                )

            r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]
            quality = 0.0
            if pose_est_success:
                quality = float(len(inliers))

    elif pnp_type is None:
        raise ValueError("Unsupported PnP type")

    return pose_est_success, r_est_m2c, t_est_m2c, inliers, quality


def correspondences_2d_from_flows(
    flows: torch.Tensor, flow_masks: torch.Tensor, return_int: bool = True
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

    height, width = flow_masks.shape[1:]

    nonzero_coords_source = flow_masks.nonzero()
    batch_indexes, source_v, source_u = (
        nonzero_coords_source[:, 0],
        nonzero_coords_source[:, 1],
        nonzero_coords_source[:, 2],
    )

    target_u = source_u + flows[batch_indexes, source_v, source_u, 0]
    target_v = source_v + flows[batch_indexes, source_v, source_u, 1]

    target_u = torch.clamp(target_u, 0, width - 1)
    target_v = torch.clamp(target_v, 0, height - 1)
    if return_int:
        target_u = target_u.long()
        target_v = target_v.long()
    return batch_indexes, (source_u, source_v), (target_u, target_v)


def eval_pnp_output(
    width: int,
    height: int,
    corresp_3d: np.ndarray,
    corresp_2d: np.ndarray,
    corresp_weight: np.ndarray,
    intrinsic: np.ndarray,
    t_est_m2c: np.ndarray,
    rvec_est_m2c: np.ndarray,
    pnp_opts: Optional[config.PnPOpts] = None,
) -> Dict[str, Any]:

    if pnp_opts is None:
        pnp_opts = config.PnPOpts()
    outputs = {}
    pose_m2c = np.eye(4)
    pose_m2c[:3, :3] = cv2.Rodrigues(rvec_est_m2c)[0]
    pose_m2c[:3, 3:] = t_est_m2c

    outputs["is_behind_camera"] = pose_m2c[2, 3] < 0

    corresp_3d_in_cam = transform3d.transform_points(points=corresp_3d, matrix=pose_m2c)
    assert isinstance(corresp_3d_in_cam, np.ndarray)
    assert isinstance(corresp_2d, np.ndarray)
    proj_corresp_3d_in_cam = transform3d.project_3d_points_pinhole_numpy(
        points=corresp_3d_in_cam,
        intrinsics=intrinsic,
    )

    assert isinstance(proj_corresp_3d_in_cam, np.ndarray)
    proj_corresp_3d_in_cam[:, 0] = np.clip(proj_corresp_3d_in_cam[:, 0], 0, width - 1)
    proj_corresp_3d_in_cam[:, 1] = np.clip(proj_corresp_3d_in_cam[:, 1], 0, height - 1)

    reprojection_error = np.linalg.norm(corresp_2d - proj_corresp_3d_in_cam, axis=1)
    p50 = np.percentile(reprojection_error, 50)
    p95 = np.percentile(reprojection_error, 95)
    p100 = np.max(reprojection_error)

    inlier_masks = reprojection_error <= pnp_opts.pnp_inlier_thresh

    outputs["num_inliers"] = np.sum(inlier_masks)

    proj_err = np.zeros((height, width), dtype=np.float32)
    corresp_2d_u = np.clip(corresp_2d[:, 0], 0, width - 1)
    corresp_2d_v = np.clip(corresp_2d[:, 1], 0, height - 1)
    proj_err[np.int32(corresp_2d_v), np.int32(corresp_2d_u)] = reprojection_error

    proj_err = vis_base_util.rgb_from_error_map(proj_err, max_norm=None)

    pose_quality = np.sum(corresp_weight[inlier_masks]) / np.sum(corresp_weight)

    assert pose_quality <= 1.0
    outputs["quality"] = pose_quality

    assert isinstance(proj_err, np.ndarray)
    proj_err_img = vis_base_util.write_text_on_image(
        np.uint8(proj_err),  # pyre-ignore
        [
            {"name": f"Pose quality: {pose_quality:.2f}"},
            {"name": f"Proj err p50|95|100: {p50:.1f}|{p95:.1f}|{p100:.1f}"},
        ],
        size=14,
    )
    outputs["proj_err"] = np.float32(proj_err_img) / 255.0
    return outputs


def poses_from_correspondences(  # noqa: C901
    corresps_2d: torch.Tensor,
    corresps_3d: torch.Tensor,
    corresps_weight: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    pnp_solver_name: str,
    init_poses: Optional[torch.Tensor] = None,
    pnp_opts: Optional[config.PnPOpts] = None,
    weight_threshold: float = 0.3,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:

    if pnp_opts is None:
        pnp_opts = config.PnPOpts()
    batch_size = len(corresps_2d)
    width, height = masks.shape[-2], masks.shape[-1]
    device = corresps_2d[0].device

    corresps_2d = rearrange(corresps_2d, "b h w c -> b (h w) c")
    corresps_3d = rearrange(corresps_3d, "b h w c -> b (h w) c")
    corresps_weight = rearrange(corresps_weight, "b h w -> b (h w)")

    visible_masks = corresps_weight > weight_threshold

    if init_poses is not None and pnp_solver_name == "opencv":
        init_rvecs = rotation_matrix_to_axis_angle(init_poses[:, :3, :3])
        init_tvecs = init_poses[:, :3, 3]

    outputs = {
        "estimated_poses": [],
        "num_inliers": [],
        "is_behind_camera": [],
        "quality": [],
        "proj_err": [],
        "run_time": [],
        "failed": [],
    }

    corresps_2d = torch_helpers.tensor_to_array(corresps_2d).astype(np.float64)
    corresps_3d = torch_helpers.tensor_to_array(corresps_3d).astype(np.float64)
    weights = torch_helpers.tensor_to_array(corresps_weight).astype(np.float64)
    visible_masks = torch_helpers.tensor_to_array(visible_masks).astype(np.bool_)
    intrinsics = torch_helpers.tensor_to_array(intrinsics).astype(np.float64)

    if init_poses is not None:
        init_rvecs = torch_helpers.tensor_to_array(init_rvecs)
        init_tvecs = torch_helpers.tensor_to_array(init_tvecs)

    solvePnPRansac = partial(
        cv2.solvePnPRansac,
        distCoeffs=None,
        iterationsCount=pnp_opts.pnp_ransac_iter,
        reprojectionError=pnp_opts.pnp_inlier_thresh,
        confidence=pnp_opts.pnp_required_ransac_conf,
    )

    for i in range(batch_size):
        visible_mask = visible_masks[i]
        corresp_2d_ = corresps_2d[i][visible_mask]
        corresp_3d_ = corresps_3d[i][visible_mask]
        corresp_weight_ = weights[i][visible_mask]
        intrinsic = intrinsics[i]
        if init_poses is not None:
            init_rvecs_ = init_rvecs[i]  # pyre-ignore
            init_tvecs_ = init_tvecs[i]  # pyre-ignore
        else:
            init_rvecs_ = None
            init_tvecs_ = None

        if pnp_solver_name == "opencv":
            num_corresp_avail = np.sum(corresp_weight_ >= weight_threshold)
            pose_est_success = True
            if num_corresp_avail >= 6:
                selected_corresp_2d_ = corresp_2d_
                selected_corresp_3d_ = corresp_3d_
                if num_corresp_avail >= pnp_opts.max_num_corresps:
                    sampled_ids = np.random.choice(
                        len(corresp_2d_),
                        pnp_opts.max_num_corresps,
                        replace=False,
                    )
                    selected_corresp_2d_ = selected_corresp_2d_[sampled_ids]
                    selected_corresp_3d_ = selected_corresp_3d_[sampled_ids]

                start_time = time.time()
                if pnp_opts.use_iterative_pnp:
                    flags = cv2.SOLVEPNP_ITERATIVE
                else:
                    flags = cv2.SOLVEPNP_EPNP
                pose_est_success, rvec_est_m2c, t_est_m2c, idx_inliers = solvePnPRansac(
                    objectPoints=selected_corresp_3d_,
                    imagePoints=selected_corresp_2d_,
                    cameraMatrix=intrinsic,
                    rvec=init_rvecs_,
                    tvec=init_tvecs_,
                    flags=flags,
                )

                run_time = time.time() - start_time
                if pose_est_success and idx_inliers is not None:
                    num_inliers = len(idx_inliers)
                else:
                    num_inliers = 0

                if pnp_opts.use_iterative_pnp and t_est_m2c[2] < 0:
                    start_time = time.time()
                    (
                        pose_est_success,
                        rvec_est_m2c,
                        t_est_m2c,
                        idx_inliers,
                    ) = solvePnPRansac(
                        objectPoints=selected_corresp_3d_,
                        imagePoints=selected_corresp_2d_,
                        cameraMatrix=intrinsic,
                        rvec=init_rvecs_,
                        tvec=init_tvecs_,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    run_time = time.time() - start_time
                    if pose_est_success and idx_inliers is not None:
                        num_inliers = len(idx_inliers)
                    else:
                        num_inliers = 0

                if pose_est_success and num_inliers >= 6 and idx_inliers is not None:
                    start_time = time.time()
                    rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                        objectPoints=selected_corresp_3d_[idx_inliers],
                        imagePoints=selected_corresp_2d_[idx_inliers],
                        cameraMatrix=intrinsic,
                        distCoeffs=None,
                        rvec=rvec_est_m2c,
                        tvec=t_est_m2c,
                    )
                    run_time += time.time() - start_time

                if pose_est_success:
                    pnp_stats = eval_pnp_output(
                        width=width,
                        height=height,
                        pnp_opts=pnp_opts,
                        corresp_3d=corresp_3d_,  # pyre-ignore
                        corresp_2d=corresp_2d_,  # pyre-ignore
                        corresp_weight=corresp_weight_,
                        intrinsic=intrinsic,  # pyre-ignore
                        t_est_m2c=t_est_m2c,
                        rvec_est_m2c=rvec_est_m2c,
                    )
                    for k, v in pnp_stats.items():
                        outputs[k].append(v)
                    outputs["run_time"].append(run_time)
                    outputs["failed"].append(False)

            if num_corresp_avail <= 5 or not pose_est_success:
                rvec_est_m2c, t_est_m2c = init_rvecs_, init_tvecs_  # pyre-ignore
                idx_inliers = []
                outputs["num_inliers"].append(0)
                outputs["is_behind_camera"].append(False)
                outputs["quality"].append(0.0)
                outputs["proj_err"].append(np.zeros((height, width, 3)))
                outputs["run_time"].append(0.0)
                outputs["failed"].append(not pose_est_success)

            r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]  #  pyre-ignore
            pose = transform3d.Rt_to_4x4_numpy(R=r_est_m2c, t=t_est_m2c.reshape(-1, 3))
            pose = torch.from_numpy(pose).float().to(device)
            outputs["estimated_poses"].append(pose)

        else:
            raise ValueError(f"Unknown PnP type {pnp_solver_name}")
    assert len(outputs["estimated_poses"]) == batch_size, f"{len(outputs['estimated_poses'])} != {batch_size}"
    for k in [
        "estimated_poses",
        "num_inliers",
        "is_behind_camera",
        "quality",
        "proj_err",
    ]:
        if len(outputs[k]):
            if isinstance(outputs[k][0], torch.Tensor):
                outputs[k] = torch.stack(outputs[k]).to(device)  # pyre-ignore
            else:
                outputs[k] = np.asarray(outputs[k])  # pyre-ignore
    outputs["run_time"] = np.sum(outputs["run_time"])
    outputs["failed"] = np.asarray(outputs["failed"])  # pyre-ignore
    return outputs  # pyre-ignore
