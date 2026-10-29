"""Coarse 6D pose of one detection: `load_object`, `prepare_image`, `estimate_coarse_pose`.

Same crop, features, tf-idf template retrieval, cyclic-buddies matching and PnP-RANSAC as `scripts/infer_bop24_v2.py` with `final_pose_type: best_coarse`, with no hydra, rootutils or dataset paths. One detection per call, so the features are not batched over the instances of an image.
"""

import dataclasses
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from utils import corresp_util, feature_util, knn_util, misc, pnp_util, projector_util, repre_util, structs
from utils.misc import array_to_tensor, tensor_to_array, warp_image
from utils.structs import AlignedBox2f, PinholePlaneCameraModel


@dataclasses.dataclass
class CoarseObject:
    """One object's `repre` and its knn indices (faiss, on the CPU)."""

    repre: repre_util.FeatureBasedObjectRepre
    visual_words_knn_index: knn_util.KNN
    template_knn_indices: List[knn_util.KNN]

    def to(self, device: str) -> "CoarseObject":
        """A copy with the repre's tensors on `device`; the knn indices are shared."""
        moved = {f.name: getattr(self.repre, f.name).to(device) for f in dataclasses.fields(self.repre) if torch.is_tensor(getattr(self.repre, f.name))}
        return dataclasses.replace(self, repre=dataclasses.replace(self.repre, **moved))


def load_object(repre_dir: str) -> CoarseObject:
    """The `repre.pth` in `repre_dir`, on the CPU, with its knn indices. Without the template images: new files have none, old files drop them here (only the visualisation reads them)."""
    repre = repre_util.load_object_repre(repre_dir=repre_dir, tensor_device="cpu")
    repre.templates = None
    visual_words_knn_index = knn_util.KNN(k=repre.template_desc_opts.tfidf_knn_k, metric=repre.template_desc_opts.tfidf_knn_metric)
    visual_words_knn_index.fit(repre.feat_cluster_centroids)
    template_knn_indices = []
    for template_id in range(len(repre.template_cameras_cam_from_model)):
        index = knn_util.KNN(k=1, metric="l2")
        index.fit(repre.feat_vectors[torch.nonzero(repre.feat_to_template_ids == template_id).flatten()])
        template_knn_indices.append(index)
    return CoarseObject(repre, visual_words_knn_index, template_knn_indices)


def prepare_image(image: np.ndarray) -> np.ndarray:
    """`image` (HxWx3 uint8 RGB) as float32 in [0, 1]: the `image` of `estimate_coarse_pose`, for all detections of the image."""
    return image.astype(np.float32) / 255.0


def inlier_ratio(corresp: Dict[str, Any], R_m2c: np.ndarray, t_m2c: np.ndarray, camera_c2w: PinholePlaneCameraModel, inlier_thresh: float) -> float:
    """The share of the matched query points with at least one correspondence that the pose (`camera_c2w` frame) projects within `inlier_thresh` px: the score of `eval_util.EvaluatorPose.update_without_anno`."""
    points = tensor_to_array(corresp["coord_3d"]) @ R_m2c.T + np.reshape(t_m2c, (1, 3))
    hits = np.linalg.norm(tensor_to_array(corresp["coord_2d"]) - camera_c2w.eye_to_window(points), axis=1) <= inlier_thresh
    _, query = np.unique(tensor_to_array(corresp["coord_2d_ids"]), return_inverse=True)
    return float(np.bincount(query, weights=hits).astype(bool).mean())


@torch.no_grad()
def estimate_coarse_pose(
    image: np.ndarray,
    K: np.ndarray,
    mask_modal: np.ndarray,
    box_amodal: np.ndarray,
    obj: CoarseObject,
    extractor: torch.nn.Module,
    *,
    crop_size: int = 420,
    crop_rel_pad: float = 0.2,
    grid_cell_size: float = 14.0,
    match_top_n_templates: int = 5,
    match_top_k_buddies: int = 300,
    pnp_ransac_iter: int = 400,
    pnp_inlier_thresh: float = 10.0,
    pnp_required_ransac_conf: float = 0.99,
    pnp_refine_lm: bool = True,
    det_score: float = 1.0,
    device: str = "cuda",
) -> Optional[Dict[str, Any]]:
    """The coarse pose of one detection in `image`, or None if no template gives a PnP pose.
    - `image`: `prepare_image` of the HxWx3 uint8 RGB image; `K`: 3x3 of the image.
    - `mask_modal`: HxW, nonzero on the visible object; opened with a 3x3 kernel.
    - `box_amodal`: x, y, w, h in pixels (BOP's `bbox`), taken as left, top, left + w, top + h.
    - `obj`: `load_object(...).to(device)`; `extractor`: `make_extractor` of the layer the repre was made with.
    - Returns `R_m2c` (3x3), `t_m2c` (3, mm), `score`, `inliers` (PnP inliers of the best template), `matches` (its 2D-3D correspondences), `template_id`, `times` (seconds per step).
    - `score`: `inlier_ratio` of the best template's correspondences at the pose, FoundPose's BOP score; `det_score` (the detection's score) if the ratio is 0, as in GFreeDet2's `infer_bop24_v2.py`.
    """
    times = {}
    started = time.perf_counter()
    height, width = image.shape[:2]
    orig_camera_c2w = PinholePlaneCameraModel(width=width, height=height, f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]))
    mask_modal = cv2.morphologyEx(np.asarray(mask_modal != 0, np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    x, y, w, h = np.asarray(box_amodal, dtype=float)
    crop_box = misc.calc_crop_box(box=AlignedBox2f(left=x, top=y, right=x + w, bottom=y + h), make_square=True)
    camera_c2w = misc.construct_crop_camera(
        box=crop_box, camera_model_c2w=orig_camera_c2w, viewport_size=(crop_size, crop_size), viewport_rel_pad=crop_rel_pad
    )
    interpolation = cv2.INTER_AREA if crop_box.width >= camera_c2w.width else cv2.INTER_LINEAR
    crop = warp_image(src_camera=orig_camera_c2w, dst_camera=camera_c2w, src_image=image, interpolation=interpolation)
    crop_mask = warp_image(src_camera=orig_camera_c2w, dst_camera=camera_c2w, src_image=mask_modal, interpolation=cv2.INTER_NEAREST)
    times["prep"] = time.perf_counter() - started

    started = time.perf_counter()
    feature_map_chw = extractor(array_to_tensor(crop[None]).permute(0, 3, 1, 2).to(device))["feature_maps"][0]
    grid_points = feature_util.generate_grid_points(grid_size=(crop_size, crop_size), cell_size=grid_cell_size).to(device)
    query_points = feature_util.filter_points_by_mask(grid_points, array_to_tensor(crop_mask).to(device))
    if query_points.shape[0] == 0:
        query_points = grid_points
    query_features = feature_util.sample_feature_map_at_points(
        feature_map_chw=feature_map_chw, points=query_points, image_size=(crop.shape[1], crop.shape[0])
    ).contiguous()
    repre = obj.repre
    if query_features.shape[1] != repre.feat_vectors.shape[1] and len(repre.feat_raw_projectors) != 0:
        query_features = projector_util.project_features(feat_vectors=query_features, projectors=repre.feat_raw_projectors).contiguous()
    times["features"] = time.perf_counter() - started

    started = time.perf_counter()
    corresp = corresp_util.establish_correspondences(
        query_points=query_points,
        query_features=query_features,
        object_repre=repre,
        template_matching_type="tfidf",
        template_knn_indices=obj.template_knn_indices,
        feat_matching_type="cyclic_buddies",
        top_n_templates=match_top_n_templates,
        top_k_buddies=match_top_k_buddies,
        visual_words_knn_index=obj.visual_words_knn_index,
    )
    times["corresp"] = time.perf_counter() - started

    started = time.perf_counter()
    best = None
    for corresp_curr in corresp:
        if len(corresp_curr["coord_2d"]) < 6:
            continue
        success, R_m2c, t_m2c, inliers, _ = pnp_util.estimate_pose(
            corresp=corresp_curr,
            camera_c2w=camera_c2w,
            pnp_type="opencv",
            pnp_ransac_iter=pnp_ransac_iter,
            pnp_inlier_thresh=pnp_inlier_thresh,
            pnp_required_ransac_conf=pnp_required_ransac_conf,
            pnp_refine_lm=pnp_refine_lm,
        )
        if success and (best is None or len(inliers) > best["inliers"]):
            best = {"R_m2c": R_m2c, "t_m2c": t_m2c, "inliers": len(inliers), "matches": len(corresp_curr["coord_2d"]), "template_id": int(corresp_curr["template_id"]), "corresp": corresp_curr}
    times["pnp"] = time.perf_counter() - started
    if best is None:
        return None
    best["score"] = inlier_ratio(best.pop("corresp"), best["R_m2c"], best["t_m2c"], camera_c2w, pnp_inlier_thresh)
    if best["score"] < 1e-6:
        best["score"] = det_score

    # Crop camera to image camera: the image camera is the world frame.
    trans_m2c = camera_c2w.T_world_from_eye.dot(misc.get_rigid_matrix(structs.ObjectPose(R=best["R_m2c"], t=best["t_m2c"])))
    return {**best, "R_m2c": trans_m2c[:3, :3], "t_m2c": trans_m2c[:3, 3], "times": times}
