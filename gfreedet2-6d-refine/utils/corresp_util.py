# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from utils import (
    knn_util,
    misc,
    pnp_util,
    repre_util,
    template_util,
    transform3d,
    config,
    logging_util,
)


logger = logging_util.get_logger(__name__)


def cyclic_buddies_matching(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    query_knn_index: knn_util.KNN,
    object_features: torch.Tensor,
    object_knn_index: knn_util.KNN,
    top_k: int,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    query2obj_nn_ids = object_knn_index.search(query_features)[1].flatten()
    obj2query_nn_ids = query_knn_index.search(object_features)[1].flatten()

    u1 = query_points

    cycle_ids = obj2query_nn_ids[query2obj_nn_ids]
    u2 = query_points[cycle_ids]

    cycle_dists = torch.linalg.norm(u1 - u2, axis=1)

    top_k = min(top_k, query_points.shape[0])
    _, query_bb_ids = torch.topk(-cycle_dists, k=top_k, sorted=True)

    bb_dists = cycle_dists[query_bb_ids]
    bb_scores = torch.as_tensor(1.0 - (bb_dists / bb_dists.max()))

    object_bb_ids = query2obj_nn_ids[query_bb_ids]

    return query_bb_ids, object_bb_ids, bb_dists, bb_scores


def establish_correspondences(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    template_matching_type: str,
    feat_matching_type: str,
    top_n_templates: int,
    top_k_buddies: int,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
    template_knn_indices: Optional[List[knn_util.KNN]] = None,
    debug: bool = False,
) -> List[Dict]:

    timer = misc.Timer(enabled=debug)
    timer.start()
    template_ids, template_scores = template_util.template_matching(
        query_features=query_features,
        object_repre=object_repre,
        top_n_templates=top_n_templates,
        matching_type=template_matching_type,
        visual_words_knn_index=visual_words_knn_index,
        debug=debug,
    )
    if debug:
        timer.elapsed("Time for template matching")
    timer.start()

    query_knn_index = None
    if feat_matching_type == "cyclic_buddies":
        query_knn_index = knn_util.KNN(k=1, metric="l2")
        query_knn_index.fit(query_features)

    corresps = []
    for template_counter, template_id in enumerate(template_ids):
        # Get IDs of features originating from the current template.
        tpl_feat_mask = torch.as_tensor(object_repre.feat_to_template_ids == template_id)
        tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

        if feat_matching_type == "cyclic_buddies":
            assert object_repre.feat_vectors is not None
            (
                match_query_ids,
                match_obj_ids,
                match_dists,
                match_scores,
            ) = cyclic_buddies_matching(
                query_points=query_points,
                query_features=query_features,
                query_knn_index=query_knn_index,
                object_features=object_repre.feat_vectors[tpl_feat_ids],
                object_knn_index=template_knn_indices[template_id],
                top_k=top_k_buddies,
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown feature matching type ({feat_matching_type}).")

        match_obj_feat_ids = tpl_feat_ids[match_obj_ids]

        coord_2d = query_points[match_query_ids]
        coord_2d_ids = match_query_ids
        assert object_repre.vertices is not None
        coord_3d = object_repre.vertices[match_obj_feat_ids]
        coord_conf = match_scores
        full_query_nn_dists = match_dists
        full_query_nn_ids = match_obj_feat_ids
        nn_vertex_ids = match_obj_feat_ids

        template_corresps = {
            "template_id": template_id,
            "template_score": template_scores[template_counter],
            "coord_2d": coord_2d,
            "coord_2d_ids": coord_2d_ids,
            "coord_3d": coord_3d,
            "coord_conf": coord_conf,
            "nn_vertex_ids": nn_vertex_ids,
        }

        if debug:
            template_corresps.update(
                {
                    "nn_dists": full_query_nn_dists,
                    "nn_indices": full_query_nn_ids,
                }
            )

        corresps.append(template_corresps)
    if debug:
        timer.elapsed("Time for establishing corresp")

    return corresps


def flows_from_poses(
    source_masks: torch.Tensor,
    source_depths: torch.Tensor,
    intrinsics: torch.Tensor,
    Ts_target_from_source: torch.Tensor,
) -> torch.Tensor:

    device = source_masks.device
    assert source_masks.dtype == torch.bool
    height = source_masks.shape[1]

    source_pixels, source_3d_points = transform3d.get_3d_points_from_depth(
        depths=source_depths,
        intrinsics=intrinsics,
    )
    source_3d_points = source_3d_points.to(device)
    source_3d_points = source_3d_points.to(device)

    source_3d_points = rearrange(source_3d_points, "b h w c -> b (h w) c")

    source_3d_points_in_target_cam = transform3d.transform_points(
        points=source_3d_points, matrix=Ts_target_from_source[:, None, ...].to(device)
    )

    proj_source_3d_points_in_target_cam = transform3d.project_3d_points_pinhole(
        source_3d_points_in_target_cam, intrinsics
    )

    proj_source_3d_points_in_target_cam = rearrange(
        proj_source_3d_points_in_target_cam, "b (h w) c -> b h w c", h=height
    )

    flows = proj_source_3d_points_in_target_cam - source_pixels

    flows[~source_masks.unsqueeze(-1).repeat(1, 1, 1, 2)] = 0.0
    return flows


def poses_from_flows(
    flows: torch.Tensor,
    source_depths: torch.Tensor,
    source_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    Ts_source_cam_from_model: Optional[torch.Tensor] = None,
    flow_weights: Optional[torch.Tensor] = None,
    weight_threshold: float = 0.3,
    pnp_solver_name: str = "opencv",
    pnp_opts: Optional[config.PnPOpts] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    (batch_size, height, width, _) = flows.shape
    device = flows.device

    source_pixels, source_3d_points = transform3d.get_3d_points_from_depth(
        depths=source_depths,
        intrinsics=intrinsics,
    )
    source_3d_points = source_3d_points.to(device)
    source_3d_points = source_3d_points.to(device)

    corresp_2d = source_pixels + flows

    if Ts_source_cam_from_model is not None:
        inv_Ts_cam_from_model_source = transform3d.inverse_se3(Ts_source_cam_from_model)
        corresp_3d = transform3d.transform_points(
            points=source_3d_points,
            matrix=inv_Ts_cam_from_model_source[:, None, None, ...],
        )
    else:
        corresp_3d = source_3d_points

    assert flow_weights is not None
    output_dict = pnp_util.poses_from_correspondences(
        corresps_2d=corresp_2d,
        corresps_3d=corresp_3d,
        corresps_weight=flow_weights,
        intrinsics=intrinsics,
        masks=source_masks,
        init_poses=Ts_source_cam_from_model,
        pnp_solver_name=pnp_solver_name,
        pnp_opts=pnp_opts,
        weight_threshold=weight_threshold,
    )
    for key in ["estimated_poses", "quality", "proj_err"]:
        assert (
            output_dict[key].shape[0] == batch_size
        ), f"{key} has different size, {output_dict[key].shape[0]} vs batch_size={batch_size}."
    return (
        torch.as_tensor(output_dict["estimated_poses"]),
        torch.as_tensor(output_dict["quality"]),
        torch.as_tensor(output_dict["proj_err"]),
    )
