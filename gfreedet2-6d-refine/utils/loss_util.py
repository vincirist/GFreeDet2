# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3


from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from utils import logging_util, structs, transform3d, config
import torch.nn.functional as F

logger = logging_util.get_logger(__name__)
binary_cross_entropy_loss = nn.BCELoss()


def flow_regression_loss(
    pred_flows: torch.Tensor,
    gt_flows: torch.Tensor,
    visib_masks: torch.Tensor,
    loss_opts: config.GoTrackRefinerLossOpts,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    assert torch.sum(torch.isnan(pred_flows)) == 0
    assert torch.sum(torch.isinf(pred_flows)) == 0

    metrics = {}

    height, width = gt_flows.shape[-2:]

    max_flow = max(height, width) * (400 / 256)

    norm_gt_flows = torch.norm(gt_flows, dim=1)
    valid = visib_masks & (norm_gt_flows < max_flow)

    if loss_opts.distance_type == "l2":
        loss = (pred_flows - gt_flows).pow(2).sum(1).sqrt()
    elif loss_opts.distance_type == "l1":
        loss = (pred_flows - gt_flows).abs().sum(1)
    else:
        raise ValueError(f"Unknown distance type ({loss_opts.distance_type}).")

    loss = loss.view(-1)[valid.view(-1)]
    loss = loss.mean()

    epe = torch.sum((pred_flows - gt_flows).pow(2), dim=1).sqrt()

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return loss, metrics


def compute_flow_losses_and_metrics(
    num_iters: int,
    name_prefix: str,
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
    loss_opts: config.GoTrackRefinerLossOpts,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    losses = {}
    metrics = {}
    loss_name_tpl = name_prefix + "{name}_loss_iter={iter_idx:01d}"
    metric_name_tpl = name_prefix + "{name}_metric_iter={iter_idx:01d}"

    gt_flows = gt_objects.flows.cuda()
    visib_masks = gt_objects.visib_masks.cuda().bool()
    assert gt_flows.shape[1] == 2

    pred_flows = predictions["flows"]
    assert pred_flows.shape[1] == 2
    assert gt_flows.shape == pred_flows.shape

    loss, flow_metrics = flow_regression_loss(
        pred_flows=pred_flows,
        gt_flows=gt_flows,
        visib_masks=visib_masks,
        loss_opts=loss_opts,
    )

    loss_name = loss_name_tpl.format(iter_idx=0, name="sequence_flow")
    losses[loss_name] = loss

    for metric_name, metric_value in flow_metrics.items():
        metric_name = metric_name_tpl.format(iter_idx=0, name=metric_name)
        metrics[metric_name] = metric_value
    return losses, metrics


def compute_mask_losses_and_metrics(
    num_iters: int,
    name_prefix: str,
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
    loss_opts: config.GoTrackRefinerLossOpts,
    binary_threshold: float = 0.5,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    losses = {}
    metrics = {}
    loss_name_tpl = name_prefix + "{name}_loss_iter={iter_idx:01d}"
    metric_name_tpl = name_prefix + "{name}_metric_iter={iter_idx:01d}"

    gt_visib_masks = gt_objects.visib_masks.cuda().float()

    gt_masks = gt_objects.masks.cuda()

    pred_conf = predictions["confidences"]

    pred_conf = torch.clamp(pred_conf, min=1e-6)

    loss = binary_cross_entropy_loss(
        pred_conf[gt_masks],
        gt_visib_masks[gt_masks],
    )

    predicted_mask = (pred_conf > binary_threshold).float()
    accuracy = (predicted_mask == gt_visib_masks)[gt_masks]
    accuracy = accuracy.float().mean()

    loss_name = loss_name_tpl.format(iter_idx=0, name="binary_mask")
    losses[loss_name] = loss
    metric_name = metric_name_tpl.format(iter_idx=0, name="accuracy")
    metrics[metric_name] = accuracy
    return losses, metrics


def symmetric_pose_loss(
    poses_possible_gt: torch.Tensor, poses_pred: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:

    vertices_possible_gt = transform3d.transform_points(poses_possible_gt, vertices)

    vertices_pred = transform3d.transform_points(poses_pred, vertices)

    losses_possible = (vertices_pred.unsqueeze(1) - vertices_possible_gt).flatten(-2, -1).abs().mean(-1)

    loss, _ = losses_possible.min(dim=1)

    return loss


def symmetric_disentangled_pose_loss(
    poses_possible_gt: torch.Tensor,
    poses_input: torch.Tensor,
    poses_pred_delta: torch.Tensor,
    delta_pose_repre: config.PoseRepre,
    cameras: List[structs.CameraModel],
    vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:

    bsz = poses_possible_gt.shape[0]
    assert poses_possible_gt.device == poses_input.device == poses_pred_delta.device
    assert poses_possible_gt.shape[0] == bsz
    assert poses_input.shape[0] == bsz
    assert vertices.dim() == 3 and vertices.shape[0] == bsz and vertices.shape[-1] == 3
    assert poses_possible_gt.dim() == 4 and poses_possible_gt.shape[-2:] == (4, 4)

    if delta_pose_repre.value == config.PoseRepre.XYZ_CONT6D.value:

        poses_gt = poses_possible_gt[:, 0]
        z_gt = poses_gt[:, 2, [3]]

        R_delta = transform3d.rotation_matrix_from_cont6d(poses_pred_delta[:, :6])
        poses_pred_rot = poses_gt.clone()
        poses_pred_rot[:, :3, :3] = R_delta @ poses_input[:, :3, :3]
        loss_rot = symmetric_pose_loss(poses_possible_gt, poses_pred_rot, vertices)

        xy_delta = poses_pred_delta[:, 6:8]
        xy_in = poses_input[:, :2, 3]
        z_in = poses_input[:, 2, [3]]
        fxfy = torch.as_tensor([camera.f for camera in cameras])
        fxfy = fxfy.to(poses_gt.device)
        poses_pred_xy = poses_gt.clone()
        poses_pred_xy[:, :2, 3] = ((xy_delta / fxfy) + (xy_in / z_in.repeat(1, 2))) * z_gt.repeat(1, 2)
        loss_xy = symmetric_pose_loss(poses_possible_gt, poses_pred_xy, vertices)

        z_delta = poses_pred_delta[:, [8]]
        poses_pred_z = poses_gt.clone()
        poses_pred_z[:, [2], [3]] = z_delta * z_in
        loss_z = symmetric_pose_loss(poses_possible_gt, poses_pred_z, vertices)

    else:
        raise ValueError("Unknown pose representation.")

    return {"rot": loss_rot, "xy": loss_xy, "z": loss_z}


def get_symmetric_poses_tensor(
    poses_cam_from_obj: torch.Tensor,
    symmetries_obj_from_obj: List[torch.Tensor],
) -> torch.Tensor:

    num_objects = len(symmetries_obj_from_obj)
    max_sym_count = max([len(sym) for sym in symmetries_obj_from_obj])
    symmetries_obj_from_obj_tensor = torch.tile(torch.eye((4)), [num_objects, max_sym_count, 1, 1])
    for inst_id, syms in enumerate(symmetries_obj_from_obj):
        symmetries_obj_from_obj_tensor[inst_id, : len(syms)] = syms

    device = poses_cam_from_obj.unsqueeze(1).device
    symmetric_poses_cam_from_obj = poses_cam_from_obj.unsqueeze(1) @ symmetries_obj_from_obj_tensor.to(device)

    return symmetric_poses_cam_from_obj


def compute_losses(
    pose_loss_opts: config.PoseLossOpts,
    pose_repre: config.PoseRepre,
    num_iters: int,
    name_prefix: str,
    object_symmetries: Dict[int, Any],
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
) -> Dict[str, torch.Tensor]:

    losses = {}
    loss_name_tpl = name_prefix + "_{name}_loss_iter={iter_idx:03d}"

    for iter_idx in range(num_iters):
        prefix = f"iter={iter_idx}"

        Ts_crop_cam_from_orig_cam = predictions[f"{prefix}_Ts_crop_cam_from_orig_cam"]
        gt_poses_crop_cam_from_model = Ts_crop_cam_from_orig_cam @ gt_objects.poses_cam_from_model

        poses_possible_gt_cam_from_obj = get_symmetric_poses_tensor(
            poses_cam_from_obj=gt_poses_crop_cam_from_model,
            symmetries_obj_from_obj=[object_symmetries[label.item()] for label in gt_objects.labels],
        )
        poses_possible_gt_cam_from_obj = poses_possible_gt_cam_from_obj.to(gt_objects.target_vertices.device)

        losses_curr = symmetric_disentangled_pose_loss(
            poses_possible_gt=poses_possible_gt_cam_from_obj,
            poses_input=predictions[f"{prefix}_init_Ts_crop_cam_from_model"],
            poses_pred_delta=predictions[f"{prefix}_delta_poses_in_crop_cam"],
            delta_pose_repre=pose_repre,
            cameras=predictions[f"{prefix}_crop_cameras"],
            vertices=gt_objects.target_vertices,
        )

        for loss_name, loss_values in losses_curr.items():
            if loss_name == "rot":
                weight = pose_loss_opts.loss_disentangled_rot_weight
            elif loss_name == "xy":
                weight = pose_loss_opts.loss_disentangled_xy_weight
            elif loss_name == "z":
                weight = pose_loss_opts.loss_disentangled_z_weight
            else:
                raise ValueError(f"Unknown loss ({loss_name}).")
            loss_name = loss_name_tpl.format(iter_idx=iter_idx, name=loss_name)
            losses[loss_name] = loss_values.mean() * weight
    return losses


def calculate_delta_pose_in_cam(
    pose_repre: config.PoseRepre,
    source_Ts_cam_from_model: torch.Tensor,
    target_Ts_cam_from_model: torch.Tensor,
    cameras: List[structs.CameraModel],
) -> torch.Tensor:

    batch_size = source_Ts_cam_from_model.shape[0]
    device = source_Ts_cam_from_model.device

    if pose_repre == config.PoseRepre.XYZ_CONT6D:
        delta_poses_in_cam = torch.zeros((batch_size, 9), dtype=torch.float32, device=device)

        R_delta = target_Ts_cam_from_model[:, :3, :3] @ source_Ts_cam_from_model[:, :3, :3].permute(0, 2, 1)

        delta_poses_in_cam[:, :6] = transform3d.cont6d_from_rotation_matrix(R_delta)
        R_delta_recovered = transform3d.rotation_matrix_from_cont6d(delta_poses_in_cam[:, :6])
        assert torch.allclose(R_delta, R_delta_recovered, atol=1e-4)

        assert torch.sum(source_Ts_cam_from_model[:, 2, 3] == 0) == 0
        delta_poses_in_cam[:, 8] = target_Ts_cam_from_model[:, 2, 3] / source_Ts_cam_from_model[:, 2, 3]

        fxfy = torch.as_tensor([camera.f for camera in cameras])
        fxfy = fxfy.to(device)

        source_reproj = source_Ts_cam_from_model[:, :2, 3] / source_Ts_cam_from_model[:, 2, 3:]
        target_reproj = target_Ts_cam_from_model[:, :2, 3] / target_Ts_cam_from_model[:, 2, 3:]
        delta_poses_in_cam[:, 6:8] = (target_reproj - source_reproj) * fxfy

    else:
        raise ValueError(f"Not supported pose representation: {pose_repre}")
    return delta_poses_in_cam


class PairwiseSimilarity(nn.Module):

    def __init__(self, metric="cosine", chunk_size=64):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query_descriptors, reference_descriptors):
        N_query = query_descriptors.shape[0]
        N_objects, N_templates = (
            reference_descriptors.shape[0],
            reference_descriptors.shape[1],
        )
        reference_descriptors_ = reference_descriptors.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)
        query_descriptors_ = query_descriptors.clone().unsqueeze(1).repeat(1, N_templates, 1)
        query_descriptors_ = F.normalize(query_descriptors_, dim=-1)
        reference_descriptors_ = F.normalize(reference_descriptors_, dim=-1)

        similarity = structs.BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim = F.cosine_similarity(query_descriptors_, reference_descriptors_[:, idx_obj], dim=-1)
            similarity.append(sim)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2)
        return similarity.clamp(min=0.0, max=1.0)
