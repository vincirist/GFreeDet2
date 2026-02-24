# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
import numpy.typing as npt
from utils import (
    pnp_util,
    vis_base_util,
    poser_util,
    logging_util,
    renderer_base,
    structs,
    torch_helpers,
    transform3d,
)

logger = logging_util.get_logger(__name__)


def make_colorwheel() -> np.ndarray:

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def rgb_from_uv(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:

    height, weight = u.shape
    flow_image = np.zeros((height, weight, 3), np.uint8)

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # pyre-ignore
        col[~idx] = col[~idx] * 0.75
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def rgb_from_flow(
    flow_uv: np.ndarray, convert_to_bgr: bool = False, max_norm: Optional[float] = None
) -> Tuple[np.ndarray, float]:

    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    min_visible_flow_abs = 0.1
    tmp_flow = np.copy(flow_uv)
    u = tmp_flow[:, :, 0]
    v = tmp_flow[:, :, 1]
    norm = np.sqrt(np.square(u) + np.square(v))
    if max_norm is None:
        max_norm = max(np.max(norm), min_visible_flow_abs)
    epsilon = 1e-5

    overflow_index = norm >= max_norm
    overflow_scale = max_norm / (norm + epsilon)
    u[overflow_index] = u[overflow_index] * overflow_scale[overflow_index]
    v[overflow_index] = v[overflow_index] * overflow_scale[overflow_index]
    norm = np.sqrt(np.square(u) + np.square(v))
    assert np.all(norm <= max_norm), "this should not happen"

    u = u / (max_norm + epsilon)
    v = v / (max_norm + epsilon)

    rgb = rgb_from_uv(u, v, convert_to_bgr)
    return rgb, max_norm


def vis_flows(
    flows: torch.Tensor,
    flow_masks: Optional[torch.Tensor] = None,
    max_flow_norm: Optional[float] = None,
) -> Tuple[torch.Tensor, List[float]]:

    batch_size = flows.shape[0]
    device = flows.device

    flows_numpy = flows.numpy()
    if flow_masks is not None:
        flow_masks_numpy = flow_masks.unsqueeze(-1).repeat(1, 1, 1, 2).numpy()
        flows_numpy[~flow_masks_numpy] = 0

    rgbs_flow = []
    max_norms = []
    for i in range(batch_size):
        rgb_flow, max_norm = rgb_from_flow(flows_numpy[i], max_norm=max_flow_norm)
        rgbs_flow.append(rgb_flow)
        max_norms.append(max_norm)

    rgbs_flow = np.stack(rgbs_flow).astype(np.uint8)
    if flow_masks is not None:
        flow_masks_numpy = flow_masks.unsqueeze(-1).repeat(1, 1, 1, 3).numpy()
        rgbs_flow[~flow_masks_numpy] = 0

    rgbs_flow = torch.from_numpy(rgbs_flow).float() / 255.0
    return rgbs_flow.to(device), max_norms


def wrap_flows(
    rgbs_reference: torch.Tensor,
    flows: torch.Tensor,
    flow_masks: torch.Tensor,
    rgbs_query: Optional[torch.Tensor] = None,
    interp_mode: str = "nearest",
) -> torch.Tensor:

    height, width = rgbs_reference.shape[2], rgbs_reference.shape[3]
    batch_size = rgbs_reference.shape[0]

    if interp_mode == "nearest":
        homogeneous_pixels = transform3d.homogeneous_pixel_grid(width, height)
        pixel_coords = homogeneous_pixels[:, :, :2]  # H x W x 2
        pixel_coords = pixel_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # B x H x W x 2

        (
            batch_indexes,
            (source_u, source_v),
            (target_u, target_v),
        ) = pnp_util.correspondences_2d_from_flows(flows, flow_masks, return_int=interp_mode == "nearest")

        if rgbs_query is None:
            wrapped_rgbs = torch.zeros_like(rgbs_reference)
        else:
            wrapped_rgbs = rgbs_query.clone()

        wrapped_rgbs[batch_indexes, :, target_v, target_u] = rgbs_reference[batch_indexes, :, source_v, source_u]

    else:
        target_pixels = create_meshgrid(height, width, normalized_coordinates=False)
        target_pixels = target_pixels.repeat(batch_size, 1, 1, 1)

        source_pixels = target_pixels - flows
        wrapped_rgbs = remap(
            rgbs_reference,
            source_pixels[..., 0],
            source_pixels[..., 1],
            mode="bilinear",
        )
        if rgbs_query is not None:
            wrapped_rgbs[wrapped_rgbs == 0] = rgbs_query[wrapped_rgbs == 0]
    return wrapped_rgbs


def get_vis_tiles_one_iter(
    forward_inputs: Dict[str, Any],
    forward_outputs: Dict[str, Any],
    pred_rendering: Optional[structs.Collection] = None,
    pred_rendering_orig_cam: Optional[structs.Collection] = None,
    orig_input_rgbs: Optional[torch.Tensor] = None,
    pose_fitting_tiles: Optional[torch.Tensor] = None,
    visib_threshold: float = 0.3,
    text_size: int = 12,
    vis_minimal: bool = False,
) -> Dict[str, npt.NDArray]:

    if vis_minimal:
        tile_keys = ["pred_pose", "pose_fitting"]
    else:
        tile_keys = [
            "query_image",
            "template",
            "pred_pose",
            "pred_flow",
            "pred_visib",
            "pose_fitting",
        ]

    tiles = {}

    input_rgbs = rearrange(forward_inputs["rgbs_query"].cpu(), "b c h w -> b h w c")
    input_rgbs_numpy = torch_helpers.tensors_to_arrays(input_rgbs)

    if "query_image" in tile_keys:
        tiles["query_image"] = {"name": "Query image", "image": input_rgbs_numpy}

    if "template" in tile_keys:
        input_template = rearrange(forward_inputs["rgbs_template"].cpu(), "b c h w -> b h w c")
        input_template_numpy = torch_helpers.tensors_to_arrays(input_template)
        tiles["template"] = {
            "name": "Rendering of input pose",
            "image": input_template_numpy,
        }

    if "pred_flow" in tile_keys:
        output_flows = forward_outputs["flows"].detach().clone().cpu()  # bchw
        output_flows *= forward_inputs["masks_template"].cpu().unsqueeze(1)
        output_flow_rgbs, _ = vis_flows(
            rearrange(output_flows, "b c h w -> b h w c"),
            flow_masks=forward_inputs["masks_template"].cpu(),
        )
        output_flow_rgbs_numpy = torch_helpers.tensors_to_arrays(output_flow_rgbs)
        tiles["pred_flow"] = {"name": "Predicted flow", "image": output_flow_rgbs_numpy}

    output_visib_masks = None
    if "pred_visib" in tile_keys or "warped_template" in tile_keys:
        output_visib_masks = forward_outputs["confidences"].detach().clone().cpu()  # bhw
        output_visib_masks = output_visib_masks > visib_threshold
        output_visib_masks *= forward_inputs["masks_template"].cpu()

    if "pred_visib" in tile_keys:
        output_visib_mask_rgbs = output_visib_masks.unsqueeze(-1).repeat(1, 1, 1, 3)
        output_visib_mask_rgbs_numpy = torch_helpers.tensors_to_arrays(output_visib_mask_rgbs)
        tiles["pred_visib"] = {
            "name": "Predicted visibility",
            "image": output_visib_mask_rgbs_numpy,
        }

    if "warped_template" in tile_keys:
        input_rgbs_gray = vis_base_util.grays_from_rgbs(forward_inputs["rgbs_query"].cpu())
        assert output_visib_masks is not None
        rgbs_warped_template = wrap_flows(
            rgbs_reference=forward_inputs["rgbs_template"].cpu(),
            flows=rearrange(output_flows.clone(), "b c h w -> b h w c"),
            flow_masks=output_visib_masks,
            rgbs_query=input_rgbs_gray,
        )
        rgbs_prediction = rearrange(rgbs_warped_template, "b c h w -> b h w c")
        rgbs_prediction_numpy = torch_helpers.tensors_to_arrays(rgbs_prediction)
        tiles["warped_template"] = {
            "name": "Predicted warped template",
            "image": rgbs_prediction_numpy,
        }

    if "pose_fitting" in tile_keys:
        if pose_fitting_tiles is not None:
            pose_fitting_tile = torch_helpers.tensors_to_arrays(pose_fitting_tiles)
        else:
            pose_fitting_tile = np.zeros_like(input_rgbs_numpy)
        tiles["pose_fitting"] = {
            "name": "",
            "image": pose_fitting_tile,
        }

    if "pred_pose" in tile_keys:
        assert pred_rendering is not None
        rgbs_prediction = vis_base_util.vis_masks_on_rgbs(
            rgbs=forward_inputs["rgbs_query"].cpu(),
            masks=pred_rendering.masks,
        )
        rgbs_prediction_numpy = torch_helpers.tensors_to_arrays(rgbs_prediction)
        tiles["pred_pose"] = {
            "name": "Refined pose",
            "image": rgbs_prediction_numpy,
        }

    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None
    if vis_minimal:
        grid_rows = 1
        grid_cols = 2

    merged_tiles = vis_base_util.add_text_and_merge_tiles(
        [t["image"] for t in tiles.values()],
        [t["name"] for t in tiles.values()],
        text_size=text_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    if pred_rendering_orig_cam is not None:
        assert orig_input_rgbs is not None
        rgbs_prediction_orig_cam = vis_base_util.vis_masks_on_rgbs(
            rgbs=orig_input_rgbs.cpu(),
            masks=pred_rendering_orig_cam.masks,
        )
        rgbs_prediction_orig_cam_numpy = torch_helpers.tensors_to_arrays(rgbs_prediction_orig_cam)
        assert len(merged_tiles) == len(rgbs_prediction_orig_cam_numpy)

    return {f"refinement_{sample_id}": tile for sample_id, tile in enumerate(merged_tiles)}


@torch.no_grad()
def get_vis_tiles(
    predictions: Dict[str, Any],
    renderer: renderer_base.RendererBase,
    vis_output_poses: bool,
    visib_threshold: float = 0.3,
    text_size: int = 12,
    background_type: str = "gray",
    ssaa_factor: float = 1.0,
    vis_minimal: bool = False,
) -> Dict[str, npt.NDArray]:

    pred_rendering_orig_cam = None
    orig_input_rgbs = None

    if not vis_output_poses:

        pred_rendering = None
        pose_fitting_tiles = None
        prefix = "iter=0"
    else:
        num_iters = predictions["num_iters"]
        prefix = f"iter={num_iters - 1}"

        crop_cameras = predictions[f"{prefix}_inputs"]["crop_cameras"]
        pred_poses_crop_cam_from_model = predictions[f"{prefix}_pred_poses_crop_cam_from_model"]

        pred_rendering = poser_util.batch_object_render_pinhole(
            obj_ids=predictions["objects"].labels.cpu().numpy(),
            Ts_cam_from_model=pred_poses_crop_cam_from_model.cpu().numpy(),
            cameras=crop_cameras,
            renderer=renderer,
            background=[0.5, 0.5, 0.5] if background_type == "gray" else None,
            ssaa_factor=ssaa_factor,
        )
        pose_fitting_tiles = predictions[f"{prefix}_pose_fitting_tiles"]

    return get_vis_tiles_one_iter(
        forward_inputs=predictions[f"{prefix}_inputs"],
        forward_outputs=predictions[f"{prefix}_outputs"],
        pred_rendering=pred_rendering,
        pred_rendering_orig_cam=pred_rendering_orig_cam,
        orig_input_rgbs=orig_input_rgbs,
        pose_fitting_tiles=pose_fitting_tiles,
        visib_threshold=visib_threshold,
        text_size=text_size,
        vis_minimal=vis_minimal,
    )
