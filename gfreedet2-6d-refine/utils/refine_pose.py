"""GoTrack refinement of the 6D poses in one image: `load_refiner`, `make_renderer`, `add_object`, `refine_poses`.

Same crop, render, flow and PnP iterations as `GoTrack.forward_pipeline` (`model/gotrack.py`), with no hydra, lightning trainer, dataloader or result files. The renderer is pyrender for a mesh (GoTrack) or gsplat for a splat (GoTrack-GS).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData

from model import config
from model.gotrack import GoTrack
from utils import data_util, net_util, renderer_base, renderer_builder, structs

RENDERERS = {"mesh": renderer_builder.RendererType.PYRENDER_RASTERIZER, "splat": renderer_builder.RendererType.GSPLAT_RASTERIZER}
#: The render background of each `background` option.
BACKGROUNDS = {"gray": (0.5, 0.5, 0.5), "black": None}
#: Model vertices per object for the crop box, as the BOP dataloaders.
MAX_VERTICES = 1000


def load_refiner(checkpoint: str, *, iterations: int = 5, crop_size: int = 280, background: str = "gray", device: str = "cuda") -> GoTrack:
    """GoTrack (DINOv2-S, decoder, DPT head) with the weights of `checkpoint` (`gotrack_checkpoint.pt`), in eval mode on `device`.
    - `iterations`, `crop_size` (px, square, a multiple of 14) and `background` (`gray` or `black`) as `configs/model/gotrack.yaml`.
    """
    opts = config.GoTrackOpts(num_iterations_test=iterations, crop_size=(crop_size, crop_size), background_type=background, test_chunk_size=24, debug=False)
    model = GoTrack(opts, pretrained_backbone=False)
    net_util.load_checkpoint(model=model, checkpoint_path=checkpoint, checkpoint_key="model_state_dict", prefix="models.1.")
    return model.to(device).eval()


def make_renderer(renderer: str) -> renderer_base.RendererBase:
    """An empty renderer: `mesh` (pyrender, meshes in mm) or `splat` (gsplat, splat PLYs in mm, any SH degree)."""
    if renderer == "splat":
        return renderer_builder.build(renderer_type=RENDERERS[renderer], model_scale=0.001)
    return renderer_builder.build(renderer_type=RENDERERS[renderer])


def add_object(renderer: renderer_base.RendererBase, obj_id: int, model_path: str, background: str = "gray", seed: int = 0) -> np.ndarray:
    """Adds the model at `model_path` (PLY, mm) as `obj_id`; returns up to `MAX_VERTICES` of its vertices (mesh) or means (splat), (N, 3) float32 mm, sampled with `seed`."""
    renderer.add_object_model(obj_id=obj_id, model_path=model_path, background=BACKGROUNDS[background])
    vertex = PlyData.read(model_path).elements[0]
    points = np.stack([np.asarray(vertex[axis]) for axis in "xyz"], axis=1)
    return np.random.default_rng(seed).permutation(points)[:MAX_VERTICES].astype(np.float32)


@torch.no_grad()
def refine_poses(
    model: GoTrack,
    renderer: renderer_base.RendererBase,
    image: np.ndarray,
    K: np.ndarray,
    obj_ids: List[int],
    poses_m2c: np.ndarray,
    vertices: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """The refined poses of the objects in `image`, and their pose scores.
    - `image`: HxWx3 uint8 RGB; `K`: 3x3 of the image.
    - `obj_ids`: one per pose, each added to `renderer` with `add_object`; `vertices`: `add_object`'s vertices per obj_id.
    - `poses_m2c`: (N, 4, 4) initial poses, model to camera, mm.
    - Returns the poses (N, 4, 4, mm) and scores (N,) of the last iteration. The score is the PnP inlier share of the flow confidences, in [0, 1]; a failed PnP keeps its input pose with score 0.
    """
    opts = model.opts
    height, width = image.shape[:2]
    poses = torch.as_tensor(np.asarray(poses_m2c), dtype=torch.float32)
    images = structs.Collection()
    images.bitmaps = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 255.0
    images.cameras = [structs.PinholePlaneCameraModel(width=width, height=height, f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]), T_world_from_eye=np.eye(4, dtype=np.float32))]
    objects = structs.Collection()
    objects.labels = torch.as_tensor(obj_ids, dtype=torch.int32)
    objects.frame_ids = torch.zeros(len(obj_ids), dtype=torch.int32)
    objects.poses_cam_from_model = poses
    objects.poses_world_from_model = poses.clone()
    inputs = {"images": images, "objects": objects}
    object_vertices = [vertices[obj_id] for obj_id in obj_ids]
    scores: Optional[torch.Tensor] = None
    for _ in range(opts.num_iterations_test):
        inputs = data_util.process_inputs(
            inputs=inputs,
            crop_size=opts.crop_size,
            crop_rel_pad=opts.crop_rel_pad,
            cropping_type=opts.cropping_type,
            ssaa_factor=opts.ssaa_factor,
            object_vertices=object_vertices,
            obj_ids=list(obj_ids),
            renderer=renderer,
            background_type=opts.background_type,
        )
        forward_inputs = data_util.prepare_inputs(inputs)
        _, _, poses_orig_cam, scores, _, _ = model.compute_poses_from_flows(forward_inputs, model.compute_flow_by_chunk(forward_inputs))
        objects.poses_cam_from_model = poses_orig_cam.cpu()
        objects.poses_world_from_model = poses_orig_cam.cpu()
    return objects.poses_cam_from_model.numpy(), torch.as_tensor(scores).float().cpu().numpy()
