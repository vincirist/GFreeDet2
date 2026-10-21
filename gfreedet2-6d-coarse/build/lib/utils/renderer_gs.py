import os

os.environ["DISPLAY"] = ":1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import time
from plyfile import PlyData
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import numpy as np
import trimesh
import gsplat
from gsplat import rasterization
from utils.misc import tensor_to_array
from utils import renderer_base, structs, misc, logging_util
from utils.transform3d import inverse_se3_numpy
from PIL import Image
import os.path as osp
import copy


logger = logging_util.get_logger_v2(__name__)


class RenderType(Enum):
    """The rendering type.

    COLOR: An RGB image.
    DEPTH: A depth image with the depth values in mm.
    NORMAL: A normal map with normals expressed in the camera space.
    MASK: A binary mask.
    """

    COLOR = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    MASK = "mask"


def adjust_background_color_tensor(color: torch.Tensor, mask: torch.Tensor, background: structs.Color) -> torch.Tensor:
    """调整tensor格式图像的背景颜色。

    Args:
        color: 前景颜色图像 [H, W, 3]
        mask: 前景mask [H, W]
        background: 背景颜色元组 (R, G, B)

    Returns:
        调整背景颜色后的图像
    """
    height, width = color.shape[:2]
    device = color.device
    background_color = torch.tensor(background, device=device, dtype=torch.float32)
    background_img = background_color.unsqueeze(0).unsqueeze(0).expand(height, width, 3)
    return torch.where(mask.unsqueeze(-1).expand_as(color), color, background_img)


def adjust_background_color_numpy(color: np.ndarray, mask: np.ndarray, background: structs.Color) -> np.ndarray:
    """调整numpy格式图像的背景颜色。

    Args:
        color: 前景颜色图像 [H, W, 3]
        mask: 前景mask [H, W]
        background: 背景颜色元组 (R, G, B)

    Returns:
        调整背景颜色后的图像
    """
    height, width = color.shape[:2]
    background_color = np.array(background, dtype=np.float32)
    background_img = np.full((height, width, 3), background_color, dtype=np.float32)
    return np.where(mask[..., np.newaxis], color, background_img)


def load_gs_from_ply(path, requires_grad: bool = False, max_sh_degree=None, device="cuda", scale: float = 1.0):
    """The Gaussians of a splat PLY.
    - `max_sh_degree`: `None` reads the degree from the `f_rest_*` count.
    - `scale` multiplies the means and the scales, e.g. 0.001 for a splat in mm (the renderer works in m).
    """
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if max_sh_degree is None:
        max_sh_degree = round((len(extra_f_names) / 3 + 1) ** 0.5) - 1
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz * scale
    scales = scales + np.log(scale)

    gs_data = edict()
    # print(requires_grad)
    gs_data.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device), requires_grad=requires_grad)
    gs_data.features_dc = nn.Parameter(
        torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous(),
        requires_grad=requires_grad,
    )
    gs_data.features_rest = nn.Parameter(
        torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous(),
        requires_grad=requires_grad,
    )
    gs_data.opacity = nn.Parameter(
        torch.tensor(opacities, dtype=torch.float, device=device), requires_grad=requires_grad
    )
    gs_data.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device), requires_grad=requires_grad)
    gs_data.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device), requires_grad=requires_grad)
    gs_data.max_radii2D = torch.zeros((gs_data.xyz.shape[0]), device=device)

    gs_data.active_sh_degree = max_sh_degree
    return gs_data


class GsplatRasterizer(renderer_base.RendererBase):
    """The base class which all renderers should inherit."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """`model_scale` multiplies the means and scales of every loaded splat, e.g. 0.001 for splats in mm."""

        # Per-object scene (a special scene is created for each object model).
        self.object_gaussians: Dict[int, Dict[str, Any]] = {}

        self.im_size: Optional[Tuple[int, int]] = None

        self.model_path = model_path  # template model path with {obj_id}
        self.model_scale = model_scale

    def get_object_model(
        self,
        obj_id: int,
        **kwargs: Any,
    ):
        """Gets the object model.

        Args:
            obj_id: The object ID.
        """

        # Load the object model.
        object_model_path = self.model_path.format(obj_id=obj_id)
        if obj_id not in self.object_gaussians:
            gs_data = load_gs_from_ply(object_model_path, scale=self.model_scale)
            sh_features = torch.cat((gs_data.features_dc, gs_data.features_rest), dim=1)
            quats = torch.nn.functional.normalize(gs_data.rotation)
            scales = torch.exp(gs_data.scaling)
            opacities = torch.sigmoid(gs_data.opacity).view(-1)

            # necessary for rendering
            obj_gs_data = edict()
            obj_gs_data.xyz = gs_data.xyz
            obj_gs_data.quats = quats
            obj_gs_data.scales = scales
            obj_gs_data.opacities = opacities
            obj_gs_data.sh_features = sh_features
            obj_gs_data.active_sh_degree = gs_data.active_sh_degree
            self.object_gaussians[obj_id] = obj_gs_data
        else:
            obj_gs_data = self.object_gaussians[obj_id]
        return obj_gs_data

    def add_object_model(
        self,
        obj_id: int,
        model_path: str,
        **kwargs: Any,
    ) -> None:
        """Adds an object model to the renderer.

        Args:
            asset_key: The key of an asset to add to the renderer.
            mesh_color: A single color to be applied to the whole mesh. Original
                mesh colors are used if not specified.
        """

        # Load the object model.
        if obj_id not in self.object_gaussians:
            gs_data = load_gs_from_ply(model_path, scale=self.model_scale)
            sh_features = torch.cat((gs_data.features_dc, gs_data.features_rest), dim=1)
            quats = torch.nn.functional.normalize(gs_data.rotation)
            scales = torch.exp(gs_data.scaling)
            opacities = torch.sigmoid(gs_data.opacity).view(-1)

            # necessary for rendering
            obj_gs_data = edict()
            obj_gs_data.xyz = gs_data.xyz
            obj_gs_data.quats = quats
            obj_gs_data.scales = scales
            obj_gs_data.opacities = opacities
            obj_gs_data.sh_features = sh_features
            obj_gs_data.active_sh_degree = gs_data.active_sh_degree
            self.object_gaussians[obj_id] = obj_gs_data
        else:
            # trimesh_model = self.object_meshes[obj_id]
            obj_gs_data = self.object_gaussians[obj_id]

    def render_object_model(
        self,
        obj_id: int,
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        debug: bool = False,
        model_path: Optional[str] = None,
        antialias: bool = True,
        background: Optional[structs.Color] = None,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:
        """Renders an object model in the specified pose.

        Args:
            camera_model_c2m: A camera model with the extrinsics set to a rigid
                transformation from the camera to the model frame.
            render_types: Types of images to render.
            return_tensors: Whether to return the renderings as tensors or arrays.
            debug: Whether to save/print debug outputs.
        Returns:
            A dictionary with the rendering output (an RGB image, a depth image,
            a mask, a normal map, etc.).
        """

        # Create a scene for the object model if it does not exist yet.
        if obj_id not in self.object_gaussians:
            if model_path is None:
                assert self.model_path is not None, "model_path or self.model_path must be specified"
                model_path = self.model_path.format(obj_id=obj_id)
            self.add_object_model(obj_id, model_path=model_path)

        # Render the scene.
        return self._render_scene(
            gs_obj_in_w=self.object_gaussians[obj_id],
            camera_model_c2w=camera_model_c2w,
            render_types=render_types,
            return_tensors=return_tensors,
            debug=debug,
            antialias=antialias,
            background=background,
        )

    def _render_scene(
        self,
        gs_obj_in_w,
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[renderer_base.RenderType],
        return_tensors: bool = False,
        debug: bool = False,
        antialias: bool = True,
        background: Optional[structs.Color] = None,
    ) -> Dict[renderer_base.RenderType, structs.ArrayData]:
        """Renders an object model in the specified pose (see the base class)."""
        device = torch.device("cuda")

        times = {}
        times["init_renderer"] = time.time()
        width = camera_model_c2w.width
        height = camera_model_c2w.height
        self.im_size = (width, height)
        times["init_renderer"] = time.time() - times["init_renderer"]

        times["init_scene"] = time.time()

        trans_c2w = copy.deepcopy(camera_model_c2w.T_world_from_eye)

        # Convert translation from mm to m.
        trans_c2w[:3, 3] *= 0.001

        T_c_from_w = inverse_se3_numpy(trans_c2w)

        render_pose = torch.as_tensor(T_c_from_w, device=device, dtype=torch.float32)[None]  # 1x3x3

        # Camera for rendering.
        K = np.array(
            [
                [camera_model_c2w.f[0], 0, camera_model_c2w.c[0]],
                [0, camera_model_c2w.f[1], camera_model_c2w.c[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        Ks = torch.from_numpy(K)[None].to(device)  # 1x3x3

        times["init_scene"] = time.time() - times["init_scene"]
        times["render"] = time.time()

        # Rendering.
        color = None
        depth = None
        rgbd, alphas, meta = rasterization(
            means=gs_obj_in_w.xyz,  # [N, 3]
            quats=gs_obj_in_w.quats,  # [N, 4]
            scales=gs_obj_in_w.scales,  # [N, 3]
            opacities=gs_obj_in_w.opacities,  # [N]
            colors=gs_obj_in_w.sh_features,  # [(C,) N, D] or [(C,) N, K, 3]
            viewmats=render_pose,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            # backgrounds=backgrounds,  # gsplat 1.5.3 has bug, just set to None which should also be black bg
            camera_model="pinhole",
            render_mode="RGB+ED",  # render color with expected depth
            sh_degree=gs_obj_in_w.active_sh_degree,
            rasterize_mode="antialiased" if antialias else "classic",
        )
        times["render"] = time.time() - times["render"]
        times["postprocess"] = time.time()

        if return_tensors:
            color = rgbd[0, :, :, :3].to(torch.float32)
            mask = alphas[0, ..., 0] > 0.5  # binary mask
            depth = rgbd[0, :, :, 3].to(torch.float32) * 1000.0 * mask  # mm, 0 off the mask
            if background is not None:
                color = adjust_background_color_tensor(color, mask, background)

        else:
            color = rgbd[0, :, :, :3].detach().cpu().numpy().astype(np.float32)  # 0-1
            mask = alphas[0, ..., 0].detach().cpu().numpy() > 0.5
            depth = rgbd[0, :, :, 3].detach().cpu().numpy().astype(np.float32) * 1000.0 * mask  # 0 off the mask, as the mesh renderer
            if background is not None:
                color = adjust_background_color_numpy(color, mask, background)

        # Prepare the output.
        output = {
            renderer_base.RenderType.COLOR: color,
            renderer_base.RenderType.DEPTH: depth,
            renderer_base.RenderType.MASK: mask,
        }

        times["postprocess"] = time.time() - times["postprocess"]

        if debug:
            logger.info("GsplatRasterizer run times:")
            for time_name, time_value in times.items():
                logger.info(f"- {time_name}: {time_value:.04f}s")

        return output


# def get_opengl_to_opencv_camera_trans() -> np.ndarray:
#     """Returns a transformation from OpenGL to OpenCV camera frame.

#     Returns:
#         A 4x4 transformation matrix (flipping Y and Z axes).
#     """

#     yz_flip = np.eye(4, dtype=np.float32)
#     yz_flip[1, 1], yz_flip[2, 2] = -1, -1
#     return yz_flip


# def get_opencv_to_opengl_camera_trans() -> np.ndarray:
#     return get_opengl_to_opencv_camera_trans()
