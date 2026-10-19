"""Templates of one object: `generate_templates`.

Same sampling, crop and file formats as `scripts/gen_templates_gs_v2.py`, with a choice of renderer and no hydra, rootutils or dataset paths.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from bop_toolkit_lib import inout
from utils import geometry, misc, renderer_builder, structs
from utils.misc import calc_2d_box, calc_crop_box, construct_crop_camera, sample_views, warp_depth_image, warp_image
from utils.renderer_base import RenderType
from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.transform3d import inverse_se3_numpy

RENDERERS = {
    "mesh": renderer_builder.RendererType.PYRENDER_RASTERIZER,
    "splat": renderer_builder.RendererType.GSPLAT_RASTERIZER,
}


def template_files(index: int) -> Dict[str, str]:
    """File names of template `index`: `image` (RGBA), `depth` (uint16, mm) and `K_pose`."""
    return {"image": f"{index:06d}.png", "depth": f"{index:06d}_depth.png", "K_pose": f"{index:06d}_K_pose.npy"}


def sample_template_views(num_viewpoints: int, num_inplane_rotations: int, radius: float) -> List[Dict[str, np.ndarray]]:
    """Camera views `R`, `t` (model to camera, mm) on a viewsphere of `radius`, each rotated in the image plane."""
    views_sphere = sample_views(min_n_views=num_viewpoints, radius=radius, mode="fibonacci")[0]
    if num_inplane_rotations == 1:
        return views_sphere
    inplane_angle = 2 * np.pi / num_inplane_rotations
    views = []
    for view_sphere in views_sphere:
        for inplane_id in range(num_inplane_rotations):
            R_inplane = geometry.rotation_matrix_numpy(inplane_angle * inplane_id, np.array([0, 0, 1]))[:3, :3]
            views.append({"R": R_inplane.dot(view_sphere["R"]), "t": R_inplane.dot(view_sphere["t"])})
    return views


def render_camera(image_size: Tuple[int, int], K: np.ndarray, patch_size: int, ssaa_factor: float) -> PinholePlaneCameraModel:
    """Square camera of side `patch_size * int(max side / patch_size)`, centred on the dataset camera, times `ssaa_factor`."""
    width, height = image_size
    side = patch_size * int(max(width, height) / patch_size)
    return PinholePlaneCameraModel(
        width=int(side * ssaa_factor),
        height=int(side * ssaa_factor),
        f=(K[0, 0] * ssaa_factor, K[1, 1] * ssaa_factor),
        c=((K[0, 2] - 0.5 * (width - side)) * ssaa_factor, (K[1, 2] - 0.5 * (height - side)) * ssaa_factor),
    )


def generate_templates(
    model_path: str,
    out_dir: str,
    *,
    renderer: str,
    image_size: Tuple[int, int],
    K: np.ndarray,
    radius: float,
    min_num_viewpoints: int = 57,
    num_inplane_rotations: int = 14,
    crop_size: int = 420,
    crop_rel_pad: float = 0.2,
    ssaa_factor: float = 1.0,
    patch_size: int = 14,
    progress: Optional[Callable] = None,
) -> int:
    """Render the templates of one object into `out_dir`; returns their number.
    - `renderer`: `mesh` (pyrender; `model_path` is a mesh in mm) or `splat` (gsplat; `model_path` is a splat PLY).
    - `image_size` (width, height) and `K` (3x3) are those of the dataset camera. They set the focal length and the render size.
    - Per template `NNNNNN.png` (RGBA, alpha = mask), `NNNNNN_depth.png` (uint16, mm) and `NNNNNN_K_pose.npy` (pickled dict `K`, `height`, `width`, `pose_c_o`, `box_amodal`; the camera is the crop camera).
    - `progress` wraps the view iterable, e.g. `tqdm`.
    """
    obj_id = 0
    os.makedirs(out_dir, exist_ok=True)
    render_camera_model = render_camera(image_size, K, patch_size, ssaa_factor)
    render_types = [RenderType.COLOR, RenderType.DEPTH, RenderType.MASK]
    if renderer == "mesh":
        renderer_obj = renderer_builder.build(renderer_type=RENDERERS[renderer], model_path=model_path)
    else:
        renderer_obj = renderer_builder.build(renderer_type=RENDERERS[renderer])
    renderer_obj.add_object_model(obj_id=obj_id, model_path=model_path)

    views = sample_template_views(min_num_viewpoints, num_inplane_rotations, radius)
    try:
        for template_counter, view in enumerate(progress(views) if progress else views):
            trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])
            R_c2m = trans_m2c.R.T
            trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))
            render_camera_model_c2w = PinholePlaneCameraModel(
                width=render_camera_model.width,
                height=render_camera_model.height,
                f=render_camera_model.f,
                c=render_camera_model.c,
                T_world_from_eye=misc.get_rigid_matrix(trans_c2m),
            )
            output = renderer_obj.render_object_model(
                obj_id=obj_id,
                camera_model_c2w=render_camera_model_c2w,
                render_types=render_types,
                return_tensors=False,
                debug=False,
            )
            output[RenderType.MASK] = (output[RenderType.MASK] != 0).astype(np.uint8) * 255

            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(calc_2d_box(xs, ys))
            object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])
            if (
                object_box.left == 0
                or object_box.top == 0
                or object_box.right == render_camera_model_c2w.width - 1
                or object_box.bottom == render_camera_model_c2w.height - 1
            ):
                print(f"template {template_counter}: the model does not fit the viewport.")

            crop_box = calc_crop_box(box=object_box, make_square=True)
            crop_camera_model_c2w = construct_crop_camera(
                box=crop_box,
                camera_model_c2w=render_camera_model_c2w,
                viewport_size=(int(crop_size * ssaa_factor), int(crop_size * ssaa_factor)),
                viewport_rel_pad=crop_rel_pad,
            )
            for output_key in output.keys():
                if output_key in [RenderType.DEPTH]:
                    output[output_key] = warp_depth_image(
                        src_camera=render_camera_model_c2w, dst_camera=crop_camera_model_c2w, src_depth_image=output[output_key]
                    )
                elif output_key in [RenderType.COLOR]:
                    interpolation = cv2.INTER_AREA if crop_box.width >= crop_camera_model_c2w.width else cv2.INTER_LINEAR
                    output[output_key] = warp_image(
                        src_camera=render_camera_model_c2w,
                        dst_camera=crop_camera_model_c2w,
                        src_image=output[output_key],
                        interpolation=interpolation,
                    )
                else:
                    output[output_key] = warp_image(
                        src_camera=render_camera_model_c2w,
                        dst_camera=crop_camera_model_c2w,
                        src_image=output[output_key],
                        interpolation=cv2.INTER_NEAREST,
                    )

            camera_model_c2w = crop_camera_model_c2w.copy()
            scale_factor = crop_size / float(crop_camera_model_c2w.width)
            camera_model_c2w.width = crop_size
            camera_model_c2w.height = crop_size
            camera_model_c2w.c = (camera_model_c2w.c[0] * scale_factor, camera_model_c2w.c[1] * scale_factor)
            camera_model_c2w.f = (camera_model_c2w.f[0] * scale_factor, camera_model_c2w.f[1] * scale_factor)

            if ssaa_factor != 1.0:
                target_size = (camera_model_c2w.width, camera_model_c2w.height)
                for output_key in output.keys():
                    interpolation = cv2.INTER_AREA if output_key in [RenderType.COLOR] else cv2.INTER_NEAREST
                    output[output_key] = misc.resize_image(image=output[output_key], size=target_size, interpolation=interpolation)

            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(calc_2d_box(xs, ys))
            object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])

            rgb_image = np.asarray(255.0 * output[RenderType.COLOR], np.uint8)
            depth_image = output[RenderType.DEPTH].astype("uint16")
            rgb_alpha_image = np.concatenate([rgb_image, output[RenderType.MASK][:, :, None]], axis=-1)

            files = {name: os.path.join(out_dir, file) for name, file in template_files(template_counter).items()}
            inout.save_im(files["image"], rgb_alpha_image)
            inout.save_depth(files["depth"], depth_image)
            K_save = np.array(
                [[camera_model_c2w.f[0], 0, camera_model_c2w.c[0]], [0, camera_model_c2w.f[1], camera_model_c2w.c[1]], [0, 0, 1]],
                dtype=np.float32,
            )
            np.save(
                files["K_pose"],
                {
                    "K": K_save,
                    "height": camera_model_c2w.height,
                    "width": camera_model_c2w.width,
                    "pose_c_o": inverse_se3_numpy(camera_model_c2w.T_world_from_eye),
                    "box_amodal": np.array([object_box.array_ltrb()]),
                },
            )
    finally:
        pyrender_renderer = getattr(renderer_obj, "renderer", None)
        if pyrender_renderer is not None and hasattr(pyrender_renderer, "delete"):
            pyrender_renderer.delete()
    return len(views)
