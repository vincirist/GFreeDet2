#!/usr/bin/env python3


from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import os
from pathlib import Path
import rootutils

PROJ_ROOT = rootutils.setup_root(__file__, ".project-root", pythonpath=True)
from utils.misc import set_torch_cuda_arch_list

set_torch_cuda_arch_list(all_devices=True)

import os.path as osp
import cv2

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from utils import logging_util, misc as foundpose_misc, misc, structs
from utils.transform3d import inverse_se3_numpy
from utils.structs import AlignedBox2f, PinholePlaneCameraModel

from utils.misc import warp_depth_image, warp_image
from utils import geometry, renderer_builder
from utils.renderer_base import RenderType

np.random.seed(0)
np.set_printoptions(suppress=True)

CFG_FILE = os.environ.get("CFG_FILE", osp.join(PROJ_ROOT, "configs/gen_templates/handal_gs.yaml"))
print(CFG_FILE)
if osp.isabs(CFG_FILE):
    CFG_DIR = osp.dirname(CFG_FILE)
else:
    CFG_DIR = osp.join(os.getcwd(), osp.dirname(CFG_FILE))
CFG_NAME = osp.splitext(osp.basename(CFG_FILE))[0]
print(CFG_DIR, CFG_NAME)


@hydra.main(version_base=None, config_path=CFG_DIR, config_name=CFG_NAME)
def synthesize_templates(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    datasets_path = bop_config.datasets_path

    logger = logging_util.get_logger_v2(__name__, level=logging_util.INFO if cfg.debug else logging_util.WARNING)
    timer = misc.Timer(enabled=cfg.debug)
    timer.start()

    model_tpath = cfg.model_tpath
    logger.info(f"Model template path: {model_tpath}")

    object_lids = cfg.object_lids
    bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=cfg.object_dataset)
    if object_lids is None:

        object_lids = bop_model_props["obj_ids"]

    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path, dataset_name=cfg.object_dataset, split="test"
    )

    if cfg.object_dataset == "hot3d":
        bop_camera = dict(
            im_size=(640, 480),
            depth_scale=1.0,
            K=np.array([[572.4114, 0, 320], [0, 573.57043, 240], [0, 0, 1.0]]),
        )
    else:
        bop_camera = dataset_params.get_camera_params(datasets_path=datasets_path, dataset_name=cfg.object_dataset)

    bop_camera_width = bop_camera["im_size"][0]
    bop_camera_height = bop_camera["im_size"][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = cfg.features_patch_size * int(max_image_side / cfg.features_patch_size)

    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera["K"][0, 0], bop_camera["K"][1, 1]),
        c=(
            bop_camera["K"][0, 2] - 0.5 * (bop_camera_width - image_side),
            bop_camera["K"][1, 2] - 0.5 * (bop_camera_height - image_side),
        ),
    )

    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * cfg.ssaa_factor),
        height=int(camera_model.height * cfg.ssaa_factor),
        f=(
            camera_model.f[0] * cfg.ssaa_factor,
            camera_model.f[1] * cfg.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * cfg.ssaa_factor,
            camera_model.c[1] * cfg.ssaa_factor,
        ),
    )

    render_types = [RenderType.COLOR, RenderType.DEPTH, RenderType.MASK]
    renderer_type = renderer_builder.RendererType.GSPLAT_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type)

    depth_range = bop_test_split_props["depth_range"]
    if depth_range is None:
        depth_range = (300.0, 900.0)

    min_depth = np.min(depth_range)
    max_depth = np.max(depth_range)
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(cfg.num_viewspheres)
    viewsphere_radii = []
    for depth_cell_id in range(cfg.num_viewspheres):
        viewsphere_radii.append(min_depth + (depth_cell_id + 0.5) * depth_cell_size)

    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += foundpose_misc.sample_views(
            min_n_views=cfg.min_num_viewpoints,
            radius=radius,
            mode="fibonacci",
        )[0]

    if cfg.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / cfg.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(cfg.num_inplane_rotations):
                R_inplane = geometry.rotation_matrix_numpy(inplane_angle * inplane_id, np.array([0, 0, 1]))[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )

    timer.elapsed("Time for setting up the stage")

    for object_lid in tqdm(object_lids):
        logging_util.log_heading(logger, f"Object {object_lid} from {cfg.object_dataset}")
        timer.start()

        dataset_torch_relpath = os.path.join(
            "templates",
            cfg.version,
            cfg.object_dataset,
            f"obj_{object_lid:06d}",
        )
        output_dir = os.path.join(
            bop_config.output_path,
            dataset_torch_relpath,
        )

        if os.path.exists(output_dir) and not cfg.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        config_path = os.path.join(output_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)

        model_path = model_tpath.format(obj_id=object_lid)
        renderer.add_object_model(obj_id=object_lid, model_path=model_path)

        timer.elapsed("Time for preparing object data")

        template_counter = 0
        for view_id, view in enumerate(tqdm(views)):

            for _ in range(cfg.images_per_view):

                timer.start()

                trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])

                R_c2m = trans_m2c.R.T
                trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))

                trans_c2m_matrix = misc.get_rigid_matrix(trans_c2m)
                render_camera_model_c2w = PinholePlaneCameraModel(
                    width=render_camera_model.width,
                    height=render_camera_model.height,
                    f=render_camera_model.f,
                    c=render_camera_model.c,
                    T_world_from_eye=trans_c2m_matrix,
                )

                output = renderer.render_object_model(
                    obj_id=object_lid,
                    camera_model_c2w=render_camera_model_c2w,
                    render_types=render_types,
                    return_tensors=False,
                    debug=False,
                )

                if RenderType.MASK in output:
                    output[RenderType.MASK] = (output[RenderType.MASK] != 0).astype(np.uint8) * 255

                ys, xs = output[RenderType.MASK].nonzero()
                box = np.array(foundpose_misc.calc_2d_box(xs, ys))
                object_box = AlignedBox2f(
                    left=box[0],
                    top=box[1],
                    right=box[2],
                    bottom=box[3],
                )

                if (
                    object_box.left == 0
                    or object_box.top == 0
                    or object_box.right == render_camera_model_c2w.width - 1
                    or object_box.bottom == render_camera_model_c2w.height - 1
                ):
                    logger.warning("The model does not fit the viewport.")

                if cfg.crop:

                    crop_box = foundpose_misc.calc_crop_box(
                        box=object_box,
                        make_square=True,
                    )

                    crop_camera_model_c2w = foundpose_misc.construct_crop_camera(
                        box=crop_box,
                        camera_model_c2w=render_camera_model_c2w,
                        viewport_size=(
                            int(cfg.crop_size[0] * cfg.ssaa_factor),
                            int(cfg.crop_size[1] * cfg.ssaa_factor),
                        ),
                        viewport_rel_pad=cfg.crop_rel_pad,
                    )

                    for output_key in output.keys():
                        if output_key in [RenderType.DEPTH]:
                            output[output_key] = warp_depth_image(
                                src_camera=render_camera_model_c2w,
                                dst_camera=crop_camera_model_c2w,
                                src_depth_image=output[output_key],
                            )
                        elif output_key in [RenderType.COLOR]:
                            interpolation = (
                                cv2.INTER_AREA if crop_box.width >= crop_camera_model_c2w.width else cv2.INTER_LINEAR
                            )
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
                    scale_factor = cfg.crop_size[0] / float(crop_camera_model_c2w.width)
                    camera_model_c2w.width = cfg.crop_size[0]
                    camera_model_c2w.height = cfg.crop_size[1]
                    camera_model_c2w.c = (
                        camera_model_c2w.c[0] * scale_factor,
                        camera_model_c2w.c[1] * scale_factor,
                    )
                    camera_model_c2w.f = (
                        camera_model_c2w.f[0] * scale_factor,
                        camera_model_c2w.f[1] * scale_factor,
                    )

                else:
                    camera_model_c2w = PinholePlaneCameraModel(
                        width=camera_model.width,
                        height=camera_model.height,
                        f=camera_model.f,
                        c=camera_model.c,
                        T_world_from_eye=trans_c2m,
                    )

                if cfg.ssaa_factor != 1.0:
                    target_size = (camera_model_c2w.width, camera_model_c2w.height)
                    for output_key in output.keys():
                        if output_key in [RenderType.COLOR]:
                            interpolation = cv2.INTER_AREA
                        else:
                            interpolation = cv2.INTER_NEAREST

                        output[output_key] = misc.resize_image(
                            image=output[output_key],
                            size=target_size,
                            interpolation=interpolation,
                        )

                ys, xs = output[RenderType.MASK].nonzero()
                box = np.array(foundpose_misc.calc_2d_box(xs, ys))
                object_box = AlignedBox2f(
                    left=box[0],
                    top=box[1],
                    right=box[2],
                    bottom=box[3],
                )

                rgb_image = np.asarray(255.0 * output[RenderType.COLOR], np.uint8)
                depth_image = output[RenderType.DEPTH].astype("uint16")

                rgb_alpha_image = np.concatenate([rgb_image, output[RenderType.MASK][:, :, None]], axis=-1)

                timer.elapsed("Time for template rendering")

                timer.start()
                rgb_alpha_path = os.path.join(output_dir, f"{template_counter:06d}.png")

                inout.save_im(rgb_alpha_path, rgb_alpha_image)

                depth_path = os.path.join(output_dir, f"{template_counter:06d}_depth.png")

                inout.save_depth(depth_path, depth_image)

                K_save = np.array(
                    [
                        [camera_model_c2w.f[0], 0, camera_model_c2w.c[0]],
                        [0, camera_model_c2w.f[1], camera_model_c2w.c[1]],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                pose_c_o = inverse_se3_numpy(camera_model_c2w.T_world_from_eye)
                save_info = {
                    "K": K_save,
                    "height": camera_model_c2w.height,
                    "width": camera_model_c2w.width,
                    "pose_c_o": pose_c_o,
                    "box_amodal": np.array([object_box.array_ltrb()]),
                }

                K_pose_path = os.path.join(output_dir, f"{template_counter:06d}_K_pose.npy")

                np.save(K_pose_path, save_info)

                timer.elapsed("Time for template saving")

                template_counter += 1


if __name__ == "__main__":
    synthesize_templates()
