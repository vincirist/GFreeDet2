#!/usr/bin/env python3

import os
import os.path as osp
import logging

from typing import Any, Dict, List, NamedTuple, Optional

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
import rootutils

PROJ_ROOT = rootutils.setup_root(__file__, ".project-root", pythonpath=True)
from utils.misc import array_to_tensor
from utils.transform3d import inverse_se3_numpy

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from utils import (
    cluster_util,
    feature_util,
    logging_util,
    projector_util,
    repre_util,
    template_util,
    preprocess_util,
    config_util,
    json_util,
    misc,
    structs,
)

from utils.structs import PinholePlaneCameraModel


logger = logging_util.get_logger_v2(__name__)


def generate_raw_repre(
    opts: DictConfig,
    object_dataset: str,
    object_lid: int,
    extractor: torch.nn.Module,
    output_dir: str,
    device: str = "cuda",
    debug: bool = False,
) -> repre_util.FeatureBasedObjectRepre:

    timer = misc.Timer(enabled=debug)

    obj_tem_dir = os.path.join(
        bop_config.output_path, "templates", opts.templates_version, opts.object_dataset, f"obj_{object_lid:06d}"
    )
    obj_tem_K_pose_names = [fn for fn in os.listdir(obj_tem_dir) if fn.endswith("_K_pose.npy")]
    obj_tem_K_pose_names = sorted(obj_tem_K_pose_names, key=lambda x: int(x.split("_")[0]))

    feat_vectors_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_cameras_cam_from_model_list = []

    template_id = 0
    num_templates = len(obj_tem_K_pose_names)
    for data_id, K_pose_name in enumerate(tqdm(obj_tem_K_pose_names)):
        logger.info(f"Processing dataset {opts.object_dataset} {object_lid} {data_id}/{num_templates}, ")

        timer.start()

        tem_id = int(K_pose_name.split("_")[0])

        K_pose_path = os.path.join(obj_tem_dir, K_pose_name)
        pose_data = np.load(K_pose_path, allow_pickle=True).item()
        cam_intrin = pose_data["K"]
        im_w = pose_data["width"]
        im_h = pose_data["height"]
        pose_c_o = pose_data["pose_c_o"]

        T_w_from_c = inverse_se3_numpy(pose_c_o)

        camera_world_from_cam = PinholePlaneCameraModel(
            width=im_w,
            height=im_h,
            f=(cam_intrin[0, 0], cam_intrin[1, 1]),
            c=(cam_intrin[0, 2], cam_intrin[1, 2]),
            T_world_from_eye=T_w_from_c,
        )

        image_mask_path = os.path.join(obj_tem_dir, f"{tem_id:06d}.png")
        depth_path = os.path.join(obj_tem_dir, f"{tem_id:06d}_depth.png")

        image_mask_arr = inout.load_im(image_mask_path)
        depth_image_arr = inout.load_depth(depth_path)

        image_arr = image_mask_arr[:, :, :3]
        mask_arr = image_mask_arr[:, :, 3]
        image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        depth_image_hw = array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(mask_arr).to(torch.float32).to(device)

        object_pose_rigid_matrix = np.eye(4)
        T_world_from_model = array_to_tensor(object_pose_rigid_matrix).to(torch.float32).to(device)
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = array_to_tensor(camera_world_from_cam.T_world_from_eye).to(torch.float32).to(device)
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        timer.elapsed("Time for getting template data")
        timer.start()

        (
            feat_vectors,
            feat_to_vertex_ids,
            vertices_in_model,
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera_world_from_cam,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )

        timer.elapsed("Time for feature extraction")
        timer.start()

        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids = template_id * torch.ones(feat_vectors.shape[0], dtype=torch.int32, device=device)
        feat_to_template_ids_list.append(feat_to_template_ids)

        image_chw_uint8 = (image_chw * 255).to(torch.uint8)
        templates_list.append(image_chw_uint8)

        camera_model = camera_world_from_cam.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

        template_id += 1

        timer.elapsed("Time for storing data")

    logger.info("Processing done.")

    return repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_cam_from_model_list,
    )


def generate_repre(
    opts: DictConfig,
    dataset: str,
    lid: int,
    device: str = "cuda",
    extractor: Optional[torch.nn.Module] = None,
) -> None:
    """for a single object"""
    logger = logging_util.get_logger_v2(__name__, level=logging_util.INFO if opts.debug else logging_util.WARNING)

    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
    output_dir = repre_util.get_object_repre_dir_path(base_repre_dir, opts.version, dataset, lid)
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    OmegaConf.save(opts, os.path.join(output_dir, "config.yaml"))

    if extractor is None:
        extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)

    timer.elapsed("Time for preparation")
    timer.start()

    repre = generate_raw_repre(
        opts=opts,
        object_dataset=dataset,
        object_lid=lid,
        extractor=extractor,
        output_dir=output_dir,
        device=device,
    )

    feat_vectors = repre.feat_vectors
    assert feat_vectors is not None

    timer.elapsed("Time for generating raw representation")

    if opts.apply_pca:
        timer.start()

        logger.info("Preparing PCA...")
        pca_projector = projector_util.PCAProjector(n_components=opts.pca_components, whiten=opts.pca_whiten)
        pca_projector.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_raw_projectors.append(pca_projector)

        feat_vectors = pca_projector.transform(feat_vectors)

        timer.elapsed("Time for PCA")

    if opts.cluster_features:
        timer.start()

        logger.info(f"Clustering features into {opts.cluster_num} visual words...")
        centroids, cluster_ids, centroid_distances = cluster_util.kmeans(
            samples=feat_vectors,
            num_centroids=opts.cluster_num,
            verbose=True,
        )

        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids

        unique_ids, unique_counts = torch.unique(cluster_ids, return_counts=True)

        timer.elapsed("Time for feature clustering")
        logging_util.log_heading(
            logger,
            f"{feat_vectors.shape[0]} feature vectors were clustered into {len(centroids)} clusters "
            f"with {unique_counts.min()} to {unique_counts.max()} elements.",
        )

    if opts.template_desc_opts is not None:
        timer.start()

        repre.template_desc_opts = instantiate(opts.template_desc_opts)

        if opts.template_desc_opts.desc_type == "tfidf":

            assert feat_vectors is not None
            assert repre.feat_cluster_centroids is not None
            assert repre.feat_to_cluster_ids is not None
            assert repre.feat_to_template_ids is not None
            assert repre.templates is not None

            repre.template_descs, repre.feat_cluster_idfs = template_util.calc_tfidf_descriptors(
                feat_vectors=feat_vectors,
                feat_words=repre.feat_cluster_centroids,
                feat_to_word_ids=repre.feat_to_cluster_ids,
                feat_to_template_ids=repre.feat_to_template_ids,
                num_templates=len(repre.templates),
                tfidf_knn_k=opts.template_desc_opts.tfidf_knn_k,
                tfidf_soft_assign=opts.template_desc_opts.tfidf_soft_assign,
                tfidf_soft_sigma_squared=opts.template_desc_opts.tfidf_soft_sigma_squared,
            )

        else:
            raise ValueError(f"Unknown template descriptor type: {opts.template_desc_opts.desc_type}")

        timer.elapsed("Time for generating template descriptors")

    timer.start()

    if len(repre.feat_raw_projectors) and isinstance(repre.feat_raw_projectors[0], projector_util.PCAProjector):
        repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]
    else:

        num_pca_dims_vis = 3
        pca_projector_vis = projector_util.PCAProjector(n_components=num_pca_dims_vis, whiten=False)
        pca_projector_vis.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_vis_projectors = [pca_projector_vis]

    repre.feat_vectors = feat_vectors

    timer.elapsed("Time for finding PCA for visualizations")
    timer.start()

    repre_dir = repre_util.get_object_repre_dir_path(base_repre_dir, opts.version, dataset, lid)
    repre_util.save_object_repre(repre, repre_dir)

    timer.elapsed("Time for saving the object representation")


CFG_FILE = os.environ.get("CFG_FILE", osp.join(PROJ_ROOT, "configs/gen_repre/handal_gsFixed_dinov2l.yaml"))
print(CFG_FILE)
if osp.isabs(CFG_FILE):
    CFG_DIR = osp.dirname(CFG_FILE)
else:
    CFG_DIR = osp.join(os.getcwd(), osp.dirname(CFG_FILE))
CFG_NAME = osp.splitext(osp.basename(CFG_FILE))[0]
print(CFG_DIR, CFG_NAME)


@hydra.main(version_base=None, config_path=CFG_DIR, config_name=CFG_NAME)
def generate_repre_from_list(opts: DictConfig):
    OmegaConf.set_struct(opts, False)

    object_lids = opts.object_lids
    if object_lids is None:
        datasets_path = bop_config.datasets_path
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
        object_lids = bop_model_props["obj_ids"]

    extractor = feature_util.make_feature_extractor(opts.extractor_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    for object_lid in tqdm(object_lids):
        generate_repre(opts, opts.object_dataset, object_lid, device, extractor)


if __name__ == "__main__":
    generate_repre_from_list()
