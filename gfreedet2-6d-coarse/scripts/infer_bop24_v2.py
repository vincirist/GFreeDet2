#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
import os.path as osp
import gc
import time

from typing import List, NamedTuple, Optional, Tuple
from setproctitle import setproctitle
import cv2
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
import torch
import rootutils

PROJ_ROOT = rootutils.setup_root(__file__, ".project-root", pythonpath=True)
from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays
from utils.transform3d import inverse_se3_numpy
from utils.orjson_util import load_json
from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc


from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    logging_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util,
    misc,
    structs,
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import warp_depth_image, warp_image


logger: logging_util.Logger = logging_util.get_logger_v2(__name__)


torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def feat_extract_by_chunk(extractor, images, chunk_size=24):
    bs = images.shape[0]
    if bs <= chunk_size:
        return extractor(images)
    else:

        num_chunks = (bs + chunk_size - 1) // chunk_size
        features = {}
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, bs)
            chunk_features = extractor(images[start:end])
            if i == 0:

                features = chunk_features
            else:

                for key in chunk_features.keys():
                    features[key] = torch.cat((features[key], chunk_features[key]), dim=0)
        return features


CFG_FILE = os.environ.get("CFG_FILE", osp.join(PROJ_ROOT, "configs/infer/handal_gs_dinov2l.yaml"))
print(CFG_FILE)
if osp.isabs(CFG_FILE):
    CFG_DIR = osp.dirname(CFG_FILE)
else:
    CFG_DIR = osp.join(os.getcwd(), osp.dirname(CFG_FILE))
CFG_NAME = osp.splitext(osp.basename(CFG_FILE))[0]
print(CFG_DIR, CFG_NAME)
setproctitle(f"infer_bop24_v2_{CFG_NAME}")


def load_scene_cameras(scene_im_ids, bop_test_split_props):

    scene_cameras = {}
    for scene_id in scene_im_ids.keys():
        eval_modality = dataset_params.get_scene_sensor_or_modality(bop_test_split_props["eval_modality"], scene_id)
        eval_sensor = dataset_params.get_scene_sensor_or_modality(bop_test_split_props["eval_sensor"], scene_id)
        tpath_keys = dataset_params.scene_tpaths_keys(eval_modality, eval_sensor, scene_id)
        im_size = dataset_params.get_im_size(
            bop_test_split_props,
            eval_modality,
            eval_sensor,
        )
        scene_cameras[scene_id] = data_util.load_chunk_cameras(
            bop_test_split_props[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id),
            im_size=im_size,
        )
    return scene_cameras


@hydra.main(version_base=None, config_path=CFG_DIR, config_name=CFG_NAME)
def infer(cfg: DictConfig) -> None:

    datasets_path = bop_config.datasets_path

    logger = logging_util.get_logger_v2(__name__, level=logging_util.INFO if cfg.debug else logging_util.WARNING)
    timer = misc_util.Timer(log=cfg.debug)
    timer.start()

    detections = {}
    assert cfg.use_detections, "must enable use_detections"
    if cfg.detection_path is not None:
        path = cfg.detection_path
        assert os.path.exists(path), f"path {path} does not exist!"
    else:
        path = os.path.join(
            datasets_path,
            "detections",
            "cnos-fastsam",
            f"cnos-fastsam_{cfg.object_dataset}-test.json",
        )
    detections = infer_pose_util.load_detections_in_bop_format_to_per_im_dets(path)

    extractor = feature_util.make_feature_extractor(cfg.extractor_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    object_lids = cfg.object_lids
    bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=cfg.object_dataset)
    if object_lids is None:

        object_lids = bop_model_props["obj_ids"]

    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path, dataset_name=cfg.object_dataset, split="test"
    )

    test_targets_path = os.path.join(bop_test_split_props["base_path"], "test_targets_bop24.json")
    logger.info(f"Loading test targets from: {test_targets_path}")
    targets = load_json(test_targets_path)

    scene_im_ids = {}
    for target in targets:
        scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])

    scene_cameras = load_scene_cameras(scene_im_ids, bop_test_split_props)

    renderer = None
    if cfg.vis_results:
        renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
        renderer = renderer_builder.build(renderer_type=renderer_type, model_path=bop_model_props["model_tpath"])

    pose_evaluator_dict = {}
    repre_dict = {}
    repre_np_dict = {}
    visual_words_knn_index_dict = {}
    template_knn_indices_dict = {}
    output_dir_dict = {}
    for object_lid in tqdm(object_lids):
        timer.start()

        version = cfg.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(cfg.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(bop_config.output_path, "inference", signature, str(object_lid))

        os.makedirs(output_dir, exist_ok=True)
        output_dir_dict[object_lid] = output_dir

        config_path = os.path.join(output_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)

        pose_evaluator_dict[object_lid] = eval_util.EvaluatorPose([object_lid])

        base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, cfg.repre_version, cfg.object_dataset, object_lid
        )
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
        )

        repre_dict[object_lid] = repre

        repre_np = repre_util.convert_object_repre_to_numpy(repre)
        repre_np_dict[object_lid] = repre_np

        visual_words_knn_index = None
        if cfg.match_template_type == "tfidf":
            visual_words_knn_index = knn_util.KNN(
                k=repre.template_desc_opts.tfidf_knn_k, metric=repre.template_desc_opts.tfidf_knn_metric  # 3  # "l2"
            )
            visual_words_knn_index.fit(repre.feat_cluster_centroids)
        visual_words_knn_index_dict[object_lid] = visual_words_knn_index

        template_knn_indices = []
        if cfg.match_feat_matching_type == "cyclic_buddies":

            for template_id in range(len(repre.template_cameras_cam_from_model)):

                tpl_feat_mask = repre.feat_to_template_ids == template_id
                tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

                template_feats = repre.feat_vectors[tpl_feat_ids]

                template_knn_index = knn_util.KNN(k=1, metric="l2")
                template_knn_index.fit(template_feats.cpu())
                template_knn_indices.append(template_knn_index)

        template_knn_indices_dict[object_lid] = template_knn_indices

        logging_util.log_heading(
            logger,
            f"Object: {object_lid}, vertices: {len(repre.vertices)}",
            style=logging_util.WHITE_BOLD,
        )
        timer.elapsed("Time for preparing object data")

    assert cfg.crop, "Grid sampling is only supported in crop mode."
    grid_size = cfg.crop_size

    grid_points_0 = feature_util.generate_grid_points(
        grid_size=grid_size,
        cell_size=cfg.grid_cell_size,
    )
    grid_points_0 = grid_points_0.to(device)

    timer.elapsed("Time for setting up the stage")

    for det_key in tqdm(detections.keys()):

        bop_chunk_id, bop_im_id = det_key

        if bop_im_id not in scene_im_ids[bop_chunk_id]:
            logger.warning(f"Skipping {det_key}, not in test targets.")
            continue

        timer.start()
        msg = f"Estimating pose of image: scene_id {bop_chunk_id}, im_id {bop_im_id} " f"dataset {cfg.object_dataset}"
        logging_util.log_heading(logger, msg, style=logging_util.BLUE_BOLD)

        orig_camera_c2w = scene_cameras[bop_chunk_id][bop_im_id]
        orig_image_size = (
            orig_camera_c2w.width,
            orig_camera_c2w.height,
        )

        instances = infer_pose_util.get_instances_per_im_for_pose_estimation(
            bop_chunk_id=bop_chunk_id,
            bop_im_id=bop_im_id,
            use_detections=True,
            detections=detections,
            max_num_preds=-1,
            gt_object_annos=[],
            image_size=orig_image_size,
        )
        if len(instances) == 0:

            continue

        timer.elapsed("Time for preparing image data")

        image = data_util.load_bop_test_image(cfg.object_dataset, bop_chunk_id, bop_im_id, bop_test_split_props)
        orig_image_np_hwc = image.astype(np.float32) / 255.0

        times = {}
        timer.start()
        crop_instances = []
        for inst_j, instance in enumerate(instances):

            obj_id = instance["obj_id"]

            pose_evaluator_dict[obj_id].detection_times[(bop_chunk_id, bop_im_id)] = instance["time"]
            det2d_time = instance["time"]
            logger.info(f"Time for 2d det: {det2d_time:.5f}s")

            if instance["gt_anno"] is not None:
                mask_iou = eval_errors.mask_iou(
                    instance["input_mask_modal"],
                    instance["gt_anno"].masks_modal,
                )
                if mask_iou < 0.05:
                    continue

            orig_mask_modal = instance["input_mask_modal"]
            orig_box_amodal = AlignedBox2f(
                left=instance["input_box_amodal"][0],
                top=instance["input_box_amodal"][1],
                right=instance["input_box_amodal"][2],
                bottom=instance["input_box_amodal"][3],
            )

            crop_box = misc_util.calc_crop_box(
                box=orig_box_amodal,
                make_square=True,
            )

            crop_rel_pad = cfg.crop_rel_pad
            logger.info(f"crop_rel_pad: {crop_rel_pad}")

            crop_camera_model_c2w = misc_util.construct_crop_camera(
                box=crop_box,
                camera_model_c2w=orig_camera_c2w,
                viewport_size=cfg.crop_size,
                viewport_rel_pad=crop_rel_pad,
                correct_fisheye=cfg.get("correct_fisheye", True),
            )

            interpolation = cv2.INTER_AREA if crop_box.width >= crop_camera_model_c2w.width else cv2.INTER_LINEAR
            image_np_hwc = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_image_np_hwc,
                interpolation=interpolation,
            )
            mask_modal = warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_mask_modal,
                interpolation=cv2.INTER_NEAREST,
            )

            ys, xs = mask_modal.nonzero()
            box = np.array(misc_util.calc_2d_box(xs, ys))
            box_amodal = AlignedBox2f(
                left=box[0],
                top=box[1],
                right=box[2],
                bottom=box[3],
            )

            camera_c2w = crop_camera_model_c2w
            cur_crop_inst = {
                "obj_id": obj_id,
                "inst_id": inst_j,
                "mask_modal": mask_modal,
                "box_amodal": box_amodal,
                "camera_c2w": camera_c2w,
                "image_np_hwc": image_np_hwc,
                "gt_anno": instance["gt_anno"],
            }
            crop_instances.append(cur_crop_inst)
        times["prep"] = timer.elapsed("Time for preparation")  # for all instances
        if len(crop_instances) == 0:

            continue

        timer.start()
        crop_im_np_hwc_list = [crop_inst["image_np_hwc"] for crop_inst in crop_instances]
        batch_crop_ims = np.array(crop_im_np_hwc_list, dtype=np.float32)  # BxHxWxC
        batch_crop_ims_tensor_bchw = array_to_tensor(batch_crop_ims).permute(0, 3, 1, 2).to(device)  # BxCxHxW
        chunk_size = cfg.get("extractor_chunk_size", 24)
        extractor_output = feat_extract_by_chunk(extractor, batch_crop_ims_tensor_bchw, chunk_size=chunk_size)
        feature_maps_bchw = extractor_output["feature_maps"]  # BxCxHxW
        times["feat_extract"] = timer.elapsed("Time for feature extraction")

        for crop_i, crop_inst in enumerate(crop_instances):

            obj_times = {}
            for k in times.keys():
                obj_times[k] = times[k] / len(crop_instances)

            obj_id = crop_inst["obj_id"]

            timer.start()
            mask_modal = crop_inst["mask_modal"]

            mask_modal_tensor = array_to_tensor(mask_modal).to(device)
            grid_points = grid_points_0.clone()  # Bx2
            query_points = feature_util.filter_points_by_mask(grid_points, mask_modal_tensor)
            if query_points.shape[0] == 0:
                query_points = grid_points.clone()

            if query_points.shape[0] > cfg.max_num_queries:
                perm = torch.randperm(query_points.shape[0])
                query_points = query_points[perm[: cfg.max_num_queries]]
                msg = "Randomly sumbsampled queries " f"({perm.shape[0]} -> {query_points.shape[0]}))"
                logging_util.log_heading(logger, msg, style=logging_util.RED_BOLD)

            feature_map_chw = feature_maps_bchw[crop_i]  # CxHxW
            query_features = feature_util.sample_feature_map_at_points(
                feature_map_chw=feature_map_chw,
                points=query_points,
                image_size=(crop_inst["image_np_hwc"].shape[1], crop_inst["image_np_hwc"].shape[0]),
            ).contiguous()
            obj_times["grid_sample"] = timer.elapsed("Time for grid sample")

            timer.start()

            repre = repre_dict[obj_id]
            if query_features.shape[1] != repre.feat_vectors.shape[1] and len(repre.feat_raw_projectors) != 0:
                query_features_proj = projector_util.project_features(
                    feat_vectors=query_features,
                    projectors=repre.feat_raw_projectors,
                ).contiguous()

                _c, _h, _w = feature_map_chw.shape
                feature_map_chw_proj = (
                    projector_util.project_features(
                        feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                        projectors=repre.feat_raw_projectors,
                    )
                    .view(_h, _w, -1)
                    .permute(2, 0, 1)
                )
            else:
                query_features_proj = query_features
                feature_map_chw_proj = feature_map_chw
            obj_times["proj"] = timer.elapsed("Time for projection")

            timer.start()
            template_knn_indices = template_knn_indices_dict[obj_id]
            visual_words_knn_index = visual_words_knn_index_dict[obj_id]

            corresp = []
            if len(query_points) != 0:
                corresp = corresp_util.establish_correspondences(
                    query_points=query_points,
                    query_features=query_features_proj,
                    object_repre=repre,
                    template_matching_type=cfg.match_template_type,
                    template_knn_indices=template_knn_indices,
                    feat_matching_type=cfg.match_feat_matching_type,
                    top_n_templates=cfg.match_top_n_templates,
                    top_k_buddies=cfg.match_top_k_buddies,
                    visual_words_knn_index=visual_words_knn_index,
                    debug=cfg.debug,
                )

            obj_times["corresp"] = timer.elapsed("Time for corresp")

            box_amodal = crop_inst["box_amodal"]
            camera_c2w = crop_inst["camera_c2w"]

            if cfg.get("run_retrieval_only", False) is True:
                assert len(corresp) == 1, "Only one template is expected."
                timer.start()

                tem_cam_c2w = repre.template_cameras_cam_from_model[corresp[0]["template_id"]]
                pose_crop_cam_from_model_inv = tem_cam_c2w.T_world_from_eye
                pose_crop_cam_from_model_template = inverse_se3_numpy(pose_crop_cam_from_model_inv)

                tq_est = pose_crop_cam_from_model_template[:3, 3:]
                final_poses = [
                    {
                        "R_m2c": pose_crop_cam_from_model_template[:3, :3],
                        "t_m2c": tq_est,
                        "corresp_id": 0,
                        "template_id": corresp[0]["template_id"],
                    }
                ]

                obj_times["retrieval_only"] = timer.elapsed("Time for retrieval only")

            else:
                timer.start()

                coarse_poses = []
                for corresp_id, corresp_curr in enumerate(corresp):

                    num_corresp = len(corresp_curr["coord_2d"])
                    if num_corresp < 6:
                        logger.info(f"Only {num_corresp} correspondences, skipping.")
                        continue

                    (
                        coarse_pose_success,
                        R_m2c_coarse,
                        t_m2c_coarse,
                        inliers_coarse,
                        quality_coarse,
                    ) = pnp_util.estimate_pose(
                        corresp=corresp_curr,
                        camera_c2w=camera_c2w,
                        pnp_type=cfg.pnp_type,
                        pnp_ransac_iter=cfg.pnp_ransac_iter,
                        pnp_inlier_thresh=cfg.pnp_inlier_thresh,
                        pnp_required_ransac_conf=cfg.pnp_required_ransac_conf,
                        pnp_refine_lm=cfg.pnp_refine_lm,
                    )

                    if coarse_pose_success:

                        coarse_poses.append(
                            {
                                "type": "coarse",
                                "R_m2c": R_m2c_coarse,
                                "t_m2c": t_m2c_coarse,
                                "corresp_id": corresp_id,
                                "quality": quality_coarse,
                                "inliers": inliers_coarse,
                            }
                        )

                best_coarse_quality = None
                best_coarse_pose_id = 0
                for coarse_pose_id, pose in enumerate(coarse_poses):
                    if best_coarse_quality is None or pose["quality"] > best_coarse_quality:
                        best_coarse_pose_id = coarse_pose_id
                        best_coarse_quality = pose["quality"]

                obj_times["pose_coarse"] = timer.elapsed("Time for coarse pose")

                timer.start()

                final_poses = []
                if cfg.final_pose_type in ["best_coarse"]:

                    if len(coarse_poses) == 0:
                        continue

                    final_pose = None

                    if cfg.final_pose_type in ["best_coarse"]:
                        final_pose = coarse_poses[best_coarse_pose_id]

                    if final_pose is not None:
                        final_poses.append(final_pose)
                else:
                    raise ValueError(f"Unknown final pose type {cfg.final_pose_type}")

                obj_times["final_select"] = timer.elapsed("Time for selecting final pose")

            if len(final_poses) > 0:

                time_per_hypothesis = sum(obj_times.values()) / len(final_poses)

                logging_util.log_heading(
                    logger,
                    f"SUMMARY - success: {len(final_poses)}, time: {time_per_hypothesis:.4f}s",
                    style=logging_util.WHITE_BOLD,
                )

            for hypothesis_id, final_pose in enumerate(final_poses):

                vis_tiles = []

                pose_m2w = None
                pose_m2w_coarse = None

                pose_est_m2c = structs.ObjectPose(R=final_pose["R_m2c"], t=final_pose["t_m2c"])
                trans_c2w = camera_c2w.T_world_from_eye

                trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))
                pose_m2w = structs.ObjectPose(R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:])

                vis_base_image = (255 * crop_inst["image_np_hwc"]).astype(np.uint8)

                best_corresp_np = tensors_to_arrays(corresp[final_pose["corresp_id"]])

                matched_template_ids = [c["template_id"] for c in corresp]
                matched_template_scores = [c["template_score"] for c in corresp]

                pose_eval_dict = None
                pose_eval_dict_coarse = None
                pose_evaluator: eval_util.EvaluatorPose = pose_evaluator_dict[obj_id]
                inst_j = crop_inst["inst_id"]
                object_lid = crop_inst["obj_id"]
                if True:
                    pose_eval_dict = pose_evaluator.update_without_anno(
                        scene_id=bop_chunk_id,
                        im_id=bop_im_id,
                        inst_id=inst_j,
                        hypothesis_id=hypothesis_id,
                        object_repre_vertices=tensor_to_array(repre.vertices),
                        obj_lid=object_lid,
                        object_pose_m2w=pose_m2w,
                        orig_camera_c2w=orig_camera_c2w,
                        camera_c2w=orig_camera_c2w,
                        time_per_inst=obj_times,
                        corresp=best_corresp_np,
                        inlier_radius=(cfg.pnp_inlier_thresh),
                        det_score=instances[inst_j]["score"],
                    )

                object_pose_m2w_gt = None
                if "gt_anno" in instances[inst_j] and instances[inst_j]["gt_anno"] is not None:
                    object_pose_m2w_gt = instances[inst_j]["gt_anno"].pose

                if cfg.vis_results and len(pose_evaluator.R) % cfg.get("vis_freq", 1) == 0:

                    assert renderer is not None

                    matched_template_ids = [c["template_id"] for c in corresp]
                    matched_template_scores = [c["template_score"] for c in corresp]

                    timer.start()
                    vis_tiles += vis_util.vis_inference_results(
                        base_image=vis_base_image,
                        object_repre=repre_np_dict[object_lid],
                        object_lid=object_lid,
                        object_pose_m2w=pose_m2w,
                        object_pose_m2w_gt=object_pose_m2w_gt,
                        feature_map_chw=feature_map_chw,
                        feature_map_chw_proj=feature_map_chw_proj,
                        vis_feat_map=cfg.vis_feat_map,
                        object_box=box_amodal.array_ltrb(),
                        object_mask=mask_modal,
                        camera_c2w=camera_c2w,
                        corresp=best_corresp_np,
                        matched_template_ids=matched_template_ids,
                        matched_template_scores=matched_template_scores,
                        best_template_ind=final_pose["corresp_id"],
                        renderer=renderer,
                        pose_eval_dict=pose_eval_dict,
                        corresp_top_n=cfg.vis_corresp_top_n,
                        inlier_thresh=(cfg.pnp_inlier_thresh),
                        object_pose_m2w_coarse=pose_m2w_coarse,
                        pose_eval_dict_coarse=pose_eval_dict_coarse,
                        vis_for_paper=cfg.vis_for_paper,
                        extractor=extractor,
                    )
                    timer.elapsed("Time for visualization")

                if len(vis_tiles):
                    if repre.feat_vis_projectors[0].pca.n_components == 12:
                        pca_tiles = np.vstack(vis_tiles[1:5])
                        vis_tiles = np.vstack([vis_tiles[0]] + vis_tiles[5:])
                        vis_grid = np.hstack([vis_tiles, pca_tiles])
                    else:
                        vis_grid = np.vstack(vis_tiles)
                    ext = ".png" if cfg.vis_for_paper else ".jpg"
                    vis_path = os.path.join(
                        output_dir_dict[object_lid],
                        f"{bop_chunk_id}_{bop_im_id}_{object_lid}_{inst_j}_{hypothesis_id}{ext}",
                    )
                    inout.save_im(vis_path, vis_grid)

                    if cfg.debug:
                        pts_path = os.path.join(
                            output_dir_dict[object_lid],
                            f"{bop_chunk_id}_{bop_im_id}_{object_lid}_{inst_j}_{hypothesis_id}_vertice_error.ply",
                        )

                        assert renderer is not None
                        vis_util.vis_pointcloud_error(
                            repre_np_dict[object_lid],
                            pose_m2w,
                            object_pose_m2w_gt,
                            camera_c2w,
                            0,
                            pts_path,
                        )

    if device == "cuda":

        torch.cuda.empty_cache()
        gc.collect()

    if cfg.save_estimates:
        for obj_id in object_lids:
            pose_evaluator = pose_evaluator_dict[obj_id]
            output_dir = output_dir_dict[obj_id]
            results_path = os.path.join(output_dir, "estimated-poses.json")

            pose_evaluator.save_results_json(results_path)


if __name__ == "__main__":
    infer()
