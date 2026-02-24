# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import copy
from functools import partial
import os
from pathlib import Path
from typing import List, Optional, Tuple
import time

import cv2
import numpy as np
import torch
import multiprocessing
from bop_toolkit_lib import inout, dataset_params
from utils import (
    crop_generation,
    im_util,
    knn_util,
    repre_util,
    logging_util,
    misc,
    structs,
    config,
    renderer_base,
    renderer_builder,
    torch_helpers,
    transform3d,
    json_util,
)

logger = logging_util.get_logger(__name__)


def find_nearest_object_features(
    query_features: torch.Tensor,
    knn_index: knn_util.KNN,
) -> Tuple[torch.Tensor, torch.Tensor]:

    nn_dists, nn_ids = knn_index.search(query_features)

    knn_k = nn_dists.shape[1]

    nn_dists = nn_dists[:, :knn_k]
    nn_ids = nn_ids[:, :knn_k]

    nn_dists = torch.sqrt(nn_dists)

    return nn_ids, nn_dists


def calc_tfidf(
    feature_word_ids: torch.Tensor,
    feature_word_dists: torch.Tensor,
    word_idfs: torch.Tensor,
    soft_assignment: bool = True,
    soft_sigma_squared: float = 100.0,
) -> torch.Tensor:

    device = feature_word_ids.device

    if soft_assignment:
        word_weights = torch.exp(-torch.square(feature_word_dists) / (2.0 * soft_sigma_squared))
    else:
        word_weights = torch.ones_like(feature_word_dists)

    word_weights = torch.nn.functional.normalize(word_weights, p=2, dim=1).reshape(-1)

    tf = word_weights / feature_word_ids.shape[0]

    feature_word_ids_flat = feature_word_ids.reshape(-1)
    idf = word_idfs[feature_word_ids_flat]

    tfidf = torch.multiply(tf, idf)

    num_words = word_idfs.shape[0]
    tfidf_desc = torch.zeros(num_words, dtype=word_weights.dtype, device=device).scatter_add_(
        dim=0, index=feature_word_ids_flat.to(torch.int64), src=tfidf
    )

    return tfidf_desc


def calc_tfidf_descriptors(
    feat_vectors: torch.Tensor,
    feat_to_word_ids: torch.Tensor,
    feat_to_template_ids: torch.Tensor,
    feat_words: torch.Tensor,
    num_templates: int,
    tfidf_knn_k: int,
    tfidf_soft_assign: bool,
    tfidf_soft_sigma_squared: float,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = feat_words.device.type

    word_occurances = torch.zeros(len(feat_words), dtype=torch.int64, device=device)
    for template_id in range(num_templates):
        mask = feat_to_template_ids == template_id
        unique_word_ids = torch.unique(feat_to_word_ids[mask])
        word_occurances[unique_word_ids] += 1
    word_idfs = torch.log(torch.as_tensor(float(num_templates)) / word_occurances.to(torch.float32))

    feat_knn_index = knn_util.KNN(k=tfidf_knn_k, metric="l2")
    feat_knn_index.fit(feat_words.cpu())

    tfidf_descs = []
    for template_id in range(num_templates):
        tpl_mask = feat_to_template_ids == template_id
        word_dists, word_ids = feat_knn_index.search(feat_vectors[tpl_mask])
        tfidf = calc_tfidf(
            feature_word_ids=word_ids,
            feature_word_dists=word_dists,
            word_idfs=word_idfs,
            soft_assignment=tfidf_soft_assign,
            soft_sigma_squared=tfidf_soft_sigma_squared,
        )
        tfidf_descs.append(tfidf)
    tfidf_descs = torch.stack(tfidf_descs, dim=0)

    return tfidf_descs, word_idfs


def tfidf_matching(
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    top_n_templates: int,
    visual_words_knn_index: knn_util.KNN,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if object_repre.template_desc_opts is None or object_repre.template_desc_opts.desc_type != "tfidf":
        raise ValueError(f"Template descriptors need to be tfidf, not {object_repre.template_desc_opts.desc_type}.")

    timer = misc.Timer(enabled=debug)
    timer.start()

    word_ids, word_dists = find_nearest_object_features(
        query_features=query_features,
        knn_index=visual_words_knn_index,
    )

    timer.elapsed("Time for KNN search")

    assert object_repre.feat_cluster_idfs is not None
    assert object_repre.template_desc_opts is not None
    query_tfidf = calc_tfidf(
        feature_word_ids=word_ids,
        feature_word_dists=word_dists,
        word_idfs=object_repre.feat_cluster_idfs,
        soft_assignment=object_repre.template_desc_opts.tfidf_soft_assign,
        soft_sigma_squared=object_repre.template_desc_opts.tfidf_soft_sigma_squared,
    )

    assert object_repre.template_descs is not None
    num_templates = object_repre.template_descs.shape[0]
    assert object_repre.template_descs is not None
    match_feat_cos_sims = torch.nn.functional.cosine_similarity(
        object_repre.template_descs, query_tfidf.tile(num_templates, 1)
    )

    template_scores, template_ids = torch.topk(match_feat_cos_sims, k=top_n_templates, sorted=True)

    return template_ids, template_scores


def template_matching(
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    top_n_templates: int,
    matching_type: str,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
    debug: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if matching_type == "tfidf":
        assert visual_words_knn_index is not None
        template_ids, template_scores = tfidf_matching(
            query_features=query_features,
            object_repre=object_repre,
            top_n_templates=top_n_templates,
            visual_words_knn_index=visual_words_knn_index,
        )

    else:
        raise ValueError(f"Unknown matching type '{matching_type}'.")
    if debug:
        logger.info(f"Matched templates: {list(torch_helpers.tensor_to_array(template_ids))}")
    return template_ids, template_scores


def templates_exists(bop_root_dir: Path, opts: config.GenTemplatesOpts, num_views: int) -> bool:

    exists = True
    for object_lid in opts.object_lids:
        dataset_torch_relpath = os.path.join(
            "templates",
            opts.version,
            opts.dataset_name,
            str(object_lid),
        )
        output_dir = os.path.join(str(bop_root_dir), dataset_torch_relpath)

        if not os.path.exists(output_dir):
            logger.info(f"Output directory does not exist: {output_dir}")
            exists = False
            break
        else:
            for view_id in range(num_views):
                rgb_path = os.path.join(output_dir, "rgb", f"template_{view_id:04d}.png")
                depth_path = os.path.join(output_dir, "depth", f"template_{view_id:04d}.png")
                mask_path = os.path.join(output_dir, "mask", f"template_{view_id:04d}.png")

                if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(mask_path)):
                    logger.info(f"Output files do not exist: {rgb_path}, {depth_path}, {mask_path}")
                    exists = False
                    break
    logger.info(f"Templates exist: {exists}, num_views: {num_views}, object_lids: {opts.object_lids}")
    return exists


def synthesize_templates(bop_root_dir: Path, opts: config.GenTemplatesOpts) -> None:
    datasets_path = bop_root_dir

    np.random.seed(0)

    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    object_lids = opts.object_lids
    bop_model_props = dataset_params.get_model_params(datasets_path=bop_root_dir, dataset_name=opts.dataset_name)
    if object_lids is None:
        object_lids = bop_model_props["obj_ids"]

    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=bop_root_dir, dataset_name=opts.dataset_name, split="test"
    )

    depth_range = bop_test_split_props["depth_range"]
    if depth_range is None:
        logger.info("WARNING: Depth range is None. Using default values (from LM-O dataset).")
        depth_range = (346.31, 1499.84)
    min_depth = np.min(depth_range)
    max_depth = np.max(depth_range)
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(opts.num_viewspheres)
    viewsphere_radii = []
    for depth_cell_id in range(opts.num_viewspheres):
        viewsphere_radii.append(min_depth + (depth_cell_id + 0.5) * depth_cell_size)

    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += misc.sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=radius,
            mode="fibonacci",
        )[0]
    logger.info(f"Sampled points on the sphere: {len(views_sphere)}")

    if opts.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / opts.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(opts.num_inplane_rotations):
                R_inplane = transform3d.rotation_matrix_numpy(inplane_angle * inplane_id, np.array([0, 0, 1]))[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )
    logger.info(f"Number of views: {len(views)}")

    template_already_exists = templates_exists(bop_root_dir=bop_root_dir, opts=opts, num_views=len(views))
    if opts.overwrite is False and template_already_exists:
        logger.info("Templates already exist. Exiting.")
        return
    else:
        logger.info(f"opts.overwrite={opts.overwrite}, template_already_exists={template_already_exists}")

    camera_name = opts.dataset_name
    if camera_name == "hot3d":
        camera_name = "lmo"
    bop_camera = dataset_params.get_camera_params(datasets_path=bop_root_dir, dataset_name=camera_name)

    logger.info("Bop camera details are read ")

    logger.info(f"Object lids: {object_lids}")

    bop_camera_width = bop_camera["im_size"][0]
    bop_camera_height = bop_camera["im_size"][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = opts.features_patch_size * int(max_image_side / opts.features_patch_size)
    camera_model = structs.PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera["K"][0, 0], bop_camera["K"][1, 1]),
        c=(
            bop_camera["K"][0, 2] - 0.5 * (bop_camera_width - image_side),
            bop_camera["K"][1, 2] - 0.5 * (bop_camera_height - image_side),
        ),
    )
    render_camera_model = structs.PinholePlaneCameraModel(
        width=int(camera_model.width * opts.ssaa_factor),
        height=int(camera_model.height * opts.ssaa_factor),
        f=(
            camera_model.f[0] * opts.ssaa_factor,
            camera_model.f[1] * opts.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * opts.ssaa_factor,
            camera_model.c[1] * opts.ssaa_factor,
        ),
    )
    logger.info("camera model created")

    render_types = [
        renderer_base.RenderType.COLOR,
        renderer_base.RenderType.DEPTH,
        renderer_base.RenderType.MASK,
    ]
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(
        renderer_type=renderer_type,
    )

    timer.elapsed("Time for setting up the stage")

    for object_lid in object_lids:
        logger.info(f"Object {object_lid} from {opts.dataset_name}")
        timer.start()

        dataset_torch_relpath = os.path.join(
            "templates",
            opts.version,
            opts.dataset_name,
            str(object_lid),
        )
        output_dir = os.path.join(
            datasets_path,
            dataset_torch_relpath,
        )
        logger.info(f"output_dir: {datasets_path}")
        if os.path.exists(output_dir) and not opts.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output will be saved to: {output_dir}")

        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        templates_rgb_dir = os.path.join(output_dir, "rgb")
        if opts.save_templates:
            os.makedirs(templates_rgb_dir, exist_ok=True)

        templates_depth_dir = os.path.join(output_dir, "depth")
        if opts.save_templates:
            os.makedirs(templates_depth_dir, exist_ok=True)

        templates_mask_dir = os.path.join(output_dir, "mask")
        if opts.save_templates:
            os.makedirs(templates_mask_dir, exist_ok=True)

        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        renderer.add_object_model(
            obj_id=object_lid,
            model_path=model_path,
            debug=True,
            background=[0.5, 0.5, 0.5] if opts.background_type == "gray" else None,
        )

        metadata_list = []

        timer.elapsed("Time for preparing object data")

        template_list = []
        template_counter = 0
        for view_id, view in enumerate(views):
            logger.info(f"Rendering object {object_lid} from {opts.dataset_name}, view {view_id}/{len(views)}...")
            for _ in range(opts.images_per_view):
                timer.start()

                trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])

                R_c2m = trans_m2c.R.T
                trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))

                trans_c2m_matrix = misc.get_rigid_matrix(trans_c2m)
                render_camera_model_c2w = structs.PinholePlaneCameraModel(
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
                    background=[0.5, 0.5, 0.5, 1.0] if opts.background_type == "gray" else None,
                )

                if renderer_base.RenderType.MASK in output:
                    output[renderer_base.RenderType.MASK] = (255 * output[renderer_base.RenderType.MASK]).astype(
                        np.uint8
                    )

                ys, xs = output[renderer_base.RenderType.MASK].nonzero()
                box = np.array(im_util.calc_2d_box(xs, ys))
                object_box = structs.AlignedBox2f(
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
                    raise ValueError("The model does not fit the viewport.")

                if opts.crop:
                    crop_box = im_util.calc_crop_box(
                        box=object_box,
                        make_square=True,
                    )

                    crop_camera_model_c2w = crop_generation.construct_crop_camera(
                        box=crop_box,
                        camera_model_c2w=render_camera_model_c2w,
                        viewport_size=(
                            int(opts.crop_size[0] * opts.ssaa_factor),
                            int(opts.crop_size[1] * opts.ssaa_factor),
                        ),
                        viewport_rel_pad=opts.crop_rel_pad,
                    )

                    for output_key in output.keys():
                        if output_key in [renderer_base.RenderType.DEPTH]:
                            output[output_key] = im_util.warp_depth_image(
                                src_camera=render_camera_model_c2w,
                                dst_camera=crop_camera_model_c2w,
                                src_depth_image=output[output_key],
                            )
                        elif output_key in [renderer_base.RenderType.COLOR]:
                            interpolation = (
                                cv2.INTER_AREA if crop_box.width >= crop_camera_model_c2w.width else cv2.INTER_LINEAR
                            )
                            output[output_key] = im_util.warp_image(
                                src_camera=render_camera_model_c2w,
                                dst_camera=crop_camera_model_c2w,
                                src_image=output[output_key],
                                interpolation=interpolation,
                            )
                        else:
                            output[output_key] = im_util.warp_image(
                                src_camera=render_camera_model_c2w,
                                dst_camera=crop_camera_model_c2w,
                                src_image=output[output_key],
                                interpolation=cv2.INTER_NEAREST,
                            )

                    camera_model_c2w = crop_camera_model_c2w.copy()
                    scale_factor = opts.crop_size[0] / float(crop_camera_model_c2w.width)
                    camera_model_c2w.width = opts.crop_size[0]
                    camera_model_c2w.height = opts.crop_size[1]
                    camera_model_c2w.c = (
                        camera_model_c2w.c[0] * scale_factor,
                        camera_model_c2w.c[1] * scale_factor,
                    )
                    camera_model_c2w.f = (
                        camera_model_c2w.f[0] * scale_factor,
                        camera_model_c2w.f[1] * scale_factor,
                    )

                else:
                    camera_model_c2w = structs.PinholePlaneCameraModel(
                        width=camera_model.width,
                        height=camera_model.height,
                        f=camera_model.f,
                        c=camera_model.c,
                        T_world_from_eye=trans_c2w,  # noqa: F821 # type: ignore
                    )

                if opts.ssaa_factor != 1.0:
                    target_size = (camera_model_c2w.width, camera_model_c2w.height)
                    for output_key in output.keys():
                        if output_key in [renderer_base.RenderType.COLOR]:
                            interpolation = cv2.INTER_AREA
                        else:
                            interpolation = cv2.INTER_NEAREST

                        output[output_key] = im_util.resize_image(
                            image=output[output_key],
                            size=target_size,
                            interpolation=interpolation,
                        )

                template_list.append(
                    {
                        "seq_id": template_counter,
                    }
                )

                trans_m2w = structs.RigidTransform(R=np.eye(3), t=np.zeros((3, 1)))

                visibility = 1.0

                ys, xs = output[renderer_base.RenderType.MASK].nonzero()
                box = np.array(im_util.calc_2d_box(xs, ys))
                object_box = structs.AlignedBox2f(
                    left=box[0],
                    top=box[1],
                    right=box[2],
                    bottom=box[3],
                )

                rgb_image = np.asarray(255.0 * output[renderer_base.RenderType.COLOR], np.uint8)
                depth_image = output[renderer_base.RenderType.DEPTH]

                timer.elapsed("Time for template generation")

                timer.start()
                rgb_path = os.path.join(templates_rgb_dir, f"template_{template_counter:04d}.png")
                logger.info(f"Saving template RGB {template_counter} to: {rgb_path}")
                inout.save_im(rgb_path, rgb_image)

                depth_path = os.path.join(templates_depth_dir, f"template_{template_counter:04d}.png")
                logger.info(f"Saving template depth map {template_counter} to: {depth_path}")
                inout.save_depth(depth_path, depth_image)

                mask_path = os.path.join(templates_mask_dir, f"template_{template_counter:04d}.png")
                logger.info(f"Saving template binary mask {template_counter} to: {mask_path}")
                inout.save_im(mask_path, output[renderer_base.RenderType.MASK])

                data = {
                    "dataset": opts.dataset_name,
                    "lid": object_lid,
                    "template_id": template_counter,
                    "pose": trans_m2w,
                    "boxes_amodal": np.array([object_box.array_ltrb()]).tolist(),
                    "visibilities": np.array([visibility]).tolist(),
                    "cameras": camera_model_c2w.to_json(),
                    "rgb_image_path": rgb_path,
                    "depth_map_path": depth_path,
                    "binary_mask_path": mask_path,
                }
                timer.elapsed("Time for template saving")

                metadata_list.append(data)

                template_counter += 1

        metadata_path = os.path.join(output_dir, "metadata.json")
        json_util.save_json(metadata_path, metadata_list)


def synthesize_templates_process(idx: int, bop_root_dir: Path, list_opts: List[config.GenTemplatesOpts]) -> None:
    opts = list_opts[idx]
    logger.info(f"Generating templates for object {opts.object_lids[0]}")
    synthesize_templates(bop_root_dir=bop_root_dir, opts=opts)


def batch_synthesize_templates(bop_root_dir: Path, opts: config.GenTemplatesOpts, num_workers: int = 10) -> None:

    list_opts = []
    for object_lid in opts.object_lids:
        sub_opts = copy.deepcopy(opts)
        sub_opts.object_lids = [object_lid]
        list_opts.append(sub_opts)

    synthesize_templates_worker = partial(
        synthesize_templates_process,
        bop_root_dir=bop_root_dir,
        list_opts=list_opts,
    )

    pool = multiprocessing.Pool(processes=num_workers)
    start_time = time.time()
    synthesize_templates_worker = partial(
        synthesize_templates_process,
        bop_root_dir=bop_root_dir,
        list_opts=list_opts,
    )
    pool.map(synthesize_templates_worker, range(len(list_opts)))
    pool.close()
    pool.join()
    end_time = time.time()
    run_time_per_object = (end_time - start_time) / len(list_opts)
    logger.info(f"Templage generation: run time per object: {run_time_per_object:.2f} seconds")
    logger.info("All templates generated.")
