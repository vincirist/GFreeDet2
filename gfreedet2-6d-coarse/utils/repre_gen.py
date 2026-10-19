"""Descriptors of one object's templates: `make_extractor`, `generate_repre`.

Same features, PCA, visual words and tf-idf as `scripts/gen_repre_v2.py`, with no hydra, rootutils or dataset paths.
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from bop_toolkit_lib import inout
from utils import cluster_util, feature_util, projector_util, repre_util, template_util
from utils.misc import array_to_tensor
from utils.structs import PinholePlaneCameraModel
from utils.template_gen import template_files
from utils.transform3d import inverse_se3_numpy


def extractor_name(layer: int) -> str:
    """Name of the DINOv2 ViT-L/14 (with registers) extractor on token features of `layer`."""
    return f"dinov2_version=vitl14-reg_stride=14_facet=token_layer={layer}_logbin=0_norm=1"


def make_extractor(layer: int, device: str = "cuda") -> torch.nn.Module:
    """The feature extractor. `torch.hub` loads the weights from `$TORCH_HOME/hub/checkpoints`, or downloads them there."""
    return feature_util.make_feature_extractor(extractor_name(layer)).to(device)


def generate_raw_repre(
    template_dir: str, templates: List[Dict[str, np.ndarray]], layer: int, extractor: torch.nn.Module, grid_cell_size: float, device: str
) -> repre_util.FeatureBasedObjectRepre:
    """Features of every template registered in 3D (the template depth), in the model frame."""
    feat_vectors_list, feat_to_vertex_ids_list, vertices_in_model_list = [], [], []
    feat_to_template_ids_list, templates_list, template_cameras_list = [], [], []

    for template_id, pose_data in enumerate(tqdm(templates)):
        files = {name: os.path.join(template_dir, file) for name, file in template_files(template_id).items()}
        K = pose_data["K"]
        camera = PinholePlaneCameraModel(
            width=pose_data["width"],
            height=pose_data["height"],
            f=(K[0, 0], K[1, 1]),
            c=(K[0, 2], K[1, 2]),
            T_world_from_eye=inverse_se3_numpy(pose_data["pose_c_o"]),
        )
        image_mask = inout.load_im(files["image"])
        depth = inout.load_depth(files["depth"])

        image_chw = array_to_tensor(image_mask[:, :, :3]).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        depth_hw = array_to_tensor(depth).to(torch.float32).to(device)
        mask = array_to_tensor(image_mask[:, :, 3]).to(torch.float32).to(device)
        T_model_from_camera = array_to_tensor(camera.T_world_from_eye).to(torch.float32).to(device)

        feat_vectors, feat_to_vertex_ids, vertices_in_model = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_hw,
            object_mask=mask,
            camera=camera,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=grid_cell_size,
            debug=False,
        )
        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids_list.append(template_id * torch.ones(feat_vectors.shape[0], dtype=torch.int32, device=device))
        templates_list.append((image_chw * 255).to(torch.uint8))
        camera_model = camera.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_list.append(camera_model)

    return repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_opts=repre_util.FeatureOpts(extractor_name=extractor_name(layer)),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_list,
    )


def generate_repre(
    template_dir: str,
    templates: List[Dict[str, np.ndarray]],
    out_dir: str,
    extractor: torch.nn.Module,
    *,
    layer: int,
    grid_cell_size: float = 14.0,
    pca_components: int = 256,
    pca_whiten: bool = False,
    pca_max_samples: int = 100000,
    cluster_num: int = 2048,
    tfidf_knn_k: int = 3,
    seed: Optional[int] = 0,
    device: str = "cuda",
) -> None:
    """Compute the descriptors of the templates in `template_dir` and save `repre.pth` into `out_dir`.
    - `templates` is the list `generate_templates` returns.
    - `extractor` is `make_extractor(layer)`. `layer` only labels it in the file.
    - `seed` seeds the PCA (sample subset, solver) and the k-means; `None` draws a random seed.
    """
    os.makedirs(out_dir, exist_ok=True)
    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    repre = generate_raw_repre(template_dir, templates, layer, extractor, grid_cell_size, device)
    feat_vectors = repre.feat_vectors

    pca_projector = projector_util.PCAProjector(n_components=pca_components, whiten=pca_whiten, random_state=seed)
    pca_projector.fit(feat_vectors, max_samples=pca_max_samples, seed=seed)
    repre.feat_raw_projectors.append(pca_projector)
    feat_vectors = pca_projector.transform(feat_vectors)

    centroids, cluster_ids, _ = cluster_util.kmeans(samples=feat_vectors, num_centroids=cluster_num, verbose=True, seed=seed)
    repre.feat_cluster_centroids = centroids
    repre.feat_to_cluster_ids = cluster_ids

    repre.template_desc_opts = repre_util.TemplateDescOpts(tfidf_knn_k=tfidf_knn_k)
    repre.template_descs, repre.feat_cluster_idfs = template_util.calc_tfidf_descriptors(
        feat_vectors=feat_vectors,
        feat_words=repre.feat_cluster_centroids,
        feat_to_word_ids=repre.feat_to_cluster_ids,
        feat_to_template_ids=repre.feat_to_template_ids,
        num_templates=len(repre.templates),
        tfidf_knn_k=repre.template_desc_opts.tfidf_knn_k,
        tfidf_soft_assign=repre.template_desc_opts.tfidf_soft_assign,
        tfidf_soft_sigma_squared=repre.template_desc_opts.tfidf_soft_sigma_squared,
    )
    repre.feat_vis_projectors = [pca_projector]
    repre.feat_vectors = feat_vectors
    repre_util.save_object_repre(repre, out_dir)
