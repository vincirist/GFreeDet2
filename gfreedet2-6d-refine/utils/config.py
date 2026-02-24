# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3


from enum import Enum
from typing import List, NamedTuple, Optional, Tuple


class FeatureOpts(NamedTuple):
    extractor_name: str


class TemplateDescOpts(NamedTuple):
    desc_type: str = "tfidf"

    tfidf_knn_metric: str = "l2"
    tfidf_knn_k: int = 3
    tfidf_soft_assign: bool = False
    tfidf_soft_sigma_squared: float = 10.0


class GenTemplatesOpts(NamedTuple):

    version: str
    dataset_name: str
    object_lids: Optional[List[int]] = None

    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    ssaa_factor: float = 4.0
    background_type: str = "gray"
    light_type: str = "multi_directional"

    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True


class GenRepreOpts(NamedTuple):

    version: str
    templates_version: str
    dataset_name: str
    object_lids: Optional[List[int]] = None

    extractor_name: str = "dinov2_vits14_reg"
    grid_cell_size: float = 14.0

    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    cluster_features: bool = True
    cluster_num: int = 2048

    template_desc_opts: Optional[TemplateDescOpts] = None

    overwrite: bool = True
    debug: bool = True


class PnPOpts(NamedTuple):

    pnp_ransac_iter: int = 100

    pnp_inlier_thresh: float = 2.0
    pnp_required_ransac_conf: float = 0.999
    sub_sample_correspondences: bool = True
    max_num_corresps: int = 10000

    use_iterative_pnp: bool = False  # NOTE: added; if false, use EPnP instead of iterative PnP


class GoTrackRefinerLossOpts(NamedTuple):

    distance_type: str = "l1"


class InitPoseType(Enum):

    GT_POSES = "gt_poses"
    FROM_GT_BOXES = "from_gt_boxes"
    FROM_DETECTIONS = "from_detections"
    FROM_COARSE_POSES = "from_coarse_poses"


class InitPoseOpts(NamedTuple):

    pose_type: InitPoseType = InitPoseType.GT_POSES
    noise_name: str = "cosypose"
    noise_rotation_std: List[float] = [15.0, 15.0, 15.0]
    noise_translation_std: List[float] = [15.0, 15.0, 15.0]


class PoseLossOpts(NamedTuple):

    loss_disentangled: bool = True
    loss_disentangled_rot_weight: float = 0.1
    loss_disentangled_xy_weight: float = 0.1
    loss_disentangled_z_weight: float = 0.1
    loss_add_weight: float = 1.0
    num_model_vertices: int = 500


class PoseRepre(Enum):

    XYZ_CONT6D = "xyz_cont6d"
    XYZ_QUAT = "xyz_quat"
