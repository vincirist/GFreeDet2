#!/usr/bin/env python3

import math

from utils import misc
import numpy as np

from scipy.spatial.transform import Rotation
from bop_toolkit_lib import misc as bop_misc


def mssd(R_est, t_est, R_gt, t_gt, pts, syms):

    pts_est = bop_misc.transform_pts_Rt(pts, R_est, t_est)
    es = []
    es_ind = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = bop_misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        err = np.linalg.norm(pts_gt_sym - pts_est, axis=1)
        es.append(err.max())
        es_ind.append(err.argmax())
    return min(es), es_ind[np.argmin(np.array(es))]


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):

    proj_est = bop_misc.project_pts(pts, K, R_est, t_est)
    es = []
    es_ind = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = bop_misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        err = np.linalg.norm(proj_est - proj_gt_sym, axis=1)
        es.append(err.max())
        es_ind.append(err.argmax())
    return min(es), es_ind[np.argmin(np.array(es))]


def re(R_est, R_gt) -> float:

    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error


def compute_rotation_error(R_est, R_gt):

    R = R_est.dot(R_gt.T)
    angles = Rotation.from_matrix(R).magnitude()

    return np.rad2deg(angles)


def compute_translation_errors(t_est, t_gt):

    err = np.abs(t_est - t_gt)
    return np.linalg.norm(err, axis=-1)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):

    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_count = float(union.sum())
    if union_count > 0:
        return inter.sum() / union_count
    else:
        return 0.0
