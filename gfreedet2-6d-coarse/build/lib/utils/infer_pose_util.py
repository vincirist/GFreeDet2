#!/usr/bin/env python3

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2

import numpy as np

from utils import (
    eval_errors,
    config_util,
    json_util,
    logging_util,
    misc,
    orjson_util,
)

from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

logger = logging_util.get_logger_v2(__name__)


def load_detections_in_bop_format(path: str):

    pred_mask_list = orjson_util.load_json(path)
    detections = defaultdict(list)

    for pred in pred_mask_list:
        key = (pred["scene_id"], pred["image_id"], pred["category_id"])
        detections[key].append(
            {
                "bbox": pred["bbox"],
                "segmentation": pred["segmentation"],
                "score": pred["score"],
                "time": pred["time"],
            }
        )

    return detections


def load_detections_in_bop_format_to_per_im_dets(path: str):

    pred_mask_list = orjson_util.load_json(path)
    detections = defaultdict(list)

    for pred in pred_mask_list:
        key = (pred["scene_id"], pred["image_id"])
        detections[key].append(
            {
                "obj_id": pred["category_id"],
                "bbox": pred["bbox"],
                "segmentation": pred.get("segmentation", None),
                "score": pred["score"],
                "time": pred["time"],
            }
        )
    sorted_detections = sorted(detections.items(), key=lambda x: x[0])
    return dict(sorted_detections)


def get_instances_for_pose_estimation(
    bop_chunk_id: int,
    bop_im_id: int,
    obj_id: int,
    use_detections: bool,
    detections: Dict[Any, Any],
    max_num_preds: int,
    gt_object_annos: List[Any],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:

    instance_infos = []

    if use_detections:
        key = (bop_chunk_id, bop_im_id, obj_id)

        if key not in detections:
            logger.info(f"No detection found for {key}.")
            return []

        if len(detections[key]) == 1:

            preds = detections[key]
        else:
            if max_num_preds == -1 or max_num_preds is None:
                preds = detections[key]
            else:
                preds = sorted(detections[key], key=lambda x: x["score"], reverse=True)[:max_num_preds]

        for pred in preds:

            box_amodal = np.array(pred["bbox"])

            mask_modal = rle_to_binary_mask(pred["segmentation"]).astype(np.uint8)
            mask_size = (mask_modal.shape[1], mask_modal.shape[0])  # (w, h)

            mask_modal = cv2.morphologyEx(
                mask_modal,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )

            shift_x = 0
            if image_size[0] < mask_size[0]:
                shift_x = (mask_size[0] - image_size[0]) // 2
            elif image_size[0] > mask_size[0]:
                raise ValueError("Image is larger than mask.")
            shift_y = 0
            if image_size[1] < mask_size[1]:
                shift_y = (mask_size[1] - image_size[1]) // 2
            elif image_size[1] > mask_size[1]:
                raise ValueError("Image is larger than mask.")

            if shift_x > 0 and shift_y > 0:

                mask_modal = mask_modal[shift_y:-shift_y, shift_x:-shift_x]
            elif shift_x > 0 and shift_y == 0:
                mask_modal = mask_modal[:, shift_x:-shift_x]
            elif shift_x == 0 and shift_y > 0:
                mask_modal = mask_modal[shift_y:-shift_y, :]

            box_amodal[0] -= shift_x
            box_amodal[1] -= shift_y

            box_amodal[2] += box_amodal[0]
            box_amodal[3] += box_amodal[1]

            best_anno_id = 0
            best_anno_iou = 0
            gt_anno = None
            if len(gt_object_annos) != 0:
                for anno_id, anno in enumerate(gt_object_annos):

                    iou = eval_errors.mask_iou(mask_modal, anno.masks_modal)

                    if iou > best_anno_iou:
                        best_anno_iou = iou
                        best_anno_id = anno_id

                gt_anno = gt_object_annos[best_anno_id]

            instance_infos.append(
                {
                    "input_box_amodal": box_amodal,
                    "input_mask_modal": mask_modal,
                    "gt_anno": gt_anno,
                    "gt_iou": best_anno_iou,
                    "time": pred["time"],
                    "score": pred["score"],
                }
            )

    else:

        for anno in gt_object_annos:
            instance_infos.append(
                {
                    "input_box_amodal": anno.boxes_amodal.copy(),
                    "input_mask_modal": anno.masks_modal.copy(),
                    "gt_anno": anno,
                    "score": 1.0,
                }
            )

    return instance_infos


def get_instances_per_im_for_pose_estimation(
    bop_chunk_id: int,
    bop_im_id: int,
    use_detections: bool,
    detections: Dict[Any, Any],
    max_num_preds: int,
    gt_object_annos: List[Any],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:

    instance_infos = []

    if use_detections:
        key = (bop_chunk_id, bop_im_id)

        if key not in detections:

            return []

        if len(detections[key]) == 1:

            preds = detections[key]
        else:
            if max_num_preds == -1 or max_num_preds is None:
                preds = detections[key]
            else:
                preds = sorted(detections[key], key=lambda x: x["score"], reverse=True)[:max_num_preds]

        for pred in preds:

            obj_id = pred["obj_id"]

            box_amodal = np.array(pred["bbox"])

            if pred["segmentation"] is not None:
                mask_modal = rle_to_binary_mask(pred["segmentation"]).astype(np.uint8)
            else:

                mask_modal = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
                x_, y_, w_, h_ = box_amodal
                x_int = int(x_)
                y_int = int(y_)
                w_int = int(w_)
                h_int = int(h_)
                mask_modal[y_int : y_int + h_int, x_int : x_int + w_int] = 1
            mask_size = (mask_modal.shape[1], mask_modal.shape[0])

            mask_modal = cv2.morphologyEx(
                mask_modal,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )

            shift_x = 0
            if image_size[0] < mask_size[0]:
                shift_x = (mask_size[0] - image_size[0]) // 2
            elif image_size[0] > mask_size[0]:
                raise ValueError("Image is larger than mask.")
            shift_y = 0
            if image_size[1] < mask_size[1]:
                shift_y = (mask_size[1] - image_size[1]) // 2
            elif image_size[1] > mask_size[1]:
                raise ValueError("Image is larger than mask.")

            if shift_x > 0 and shift_y > 0:

                mask_modal = mask_modal[shift_y:-shift_y, shift_x:-shift_x]
            elif shift_x > 0 and shift_y == 0:
                mask_modal = mask_modal[:, shift_x:-shift_x]
            elif shift_x == 0 and shift_y > 0:
                mask_modal = mask_modal[shift_y:-shift_y, :]

            box_amodal[0] -= shift_x
            box_amodal[1] -= shift_y

            box_amodal[2] += box_amodal[0]
            box_amodal[3] += box_amodal[1]

            best_anno_id = 0
            best_anno_iou = 0
            gt_anno = None
            if len(gt_object_annos) != 0:
                for anno_id, anno in enumerate(gt_object_annos):

                    iou = eval_errors.mask_iou(mask_modal, anno.masks_modal)

                    if iou > best_anno_iou:
                        best_anno_iou = iou
                        best_anno_id = anno_id

                gt_anno = gt_object_annos[best_anno_id]

            instance_infos.append(
                {
                    "obj_id": obj_id,
                    "input_box_amodal": box_amodal,
                    "input_mask_modal": mask_modal,
                    "gt_anno": gt_anno,
                    "gt_iou": best_anno_iou,
                    "time": pred["time"],
                    "score": pred["score"],
                }
            )

    else:

        for anno in gt_object_annos:
            instance_infos.append(
                {
                    "obj_id": anno.lid,
                    "input_box_amodal": anno.boxes_amodal.copy(),
                    "input_mask_modal": anno.masks_modal.copy(),
                    "gt_anno": anno,
                    "score": 1.0,
                }
            )

    return instance_infos
