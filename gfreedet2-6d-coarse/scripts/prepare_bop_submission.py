# Read the json files that are the output of the infer.py script.
import os
import datetime
from collections import defaultdict
import argparse
import numpy as np
import re
import rootutils

PROJ_ROOT = rootutils.setup_root(__file__, ".project-root", pythonpath=True)
import bop_toolkit_lib.config as bop_config
from bop_toolkit_lib import dataset_params
import logging
import sys
from utils import misc, json_util

FORMAT = "[%(asctime)s %(filename)s@%(lineno)d] %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout, force=True)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def safe_to_camel_case(text, upper_first=False):
    if not isinstance(text, str):
        raise TypeError("输入必须是字符串")
    if not text:
        return ""

    cleaned = re.sub(r"[-_]+", "_", text.strip())
    words = cleaned.split("_")

    def capitalize_word(w):
        return w.capitalize() if w.isupper() else w[0].upper() + w[1:].lower()

    if upper_first:
        return "".join(capitalize_word(w) for w in words)
    else:
        return words[0].lower() + "".join(capitalize_word(w) for w in words[1:])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a BOP submission file from the estimated poses saved in the json files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="handal",
        help="The object dataset name, e.g. 'handal', 'hopev2', 'hot3d', etc.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="handal_cad_dinov2lreg_18",
        help="The version of the estimated poses",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=1e-6,
        help="The score threshold for the estimated poses. Only poses with score >= score_thr are included in the submission.",
    )
    parser.add_argument(
        "--object_lids",
        type=int,
        nargs="+",
        default=None,
        help="The list of object IDs to include in the submission. If not set, all objects are included.",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    object_dataset = args.dataset
    version = args.version
    object_lids = args.object_lids
    score_thr = args.score_thr

    signature = misc.slugify(object_dataset) + "_{}".format(version)
    output_dir = os.path.join(
        bop_config.output_path,
        "inference",
        signature,
    )

    if object_lids is None:
        datasets_path = bop_config.datasets_path
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=object_dataset)
        object_lids = bop_model_props["obj_ids"]

    detection_time_per_image = {}
    run_time_per_image = defaultdict(float)
    total_run_time = defaultdict(float)

    for object_lid in object_lids:

        # Load the estimated poses from the json file
        results_path = os.path.join(output_dir, str(object_lid), "estimated-poses.json")
        logger.info("Loading estimated poses from: {}".format(results_path))

        estimated_poses = json_util.load_json(results_path)

        for estimated_pose_data in estimated_poses:
            scene_id = estimated_pose_data["scene_id"]
            img_id = estimated_pose_data["img_id"]
            obj_id = estimated_pose_data["obj_id"]
            inst_id = estimated_pose_data["inst_id"]
            det2d_time = estimated_pose_data["det2d_time"]

            detection_time_per_image[(scene_id, img_id)] = det2d_time

            run_time = 0
            for k, val in estimated_pose_data["time"].items():
                run_time += val

            run_time_per_image[(scene_id, img_id)] += run_time

        for scene_id, img_id in run_time_per_image.keys():
            total_run_time[(scene_id, img_id)] = (
                run_time_per_image[(scene_id, img_id)] + detection_time_per_image[(scene_id, img_id)]
            )

    # BOP 19 format
    lines = ["scene_id,im_id,obj_id,score,R,t,time"]
    for object_lid in object_lids:

        # Load the estimated poses from the json file
        results_path = os.path.join(output_dir, str(object_lid), "estimated-poses.json")
        logger.info("Loading estimated poses from: {}".format(results_path))

        estimated_poses = json_util.load_json(results_path)

        for estimated_pose_data in estimated_poses:
            score = estimated_pose_data["score"]
            # if float(score) < score_thr:
            #     continue
            scene_id = estimated_pose_data["scene_id"]
            img_id = estimated_pose_data["img_id"]
            obj_id = estimated_pose_data["obj_id"]
            inst_id = estimated_pose_data["inst_id"]

            R = estimated_pose_data["R"]
            t = estimated_pose_data["t"]

            run_time = total_run_time[(scene_id, img_id)]

            lines.append(
                "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                    scene_id=scene_id,
                    im_id=img_id,
                    obj_id=obj_id,
                    score=score,
                    R=" ".join(map(str, np.array(R).flatten().tolist())),
                    t=" ".join(map(str, np.array(t).flatten().tolist())),
                    time=run_time,
                )
            )

    version_str = safe_to_camel_case(version, upper_first=True)
    bop_path = os.path.join(output_dir, f"foundpose{version_str}_{object_dataset}-test.csv")
    logger.info("Saving BOP submission file to: {}".format(bop_path))
    with open(bop_path, "wb") as f:
        f.write("\n".join(lines).encode("utf-8"))


if __name__ == "__main__":

    main()
