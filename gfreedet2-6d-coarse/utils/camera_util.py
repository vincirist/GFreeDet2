"""`image_camera`: the camera model of an image from its pinhole `K`, its size and an optional lens distortion."""

from typing import Any, Dict, Optional

import numpy as np
from hand_tracking_toolkit.camera import CameraModel, model_by_name

from utils.structs import PinholePlaneCameraModel

#: Distortion model -> its `model_by_name` key in hand_tracking_toolkit.
MODELS = {"fisheye624": "CameraModelType.FISHEYE624"}


def image_camera(K: np.ndarray, width: int, height: int, distortion: Optional[Dict[str, Any]] = None) -> CameraModel:
    """The camera of a `width` x `height` image with the 3x3 `K`, at the origin.
    - Without `distortion`: `PinholePlaneCameraModel` of `K`.
    - With `distortion` `{model: fisheye624, coeffs: k0..k5, p0, p1, s0..s3}`: hand_tracking_toolkit's FISHEYE624 model with `f, cx, cy` of `K`, as `data_util.parse_chunk_cameras` builds it.
    """
    f, c = (K[0, 0], K[1, 1]), (K[0, 2], K[1, 2])
    if distortion is None:
        return PinholePlaneCameraModel(width=width, height=height, f=f, c=c)
    if distortion["model"] not in MODELS:
        raise ValueError(f"unknown distortion model {distortion['model']!r}")
    return model_by_name[MODELS[distortion["model"]]](width, height, f, c, distortion["coeffs"])
