# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from enum import Enum
from typing import Any, Dict, Optional, Sequence

import numpy as np
import trimesh
from utils import structs, logging_util

logger = logging_util.get_logger(__name__)


def get_single_model_color(mesh: trimesh.Trimesh) -> structs.Color:

    try:
        return tuple(np.mean(mesh.visual.vertex_colors[:, :3], axis=0) / 255.0)
    except AttributeError:
        return (0.0, 1.0, 0.0)


class RenderType(Enum):

    COLOR = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    MASK = "mask"


class RendererBase:

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError()

    def add_object_model(
        self,
        obj_id: int,
        mesh_color: Optional[structs.Color] = None,
        **kwargs: Any,
    ) -> None:

        raise NotImplementedError()

    def render_object_model(
        self,
        obj_id: int,
        camera_model_c2m: structs.CameraModel,
        render_types: Sequence[RenderType],
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:

        raise NotImplementedError()

    def render_meshes(
        self,
        meshes_in_w: Sequence[trimesh.Trimesh],
        camera_model_c2w: structs.CameraModel,
        render_types: Sequence[RenderType],
        mesh_colors: Optional[Sequence[structs.Color]] = None,
        return_tensors: bool = False,
        **kwargs: Any,
    ) -> Dict[RenderType, structs.ArrayData]:

        raise NotImplementedError()
