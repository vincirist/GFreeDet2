# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

from enum import Enum
from typing import Any

from utils import renderer_base


class RendererType(Enum):

    PYRENDER_RASTERIZER = "pyrender_rasterizer"
    GSPLAT_RASTERIZER = "gsplat_rasterizer"


def build(
    renderer_type: RendererType,
    **kwargs: Any,
) -> renderer_base.RendererBase:

    if renderer_type == RendererType.PYRENDER_RASTERIZER:
        from utils import renderer

        return renderer.PyrenderRasterizer(**kwargs)
    elif renderer_type == RendererType.GSPLAT_RASTERIZER:
        from utils import renderer_gs

        return renderer_gs.GsplatRasterizer(**kwargs)
    else:
        raise ValueError(f"Unknown renderer `{renderer_type}`.")
