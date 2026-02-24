# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from pathlib import Path
import distinctipy
import pytorch_lightning as pl
from tqdm import tqdm
from utils import renderer_builder, repre_util, logging_util
from dataloader import base

logger = logging_util.get_logger(__name__)


class ModelBase(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.objects_repre = None
        self.renderer = None
        self.colors = None

    def onboarding(self, repre_dir: Path):

        obj_ids = [int(repr_dir.name) for repr_dir in repre_dir.iterdir()]
        obj_ids = sorted(obj_ids)

        self.objects_repre = {}
        for obj_id in tqdm(obj_ids, desc="Loading object representations"):

            obj_repre_dir = repre_dir / f"{obj_id}"
            repre = repre_util.load_object_repre(
                repre_dir=obj_repre_dir,
                tensor_device=self.device,
            )
            self.objects_repre[obj_id] = repre
        logger.info("Object representations loaded.")
        return self.objects_repre

    def set_renderer(self, dataset: base.GoTrackDataset) -> None:

        renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
        self.renderer = renderer_builder.build(renderer_type=renderer_type, model_path=dataset.dp_model["model_tpath"])
        self.object_vertices = dataset.models_vertices
        max_obj_ids = max(dataset.models.keys())
        self.colors = distinctipy.get_colors(max_obj_ids + 1)

    def post_onboarding_processing(self):

        raise NotImplementedError("This function should be overridden by subclasses.")

    def set_device(self):

        raise NotImplementedError("This function should be overridden by subclasses.")
