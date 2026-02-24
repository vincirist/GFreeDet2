import os
import os.path as osp
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import warnings

import rootutils

PROJ_ROOT = rootutils.setup_root(__file__, pythonpath=True)

from utils import data_util, net_util  # noqa: F401
from utils.logging_util import get_logger
from utils.str_util import safe_to_camel_case


warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="inference_gotrack_gs")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info("Initializing logger, callbacks and trainer")
    cfg_trainer = cfg.machine.trainer

    if "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    else:
        raise NotImplementedError("Only Tensorboard loggers are supported")
    os.makedirs(cfg.save_dir, exist_ok=True)

    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    model = instantiate(cfg.model)
    # Load the model checkpoint
    if cfg.model.checkpoint_path:
        net_util.load_checkpoint(
            model=model,
            checkpoint_path=cfg.model.checkpoint_path,
            checkpoint_key="model_state_dict",
            prefix="models.1.",
        )
    else:
        raise ValueError("Checkpoint path is required for inference")

    if cfg.mode == "pose_refinement":
        # Define the dataloader using dataset_name, coarse_pose_method
        cfg.data.dataloader.dataset_name = cfg.dataset_name
        # coarse_pose_method and coarse_pose_path should not be set simultaneously
        cfg.data.dataloader.coarse_pose_method = cfg.coarse_pose_method
        cfg.data.dataloader.coarse_pose_path = cfg.coarse_pose_path
        cfg.data.dataloader.gs_model_tpath = cfg.gs_model_tpath
        test_dataset = instantiate(cfg.data.dataloader)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            collate_fn=data_util.convert_list_scene_observations_to_gotrack_inputs,
        )
        if cfg.coarse_pose_method is not None:
            coarse_pose_str = safe_to_camel_case(cfg.coarse_pose_method, upper_first=True)
        else:
            coarse_pose_str = safe_to_camel_case(Path(cfg.coarse_pose_path).stem, upper_first=True)
        result_dir = Path(cfg.save_dir) / f"{cfg.mode}" / f"{cfg.dataset_name}_{coarse_pose_str}"
        result_file_name = "GotrackWith{}".format(
            safe_to_camel_case(test_dataset.coarse_poses_file_name, upper_first=True)
        )
        logger.info(f"Result dir: {result_dir}")
        logger.info(f"Result file name: {result_file_name}")
    else:  # for tracking?
        cfg.data.rbot.dataloader.object_name = cfg.object_name
        cfg.data.rbot.dataloader.sequence_name = cfg.sequence_name
        test_dataset = instantiate(cfg.data.rbot.dataloader)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            collate_fn=data_util.collate_fn,
        )
        result_dir = Path(cfg.save_dir) / f"{cfg.mode}" / f"{cfg.object_name}_{cfg.sequence_name}"
        result_file_name = f"gotrack-{cfg.object_name}-{cfg.sequence_name}.json"
    logger.info("Dataloaders initialized!")

    model.set_renderer(test_dataset)
    model.result_dir = Path(result_dir)
    model.result_file_name = result_file_name
    logger.info("Model initialized!")
    trainer.test(
        model,
        dataloaders=test_dataloader,
    )

    logger.info("Done!")


if __name__ == "__main__":
    run_inference()
