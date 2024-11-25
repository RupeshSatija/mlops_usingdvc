from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig, OmegaConf

# Setup the root of the project
root = rootutils.setup_root(__file__, pythonpath=True)
from loguru import logger

from src.utils.utils import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
    instantiate_model,
    instantiate_trainer,
    setup_logger,
    task_wrapper,
)


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
) -> dict:
    """Train the model."""
    logger.info(f"Starting training! {cfg}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = trainer.callback_metrics
    logger.info(f"Training metrics:\n{train_metrics}")

    # Log the path of the best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model checkpoint saved at: {best_model_path}")
    return train_metrics


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
) -> Optional[dict]:
    """Test the model."""
    if not cfg.get("test"):
        logger.info("Skipping testing.")
        return None

    logger.info("Starting testing!")
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        logger.warning("Best checkpoint not found! Using current model weights...")
        ckpt_path = None

    test_metrics = trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path
    )[0]
    logger.info(f"Test metrics:{test_metrics}")
    return test_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training and optional testing."""
    print(OmegaConf.to_yaml(cfg))

    log_dir = Path(cfg.paths.log_dir)
    setup_logger(f"{log_dir}/train.log")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate_datamodule(cfg.data)
    model = instantiate_model(cfg.model)
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = instantiate_trainer(cfg.get("trainer"), callbacks, loggers)

    train_metrics = {}
    test_metrics = {}

    if cfg.get("train"):
        train_metrics = train(cfg, trainer, model, datamodule)

    if cfg.get("test"):
        test_metrics = test(cfg, trainer, model, datamodule)

    logger.info(f"Best checkpoint path:\n{cfg.callbacks.model_checkpoint.dirpath}")

    all_metrics = {**train_metrics, **(test_metrics or {})}
    logger.info(f"All metrics:{all_metrics}")


if __name__ == "__main__":
    main()
