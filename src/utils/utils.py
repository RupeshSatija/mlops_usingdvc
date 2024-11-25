import sys
from functools import wraps
from typing import List

import hydra
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from loguru import logger
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, TextColumn


def setup_logger(log_file):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(log_file, rotation="10 MB")


def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper


def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiate loggers from config."""
    loggers: List[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def instantiate_trainer(
    cfg: DictConfig, callbacks: List[Callback], loggers: List[Logger]
) -> L.Trainer:
    """Instantiate the trainer."""
    logger.info(f"Instantiating trainer <{cfg._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg, callbacks=callbacks, logger=loggers
    )
    return trainer


def instantiate_model(cfg: DictConfig) -> L.LightningModule:
    """Instantiate the model."""
    logger.info(f"Instantiating model <{cfg._target_}>")
    return hydra.utils.instantiate(cfg)


def instantiate_datamodule(cfg: DictConfig) -> L.LightningDataModule:
    """Instantiate the datamodule."""
    logger.info(f"Instantiating datamodule <{cfg._target_}>")
    return hydra.utils.instantiate(cfg)
