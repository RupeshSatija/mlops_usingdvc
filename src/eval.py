from pathlib import Path

import hydra
import lightning as L
import rootutils
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

# Setup the root of the project
root = rootutils.setup_root(__file__, pythonpath=True)
from src.datamodules.dogbreed_dataset import DogBreedDataModule
from src.models.dogbreed_classifier import DogBreedGenericClassifier
from src.utils.utils import (
    instantiate_datamodule,
    setup_logger,
)


def load_model_from_checkpoint(checkpoint_path: str) -> DogBreedGenericClassifier:
    return DogBreedGenericClassifier.load_from_checkpoint(checkpoint_path)


def evaluate(
    cfg: DictConfig, data_module: DogBreedDataModule, model: DogBreedGenericClassifier
) -> None:
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        logger=TensorBoardLogger(
            save_dir=cfg.paths.log_dir, name="dogbreed_evaluation"
        ),
    )
    results = trainer.validate(model, datamodule=data_module)

    print("\nValidation Metrics:")
    for k, v in results[0].items():
        print(f"{k}: {v:.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    # Set up paths
    base_dir = Path(cfg.paths.base_dir)
    data_dir = base_dir / cfg.paths.data_dir
    log_dir = base_dir / cfg.paths.log_dir

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    datamodule = instantiate_datamodule(cfg.data)
    # data_module = DogBreedDataModule(**cfg.data)
    datamodule.setup(stage="validate")

    # Load model from checkpoint
    model = load_model_from_checkpoint(cfg.checkpoint_path)

    # Run evaluation
    evaluate(cfg, datamodule, model)


if __name__ == "__main__":
    main()
