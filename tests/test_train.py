import os
from unittest.mock import MagicMock

import lightning as L
import omegaconf
import pytest
import rootutils
import torch
from lightning import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from omegaconf import DictConfig

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from unittest.mock import patch

import src.train as train_module
from src.datamodules.dogbreed_dataset import DogBreedDataModule
from src.models.dogbreed_classifier import DogBreedGenericClassifier
from src.utils.utils import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
    instantiate_model,
    instantiate_trainer,
)


@pytest.fixture
def config():
    return omegaconf.OmegaConf.create(
        {
            "seed": 42,
            "paths": {"log_dir": "logs"},
            "data": {
                "_target_": "src.datamodules.dogbreed_dataset.DogBreedDataModule",
                "dir": "./data/dogbreeds",
                "batch_size": 32,
                "num_workers": 0,
                "pin_memory": False,
                "train_val_test_split": [0.7, 0.2, 0.1],
                "google_drive_id": "1WZ_H2GxgNr7_HWtJHgmy70d7R2_QMBZ2",
                "image_size": 224,
                "crop_size": 224,
            },
            "model": {
                "_target_": "src.models.dogbreed_classifier.DogBreedGenericClassifier",
                "model_name": "resnet50",
                "num_classes": 10,
                "pretrained": True,
                "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
                "scheduler": {
                    "mode": "min",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-6,
                    "verbose": True,
                },
            },
            "callbacks": {},
            "logger": {},
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "default_root_dir": "./outputs/",
                "min_epochs": 1,
                "max_epochs": 100,
                "accelerator": "auto",
                "devices": 1,
                "precision": 32,
                "val_check_interval": 1.0,
                "deterministic": False,
                # "fast_dev_run": True,
            },
        }
    )


@pytest.fixture
def config_concrete():
    return omegaconf.OmegaConf.create(
        {
            "seed": 42,
            "paths": {"log_dir": "logs"},
            "data": {
                "dir": "./data/dogbreeds",
                "batch_size": 32,
                "num_workers": 0,
                "pin_memory": False,
                "train_val_test_split": [0.7, 0.2, 0.1],
                "google_drive_id": "1WZ_H2GxgNr7_HWtJHgmy70d7R2_QMBZ2",
                "image_size": 224,
                "crop_size": 224,
            },
            "model": {
                "model_name": "resnet50",
                "num_classes": 10,
                "pretrained": True,
                "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
                "scheduler": {
                    "mode": "min",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-6,
                    "verbose": True,
                },
            },
            "callbacks": {},
            "logger": {},
            "trainer": {
                # "default_root_dir": "./outputs/",
                "min_epochs": 1,
                "max_epochs": 100,
                "accelerator": "auto",
                "devices": 1,
                "precision": 32,
                "val_check_interval": 1.0,
                "deterministic": False,
                # "fast_dev_run": True,
            },
        }
    )


@pytest.fixture
def model(config_concrete):
    return DogBreedGenericClassifier(**config_concrete.model)


@pytest.fixture
def datamodule(config_concrete):
    return DogBreedDataModule(**config_concrete.data)


def test_instantiate_callbacks(config):
    callbacks = instantiate_callbacks(config.callbacks)
    assert isinstance(callbacks, list)


def test_instantiate_callbacks_not_dictconfig():
    config_callbacks = [
        {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "monitor": "val_loss",
        }
    ]
    with pytest.raises(TypeError, match="Callbacks config must be a DictConfig!"):
        instantiate_callbacks(config_callbacks)


def test_instantiate_callbacks_with_target():
    config_callbacks = DictConfig(
        {
            "checkpoint": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "monitor": "val_loss",
                "save_top_k": 1,
                "mode": "min",
            }
        }
    )
    callbacks = instantiate_callbacks(config_callbacks)
    assert isinstance(callbacks, list)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ModelCheckpoint)
    assert callbacks[0].monitor == "val_loss"
    assert callbacks[0].save_top_k == 1
    assert callbacks[0].mode == "min"


def test_instantiate_loggers(config):
    loggers = instantiate_loggers(config.logger)
    assert isinstance(loggers, list)


def test_instantiate_trainer(config):
    callbacks = []
    loggers = []
    trainer = instantiate_trainer(config.trainer, callbacks, loggers)
    assert isinstance(trainer, L.Trainer)


def test_instantiate_model(config):
    model = instantiate_model(config.model)
    assert isinstance(model, DogBreedGenericClassifier)


def test_instantiate_datamodule(config):
    datamodule = instantiate_datamodule(config.data)
    assert isinstance(datamodule, DogBreedDataModule)


@patch("lightning.pytorch.Trainer.fit")
def test_train(mock_fit, config_concrete, model, datamodule):
    trainer = L.Trainer(**config_concrete.trainer)
    train_metrics = train_module.train(config_concrete, trainer, model, datamodule)
    mock_fit.assert_called_once()
    assert isinstance(train_metrics, dict)

    assert "loss" in train_metrics
    assert isinstance(train_metrics["loss"], float)

    config_concrete.trainer["max_epochs"] = 5
    train_metrics_5_epochs = train_module.train(
        config_concrete, trainer, model, datamodule
    )
    assert train_metrics_5_epochs != train_metrics


def test_train_integration(config_concrete, model, datamodule):
    trainer = L.Trainer(**config_concrete.trainer, fast_dev_run=True)
    train_metrics = train_module.train(config_concrete, trainer, model, datamodule)
    assert isinstance(train_metrics, dict)
    assert len(train_metrics) > 0


@patch("lightning.pytorch.Trainer.fit")
def test_train(mock_fit, config_concrete, model, datamodule):
    trainer = L.Trainer(**config_concrete.trainer)
    train_metrics = train_module.train(config_concrete, trainer, model, datamodule)
    mock_fit.assert_called_once()
    assert isinstance(train_metrics, dict)


def test_train_integration(config_concrete, model, datamodule):
    trainer = Trainer(**config_concrete.trainer, fast_dev_run=True)
    trainer.fit(model, datamodule)
    assert trainer.state.finished, "Training failed to complete"


def test_model_save(tmp_path, config_concrete, model, datamodule):
    checkpoint_path = os.path.join(tmp_path, "model_checkpoint.ckpt")
    trainer = Trainer(**config_concrete.trainer, fast_dev_run=True)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(checkpoint_path)
    assert os.path.isfile(
        checkpoint_path
    ), f"Model checkpoint not saved at {checkpoint_path}"


def test_model_load(tmp_path, config_concrete, model, datamodule):
    checkpoint_path = os.path.join(tmp_path, "model_checkpoint.ckpt")
    trainer = L.Trainer(**config_concrete.trainer, fast_dev_run=True)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(checkpoint_path)
    assert os.path.isfile(
        checkpoint_path
    ), f"Model checkpoint not saved at {checkpoint_path}"
    loaded_model = DogBreedGenericClassifier.load_from_checkpoint(checkpoint_path)
    assert isinstance(
        loaded_model, DogBreedGenericClassifier
    ), "Failed to load model from checkpoint"
    for orig_param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(
            orig_param, loaded_param
        ), "Loaded model parameters do not match original model"
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original_output = model(dummy_input)
        loaded_output = loaded_model(dummy_input)
    assert torch.allclose(
        original_output, loaded_output
    ), "Loaded model output does not match original model"


@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=L.Trainer)
    trainer.checkpoint_callback.best_model_path = "best_model.ckpt"
    trainer.test.return_value = [{"test_loss": 0.5, "test_accuracy": 0.9}]
    return trainer


@patch("src.train.logger")
def test_test_function_with_test_enabled(
    mock_logger, config_concrete, model, datamodule, mock_trainer
):
    config_concrete["test"] = True
    test_metrics = train_module.test(config_concrete, mock_trainer, model, datamodule)

    mock_trainer.test.assert_called_once_with(
        model=model, datamodule=datamodule, ckpt_path="best_model.ckpt"
    )
    assert isinstance(test_metrics, dict)
    assert "test_loss" in test_metrics
    assert "test_accuracy" in test_metrics
    mock_logger.info.assert_called_with(f"Test metrics:{test_metrics}")


@patch("src.train.logger")
def test_test_function_with_test_disabled(
    mock_logger, config_concrete, model, datamodule, mock_trainer
):
    config_concrete["test"] = False
    test_metrics = train_module.test(config_concrete, mock_trainer, model, datamodule)

    mock_trainer.test.assert_not_called()
    assert test_metrics is None
    mock_logger.info.assert_called_with("Skipping testing.")


@patch("src.train.logger")
def test_test_function_with_no_best_checkpoint(
    mock_logger, config_concrete, model, datamodule, mock_trainer
):
    config_concrete["test"] = True
    mock_trainer.checkpoint_callback.best_model_path = ""

    test_metrics = train_module.test(config_concrete, mock_trainer, model, datamodule)

    mock_trainer.test.assert_called_once_with(
        model=model, datamodule=datamodule, ckpt_path=None
    )
    assert isinstance(test_metrics, dict)
    mock_logger.warning.assert_called_with(
        "Best checkpoint not found! Using current model weights..."
    )


# @pytest.mark.parametrize("test_enabled", [True, False])
# def test_main_function_test_execution(test_enabled, config_concrete, monkeypatch):
#     config_concrete["test"] = test_enabled

#     mock_train = MagicMock(return_value={})
#     mock_test = MagicMock(return_value={})

#     monkeypatch.setattr(train_module, "train", mock_train)
#     monkeypatch.setattr(train_module, "test", mock_test)

#     with patch("src.train.instantiate_trainer"), patch(
#         "src.train.instantiate_model"
#     ), patch("src.train.instantiate_datamodule"), patch(
#         "src.train.instantiate_callbacks"
#     ), patch("src.train.instantiate_loggers"), patch("src.train.setup_logger"), patch(
#         "src.train.logger"
#     ):
#         train_module.main(config_concrete)

#         mock_train.assert_called_once()
#         if test_enabled:
#             mock_test.assert_called_once()
#         else:
#             mock_test.assert_not_called()
