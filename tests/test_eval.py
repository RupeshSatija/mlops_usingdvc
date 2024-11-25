from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from src.datamodules.dogbreed_dataset import DogBreedDataModule
from src.eval import evaluate, load_model_from_checkpoint
from src.models.dogbreed_classifier import DogBreedGenericClassifier


@pytest.fixture
def mock_data_module():
    return MagicMock(spec=DogBreedDataModule)


@pytest.fixture
def mock_model():
    return MagicMock(spec=DogBreedGenericClassifier)


@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=L.Trainer)
    trainer.validate.return_value = [{"val_loss": 0.3, "val_accuracy": 0.85}]
    return trainer


@pytest.fixture
def mock_cfg():
    # Create a mock configuration object
    cfg = MagicMock()
    cfg.trainer.accelerator = "cpu"  # or whatever value is appropriate
    cfg.paths.log_dir = "dummy_log_dir"  # or whatever value is appropriate
    return cfg


def test_load_model_from_checkpoint(mock_model):
    with patch.object(
        DogBreedGenericClassifier, "load_from_checkpoint", return_value=mock_model
    ):
        model = load_model_from_checkpoint("dummy_path.ckpt")
        assert isinstance(model, DogBreedGenericClassifier)


def test_evaluate(mock_cfg, mock_data_module, mock_model, mock_trainer):
    with patch("src.eval.L.Trainer", return_value=mock_trainer):
        # Ensure to pass the model argument and the config
        evaluate(mock_cfg, mock_data_module, mock_model)  # Updated line
        mock_trainer.validate.assert_called_once_with(
            mock_model, datamodule=mock_data_module
        )


# @patch("src.eval.setup_logger")
# @patch("src.eval.DogBreedDataModule")
# @patch("src.eval.load_model_from_checkpoint")
# @patch("src.eval.evaluate")
# def test_main(mock_evaluate, mock_load_model, mock_data_module, mock_setup_logger):
#     mock_args = MagicMock()
#     mock_args.checkpoint = "dummy_checkpoint.ckpt"  # Ensure this is the expected path

#     with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
#         # Mock the existence of the checkpoint file
#         with patch(
#             "pathlib.Path.exists", return_value=True
#         ):  # Ensure this returns True
#             main()

#     mock_setup_logger.assert_called_once()
#     mock_data_module.assert_called_once()
#     mock_load_model.assert_called_once_with("dummy_checkpoint.ckpt")
#     mock_evaluate.assert_called_once()


@pytest.mark.parametrize(
    "val_results",
    [
        [{"val_loss": 0.3, "val_accuracy": 0.85}],
        [{"val_loss": 0.5, "val_accuracy": 0.75}],
    ],
)
def test_evaluate_different_results(
    mock_cfg, mock_data_module, mock_model, val_results
):
    mock_trainer = MagicMock(spec=L.Trainer)
    mock_trainer.validate.return_value = val_results

    with patch("src.eval.L.Trainer", return_value=mock_trainer):
        # Ensure to pass the model argument and the config
        evaluate(mock_cfg, mock_data_module, mock_model)  # Updated line
        mock_trainer.validate.assert_called_once_with(
            mock_model, datamodule=mock_data_module
        )


# def test_main_file_not_found():
#     with pytest.raises(FileNotFoundError):
#         with patch("src.eval.argparse.ArgumentParser.parse_args") as mock_args:
#             mock_args.return_value.checkpoint = "non_existent_file.ckpt"
#             main()
