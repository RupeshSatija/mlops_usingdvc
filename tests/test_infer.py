from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import rootutils
import torch
from PIL import Image

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from src.infer import infer, load_image, main, save_prediction_image
from src.models.dogbreed_classifier import DogBreedGenericClassifier


@pytest.fixture
def mock_model():
    return MagicMock(spec=DogBreedGenericClassifier)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100))


def test_load_image(sample_image):
    with patch("src.infer.Image.open", return_value=sample_image):
        img, tensor = load_image("dummy_path.jpg")
        assert isinstance(img, Image.Image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)


def test_infer(mock_model):
    mock_model.eval.return_value = None
    mock_model.return_value = torch.randn(1, 10)

    image_tensor = torch.randn(1, 3, 224, 224)
    label, confidence = infer(mock_model, image_tensor)

    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1


@patch("src.infer.plt")
def test_save_prediction_image(mock_plt, sample_image):
    save_prediction_image(sample_image, "Beagle", 0.95, "output.png")
    mock_plt.savefig.assert_called_once_with("output.png", dpi=300, bbox_inches="tight")


# @patch("src.infer.DogBreedGenericClassifier.load_from_checkpoint")
# @patch("src.infer.Path")
# @patch("src.infer.load_image")
# @patch("src.infer.infer")
# @patch("src.infer.save_prediction_image")
# def test_main(
#     mock_save, mock_infer, mock_load_image, mock_path, mock_load_model, mock_model
# ):
#     mock_args = MagicMock()
#     mock_args.input_folder = "input"
#     mock_args.output_folder = "output"
#     mock_args.ckpt_path = "model.ckpt"

#     mock_path.return_value.glob.return_value = ["image1.jpg", "image2.png"]
#     mock_load_image.return_value = (MagicMock(), torch.randn(1, 3, 224, 224))
#     mock_infer.return_value = ("Beagle", 0.95)
#     mock_load_model.return_value = mock_model

#     with patch("src.infer.get_rich_progress"):
#         main(mock_args)

#     assert mock_load_image.call_count == 2
#     assert mock_infer.call_count == 2
#     assert mock_save.call_count == 2


@pytest.mark.parametrize("file_extension", [".jpg", ".jpeg", ".png"])
def test_main_different_file_extensions(file_extension, mock_model):
    mock_args = MagicMock()
    mock_args.input_folder = "input"
    mock_args.output_folder = "output"
    mock_args.ckpt_path = "model.ckpt"

    with patch("src.infer.Path") as mock_path, patch(
        "src.infer.load_image"
    ) as mock_load_image, patch("src.infer.infer") as mock_infer, patch(
        "src.infer.save_prediction_image"
    ) as mock_save, patch(
        "src.infer.DogBreedGenericClassifier.load_from_checkpoint",
        return_value=mock_model,
    ), patch("src.infer.get_rich_progress"):
        mock_path.return_value.glob.return_value = [f"image1{file_extension}"]
        mock_load_image.return_value = (MagicMock(), torch.randn(1, 3, 224, 224))
        mock_infer.return_value = ("Beagle", 0.95)

        main(mock_args)

        mock_load_image.assert_called_once()
        mock_infer.assert_called_once()
        mock_save.assert_called_once()


def test_main_no_images():
    mock_args = MagicMock()
    mock_args.input_folder = "input"
    mock_args.output_folder = "output"
    mock_args.ckpt_path = "model.ckpt"

    with patch("src.infer.Path") as mock_path, patch(
        "src.infer.DogBreedGenericClassifier.load_from_checkpoint"
    ), patch("src.infer.get_rich_progress"):
        mock_path.return_value.glob.return_value = []

        main(mock_args)

        # Assert that no processing occurred
        mock_path.return_value.glob.assert_called_once()


@pytest.mark.parametrize("file_extension", [".jpg", ".jpeg", ".png"])
def test_main_different_file_extensions(file_extension):
    file_path = f"dummy_image{file_extension}"  # This is a string
    path_obj = Path(file_path)  # Convert to Path object

    # Now you can safely access the suffix
    assert path_obj.suffix == file_extension

    # Call your function that uses the path
    # result = your_function(path_obj)
    # assert result == expected_value
