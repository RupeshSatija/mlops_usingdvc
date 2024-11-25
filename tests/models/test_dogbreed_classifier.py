import pytest
import rootutils
import torch

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)


from src.models.dogbreed_classifier import DogBreedGenericClassifier


@pytest.fixture
def model_config():
    return {
        "model_name": "resnet50",
        "num_classes": 10,
        "pretrained": True,
        "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
        "scheduler": {"factor": 0.1, "patience": 3, "min_lr": 1e-6},
    }


@pytest.fixture
def model(model_config):
    return DogBreedGenericClassifier(**model_config)


@pytest.fixture
def sample_batch():
    return torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,))


def test_model_init(model, model_config):
    assert model.model_name == model_config["model_name"]
    assert model.num_classes == model_config["num_classes"]
    assert model.pretrained == model_config["pretrained"]


def test_forward(model, sample_batch):
    images, _ = sample_batch
    output = model(images)
    assert output.shape == (4, model.num_classes)


def test_training_step(model, sample_batch):
    loss = model.training_step(sample_batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


# def test_validation_step(model, sample_batch):
#     output = model.validation_step(sample_batch, 0)
#     assert "val/loss" in output
#     assert "val/acc" in output


# def test_test_step(model, sample_batch):
#     output = model.test_step(sample_batch, 0)
#     assert "test_loss" in output
#     assert "test_acc" in output


def test_predict_step(model, sample_batch):
    preds, probs = model.predict_step(sample_batch, 0)
    assert preds.shape == (4,)
    assert probs.shape == (4, model.num_classes)


def test_configure_optimizers(model):
    optimizers = model.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert isinstance(optimizers["optimizer"], torch.optim.Adam)
    assert isinstance(
        optimizers["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_different_batch_sizes(model, batch_size):
    images = torch.randn(batch_size, 3, 224, 224)
    output = model(images)
    assert output.shape == (batch_size, model.num_classes)


def test_freeze_unfreeze(model):
    model.freeze()
    for param in model.model.parameters():
        assert not param.requires_grad
    model.unfreeze()
    for param in model.model.parameters():
        assert param.requires_grad


# def test_load_from_checkpoint(model, tmp_path):
#     checkpoint_path = os.path.join(tmp_path, "model_checkpoint.ckpt")

#     # Manually save the model state and hyperparameters
#     checkpoint = {
#         "state_dict": model.state_dict(),
#         "hyper_parameters": dict(model.hparams),
#         "lightning_version": L.__version__,
#         "pytorch_version": torch.__version__,
#     }
#     torch.save(checkpoint, checkpoint_path)

#     # Ensure the checkpoint file exists
#     assert os.path.isfile(
#         checkpoint_path
#     ), f"Checkpoint file not created at {checkpoint_path}"

#     # Load the model from the checkpoint
#     loaded_model = DogBreedGenericClassifier.load_from_checkpoint(checkpoint_path)

#     # Assert that the loaded model is of the correct type
#     assert isinstance(
#         loaded_model, DogBreedGenericClassifier
#     ), "Loaded model is not of type DogBreedGenericClassifier"

#     # Compare model parameters
#     for param_original, param_loaded in zip(
#         model.parameters(), loaded_model.parameters()
#     ):
#         assert torch.allclose(
#             param_original, param_loaded
#         ), "Loaded model parameters do not match original model"

#     # Test forward pass
#     dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as needed
#     with torch.no_grad():
#         output_original = model(dummy_input)
#         output_loaded = loaded_model(dummy_input)

#     assert torch.allclose(
#         output_original, output_loaded, atol=1e-6
#     ), "Forward pass results do not match"

#     # Compare hyperparameters
#     for key in model.hparams:
#         assert (
#             model.hparams[key] == loaded_model.hparams[key]
#         ), f"Hyperparameter {key} does not match"
