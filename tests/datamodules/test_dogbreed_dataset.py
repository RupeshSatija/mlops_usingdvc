import os

import pytest
import rootutils
import torch

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from src.datamodules.dogbreed_dataset import DATASET_FLAG_FILE, DogBreedDataModule


@pytest.fixture
def datamodule_config():
    return {
        "dir": "data/",
        "batch_size": 32,
        "num_workers": 2,
        "pin_memory": True,
        "train_val_test_split": [0.7, 0.15, 0.15],
        "google_drive_id": "1WZ_H2GxgNr7_HWtJHgmy70d7R2_QMBZ2",
        "image_size": 224,
        "crop_size": 224,
    }


@pytest.fixture
def datamodule(datamodule_config):
    return DogBreedDataModule(**datamodule_config)


def test_datamodule_init(datamodule, datamodule_config):
    for key, value in datamodule_config.items():
        assert getattr(datamodule, key) == value


def test_prepare_data(datamodule, tmp_path, monkeypatch):
    # def mock_download(*args, **kwargs):
    #     pass

    # monkeypatch.setattr("gdown.download", mock_download)
    # datamodule.data_dir = tmp_path
    datamodule.prepare_data()
    assert os.path.exists(os.path.join(datamodule.dir, DATASET_FLAG_FILE))


def test_setup(datamodule):
    datamodule.setup()
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert len(datamodule.train_dataset) > len(datamodule.val_dataset)
    assert len(datamodule.train_dataset) > len(datamodule.test_dataset)


def test_train_dataloader(datamodule):
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert len(train_loader) > 0


def test_val_dataloader(datamodule):
    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert len(val_loader) > 0


def test_test_dataloader(datamodule):
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    assert len(test_loader) > 0


def test_transforms(datamodule):
    assert datamodule.train_transform is not None
    assert datamodule.valid_transform is not None
    assert datamodule.normalize_transform is not None


def test_class_names(datamodule):
    datamodule.setup()
    class_names = datamodule.get_class_names()
    assert isinstance(class_names, list)
    assert len(class_names) > 0


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_different_batch_sizes(datamodule_config, batch_size):
    datamodule_config["batch_size"] = batch_size
    datamodule = DogBreedDataModule(**datamodule_config)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    assert batch[0].shape[0] == batch_size


def test_num_workers(datamodule_config):
    datamodule_config["num_workers"] = 4
    datamodule = DogBreedDataModule(**datamodule_config)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    assert train_loader.num_workers == 4


def test_pin_memory(datamodule_config):
    datamodule_config["pin_memory"] = False
    datamodule = DogBreedDataModule(**datamodule_config)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    assert train_loader.pin_memory == False
