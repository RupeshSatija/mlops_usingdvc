import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def config():
    return OmegaConf.create(
        {
            "model": {"num_classes": 10, "pretrained": False},
            "data": {"batch_size": 32, "num_workers": 0},
            "trainer": {"max_epochs": 1, "gpus": 0},
        }
    )


@pytest.fixture
def sample_batch():
    return torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,))
