import sys
from unittest.mock import patch

import pytest
import rootutils
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from src.utils.utils import get_rich_progress, setup_logger, task_wrapper


def test_setup_logger():
    with patch.object(logger, "add") as mock_add, patch.object(
        logger, "remove"
    ) as mock_remove:
        setup_logger("test.log")
        mock_remove.assert_called_once()
        mock_add.assert_any_call(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        mock_add.assert_any_call("test.log", rotation="10 MB")


def test_task_wrapper_success():
    @task_wrapper
    def sample_function(x, y):
        return x + y

    with patch.object(logger, "info") as mock_info:
        result = sample_function(2, 3)
        assert result == 5
        mock_info.assert_any_call("Starting sample_function")
        mock_info.assert_any_call("Finished sample_function")


def test_task_wrapper_exception():
    @task_wrapper
    def sample_function(x, y):
        raise ValueError("Test error")

    with patch.object(logger, "exception") as mock_exception:
        with pytest.raises(ValueError, match="Test error"):
            sample_function(2, 3)
        mock_exception.assert_called_once_with("Error in sample_function: Test error")


def test_get_rich_progress():
    progress = get_rich_progress()
    assert isinstance(progress, Progress)
    assert len(progress.columns) == 2
    assert isinstance(progress.columns[0], SpinnerColumn)
    assert isinstance(progress.columns[1], TextColumn)
