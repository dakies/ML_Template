"""Tests for data loading modules."""

import pytest

from ml_template.data import ImageClassificationDataModule


class TestImageClassificationDataModule:
    """Tests for ImageClassificationDataModule."""

    def test_cifar10_properties(self) -> None:
        """Test CIFAR-10 datamodule properties."""
        dm = ImageClassificationDataModule(dataset_name="cifar10", batch_size=32)
        assert dm.channels == 3
        assert dm.size == 32
        assert dm.num_classes == 10
        assert dm.batch_size == 32

    def test_mnist_properties(self) -> None:
        """Test MNIST datamodule properties."""
        dm = ImageClassificationDataModule(dataset_name="mnist", batch_size=64)
        assert dm.channels == 1
        assert dm.size == 28
        assert dm.num_classes == 10
        assert dm.batch_size == 64

    def test_invalid_dataset(self) -> None:
        """Test error on invalid dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            ImageClassificationDataModule(dataset_name="invalid")

    def test_hparams_saved(self) -> None:
        """Test hyperparameters are saved."""
        dm = ImageClassificationDataModule(
            dataset_name="cifar10",
            batch_size=32,
            num_workers=4,
        )
        assert dm.hparams["dataset_name"] == "cifar10"
        assert dm.hparams["batch_size"] == 32
        assert dm.hparams["num_workers"] == 4
