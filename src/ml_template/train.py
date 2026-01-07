"""Main training entrypoint with Hydra configuration."""

import hydra
import lightning as L
import mlflow
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from ml_template.callbacks import ONNXExportCallback
from ml_template.data import ImageClassificationDataModule
from ml_template.models import ImageClassifier
from ml_template.utils import get_logger, setup_logging

log = get_logger(__name__)


def train(cfg: DictConfig) -> float | None:
    """Run training with the provided configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Best validation metric achieved during training.
    """
    setup_logging(cfg.get("log_level", "INFO"))

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize data module
    datamodule = ImageClassificationDataModule(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
    )

    # Initialize model
    model = ImageClassifier(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        backbone=cfg.model.backbone,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        scheduler=cfg.model.scheduler,
    )

    # Setup callbacks
    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=cfg.trainer.checkpoint_dir,
            filename="{epoch}-{val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=cfg.trainer.save_top_k,
            save_last=True,
        ),
    ]

    if cfg.trainer.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/acc",
                patience=cfg.trainer.patience,
                mode="max",
            )
        )

    if cfg.get("export_onnx", False):
        callbacks.append(
            ONNXExportCallback(
                export_path=cfg.export_onnx_path,
                input_shape=(1, cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size),
            )
        )

    # Setup loggers
    loggers = []

    if cfg.logger.tensorboard.enabled:
        loggers.append(
            TensorBoardLogger(
                save_dir=cfg.logger.tensorboard.save_dir,
                name=cfg.experiment_name,
            )
        )

    if cfg.logger.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.logger.mlflow.tracking_uri)
        loggers.append(
            MLFlowLogger(
                experiment_name=cfg.experiment_name,
                tracking_uri=cfg.logger.mlflow.tracking_uri,
                log_model=cfg.logger.mlflow.log_model,
            )
        )

    # Initialize trainer
    trainer_kwargs = {
        "max_epochs": cfg.trainer.max_epochs,
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "callbacks": callbacks,
        "logger": loggers if loggers else True,
        "gradient_clip_val": cfg.trainer.gradient_clip_val,
        "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
        "deterministic": cfg.get("deterministic", False),
        "enable_progress_bar": True,
    }

    # Add strategy for distributed training
    if cfg.trainer.strategy:
        trainer_kwargs["strategy"] = cfg.trainer.strategy

    trainer = L.Trainer(**trainer_kwargs)

    # Train
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Test with best checkpoint
    if cfg.trainer.run_test:
        log.info("Running test evaluation...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Return best validation metric
    best_metric = trainer.callback_metrics.get("val/acc")
    return float(best_metric) if best_metric else None


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Hydra entrypoint."""
    return train(cfg)


if __name__ == "__main__":
    main()
