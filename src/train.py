
import argparse
import os
import sys
from pathlib import Path


def _set_cuda_visible_devices_from_argv() -> None:
    """Set CUDA_VISIBLE_DEVICES early, before torch initializes CUDA."""

    if "--gpu" not in sys.argv:
        return

    idx = sys.argv.index("--gpu")
    if idx + 1 >= len(sys.argv):
        raise ValueError("--gpu was provided without a value")

    gpu_value = str(sys.argv[idx + 1]).strip()
    if not gpu_value:
        raise ValueError("--gpu value is empty")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_value


_set_cuda_visible_devices_from_argv()

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from coco import get_coco_data
from datamodule import load_config
from lightning_module import DETR_Lightning


def load_checkpoint_for_finetuning(
        checkpoint_path: str, config: DictConfig
    ) -> torch.nn.Module:
    """
    Loads a pretrained checkpoint and prepare the model for fine-tuning.
    
    Parameters
    ----------
    checkpoint_path: str
        Path to the checkpoint file (.ckpt).
    config: DictConfig
        Configuration dictionary.
    
    Returns
    -------
    torch.nn.Module
        Initialized DETR_Lightning model with loaded weights.
    """

    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Initialize model with config. The backbone freeze options are in the config.
    model = DETR_Lightning(config=config)
    
    # Load checkpoint weights. PyTorch 2.6+ defaults weights_only=True.
    # This checkpoint is local/trusted, so full load is safe.
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    freeze_backbone = config["train"].get("freeze_backbone", True)
    print(f"  Successfully loaded checkpoint. Backbone frozen: {freeze_backbone}")
    
    return model


class CustomTrainer:

    def __init__(
            self, model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader | None = None,
            max_epochs: int = 10,
            limit_train_batches: float | int = 1.0,
            max_steps: int = -1,
            log_every_n_steps: int = 10,
            gradient_clip_val: float = 0.1,
            precision: str = "bf16-mixed"):
        """Init.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the training data.
        max_epochs : int, optional
            Maximum number of training epochs, by default 10.
        limit_train_batches : float | int, optional
            How much of training dataset to use (float = fraction, int = num_batches).
            Value is per device. Default: 1.0.
        max_steps : int, optional
            Global optimizer step budget. Use -1 to disable step cap.
        log_every_n_steps : int, optional
            Trainer logging cadence for step-level metrics.
        precision : str, optional
            PyTorch Lightning precision mode (e.g. "16-mixed", "bf16-mixed", "32-true").
        """

        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.limit_train_batches = limit_train_batches
        self.max_steps = max_steps
        self.log_every_n_steps = log_every_n_steps
        self.gradient_clip_val = float(gradient_clip_val)
        self.precision = str(precision)

    def train(self):
        """Run the training process.
        1. Set up callbacks (e.g., model checkpointing).
        2. Create TensorBoard logger.
        3. Initialize PyTorch Lightning Trainer and start training.
        """

        # Set the callback functions.
        callbacks = []

        checkpoint_monitor = "train_loss"
        checkpoint_mode = "min"
        checkpoint_filename = "best_epoch={epoch}-step={step}-train_loss={train_loss:.4f}"

        if self.val_dataloader is not None:
            # Standard Lightning validation loop computes val metrics.
            checkpoint_monitor = "val_epoch_score"
            checkpoint_mode = "min"
            checkpoint_filename = (
                "best_epoch={epoch}-step={step}-"
                "val_score={val_epoch_score:.4f}-train_loss={train_loss:.4f}"
            )

        best_save_checkpoint = ModelCheckpoint(
            filename=checkpoint_filename,
            auto_insert_metric_name=False,
            save_top_k=5,
            save_last=True,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_on_train_epoch_end=True,
        )

        callbacks.append(best_save_checkpoint)

        print(
            "Checkpoint monitor settings: "
            f"monitor={checkpoint_monitor}, mode={checkpoint_mode}"
        )

        # Create TensorBoard logger.
        tb_logger = TensorBoardLogger(
            save_dir="",
            # name="DINO_DETR_TB_log",
            version=None  # Auto-increment version
        )

        # Create Trainer and start training.
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=self.precision,
            benchmark=True,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            log_every_n_steps=self.log_every_n_steps,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
            logger=tb_logger,  # Add logger here
            # log_every_n_steps=300,  # Log every 10 training steps
            callbacks=callbacks,
            limit_train_batches=self.limit_train_batches,
        )

        trainer.fit(
            model=self.model,
            train_dataloaders=self.dataloader,
            # Use Lightning's built-in validation loop.
            val_dataloaders=self.val_dataloader,
        )


if __name__ == "__main__":

    print("\n    ==== START ====\n\n")

    # 1. Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train or fine-tune DINO-DETR model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="CUDA visible device(s), e.g. '0' or '0,1'"
    )
    args = parser.parse_args()
    
    print(f"  Arguments:\n" +
          f"    - config: {args.config}\n" +
          f"    - gpu: {args.gpu}\n"
    )

    # Use faster matmul kernels where possible.
    torch.set_float32_matmul_precision("high")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("  Environment info:")
        print(
            "    Device:", torch.cuda.get_device_name(0) + "\n" + 
            "    torch version: " + torch.__version__ + "\n" + 
            "    CUDA version: " + torch.version.cuda + "\n" + 
            "    CUDA architectures: " + str(torch.cuda.get_arch_list()) + "\n"
        )

    # 2. Load configuration and construct the model.
    config = load_config(args.config)
    
    checkpoint_path = config["train"].get("checkpoint_path", None)
    if checkpoint_path:
        print(f"\n  Fine-tuning from checkpoint:\n    {checkpoint_path}")
        model = load_checkpoint_for_finetuning(
            checkpoint_path=checkpoint_path,
            config=config
        )
    else:
        print("\n  Training from scratch.")
        model = DETR_Lightning(config=config)

    # 3. Get train/validation dataloaders.
    train_dataloader, val_dataloader = get_coco_data(config)

    # 4. Log settings.
    train_cfg = config["train"]
    limit_train_batches = train_cfg.get("limit_train_batches", 1.0)
    max_steps = train_cfg.get("max_steps", -1)
    if max_steps is None:
        max_steps = -1
    log_every_n_steps = int(config.get("logging", {}).get("log_interval", 10))
    gradient_clip_val = float(train_cfg.get("gradient_clip_val", 0.1))
    precision = str(train_cfg.get("precision", "bf16-mixed"))

    print(
        "  Trainer settings:\n"
        f"    limit_train_batches={limit_train_batches}\n"
        f"    max_steps={max_steps}\n"
        f"    log_every_n_steps={log_every_n_steps}\n"
        f"    gradient_clip_val={gradient_clip_val}\n"
        f"    precision={precision}\n"
    )
    
    # 5. Create and run trainer (wrapper around PyTorch Lightning's Trainer).
    trainer = CustomTrainer(
        model,
        train_dataloader,
        val_dataloader=val_dataloader,
        max_epochs=train_cfg["epochs"],
        limit_train_batches=limit_train_batches,
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
    )
    trainer.train()

    print("\n    ==== FINISH ====\n")
