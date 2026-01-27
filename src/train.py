
import argparse
# from pathlib import Path

import pytorch_lightning as pl
import torch
# from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from coco import get_coco_data
from datamodule import load_checkpoint_for_finetuning, load_config
from lightning_module import DETR_Lightning


# def load_config(config_path: str = "configs/default.yaml") -> DictConfig:
#     """Load and parse YAML config file.
#     """

#     config_path = Path(config_path)
    
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config file not found: {config_path}")
    
#     config = OmegaConf.load(config_path)
    
#     return config


# def load_checkpoint_for_finetuning(
#         checkpoint_path: str, config: DictConfig
#     ) -> torch.nn.Module:
#     """
#     Loads a pretrained checkpoint and prepare the model for fine-tuning.
    
#     Parameters
#     ----------
#     checkpoint_path: str
#         Path to the checkpoint file (.ckpt).
#     config: DictConfig
#         Configuration dictionary.
    
#     Returns
#     -------
#     torch.nn.Module
#         Initialized DETR_Lightning model with loaded weights
#     """

#     checkpoint_path = Path(checkpoint_path)
    
#     if not checkpoint_path.exists():
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
#     print(f"Loading checkpoint from: {checkpoint_path}")
    
#     # Initialize model with config. The backbone freeze options are in the config.
#     model = DETR_Lightning(config=config)
    
#     # Load checkpoint weights.
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(checkpoint['state_dict'], strict=False)
    
#     freeze_backbone = config["train"].get("freeze_backbone", True)
#     print(f"Successfully loaded checkpoint. Backbone frozen: {freeze_backbone}")
    
#     return model


class CustomTrainer:

    def __init__(
            self, model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            max_epochs: int = 10):
        """Init.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the training data.
        max_epochs : int, optional
            Maximum number of training epochs, by default 10.
        """

        self.model = model
        self.dataloader = dataloader
        self.max_epochs = max_epochs

    def train(self):
        """Run the training process.
        1. Set up callbacks (e.g., model checkpointing).
        2. Create TensorBoard logger.
        3. Initialize PyTorch Lightning Trainer and start training.
        """

        # Set the callback functions.
        callbacks = []

        best_save_checkpoint = ModelCheckpoint(
            filename=(
                "best_epoch={epoch}-step={step}-train_loss={train_loss:.4f}"
            ),
            auto_insert_metric_name=False,
            save_top_k=5,
            save_last=True,
            monitor="train_loss",
        )

        callbacks.append(best_save_checkpoint)

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
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            logger=tb_logger,  # Add logger here
            # log_every_n_steps=300,  # Log every 10 training steps
            callbacks=callbacks,
            # limit_train_batches=0.1,  # TODO: For quick testing; remove for full training
        )

        trainer.fit(
            model=self.model,
            train_dataloaders=self.dataloader,
            val_dataloaders=None
        )


if __name__ == "__main__":

    print("\n    ==== START ====\n\n")
    
    if torch.cuda.is_available():
        print("\nUsing device:", torch.cuda.get_device_name(0), "\n")

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train or fine-tune DINO-DETR model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration.
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}\n")
    
    # Load or initialize model based on config.
    checkpoint_path = config["train"].get("checkpoint_path", None)
    
    if checkpoint_path:
        model = load_checkpoint_for_finetuning(
            checkpoint_path=checkpoint_path,
            config=config
        )
        print(f"Fine-tuning from checkpoint: {checkpoint_path}")
    else:
        model = DETR_Lightning(config=config)
        print("Training from scratch")

    # Get data.
    dataset, dataloader = get_coco_data(config)

    # Create and run trainer.
    trainer = CustomTrainer(
        model,
        dataloader,
        max_epochs=config["train"]["epochs"],
    )
    trainer.train()

    print("\n    ==== FINISH ====\n")