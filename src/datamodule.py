from pathlib import Path

import timm
import torch
from omegaconf import OmegaConf, DictConfig

from lightning_module import DETR_Lightning


def load_dino(dino_version: bool) -> tuple[torch.nn.Module, int]:
    """Load DINOv2 backbone from timm library.
    
    Parameters
    ----------
    dino_version: str
        DINOv2 model version string (e.g. 'vit_base_patch14_dinov2.lvd142m').
    
    Returns
    -------
    tuple[torch.nn.Module, int]
        Backbone model and number of output channels.
    """
    
    print(f"  Loading backbone {dino_version} ...")
    # Loads DINOv2 backbone (and saves to cache).
    # Options: vit_small_patch14_dinov2.lvd142m, vit_base_patch14_dinov2.lvd142m,
    #          vit_large, etc.
    model = timm.create_model(
        dino_version,
        pretrained=True,
        features_only=True,  # important
    )

    # torch.save(model, 'data/vit_base_patch14_dinov2.lvd142m.pth')

    # DINOv2 outputs a list of features; we use the last one
    channels = model.feature_info.channels()[-1]
    print("    Backbone output channels:", channels)

    return model, channels


def load_config(config_path: str = "configs/default.yaml") -> DictConfig:
    """Load and parse YAML config file.
    """

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    return config


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
        Initialized DETR_Lightning model with loaded weights
    """

    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize model with config. The backbone freeze options are in the config.
    model = DETR_Lightning(config=config)
    
    # Load checkpoint weights.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    freeze_backbone = config["train"].get("freeze_backbone", True)
    print(f"Successfully loaded checkpoint. Backbone frozen: {freeze_backbone}")
    
    return model
