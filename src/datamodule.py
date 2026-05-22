from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import timm
import torch


def load_backbone(backbone_name: str) -> tuple[torch.nn.Module, int]:
    """Load backbone from timm library (supports DINOv2 ViTs and ResNets).
    
    Parameters
    ----------
    backbone_name: str
        Backbone model name (e.g. 'vit_base_patch14_dinov2.lvd142m', 'resnet50').
    
    Returns
    -------
    tuple[torch.nn.Module, int]
        Backbone model and number of output channels.
    """

    print(f"  Loading backbone {backbone_name} ...")
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        features_only=True,  # Get intermediate feature maps instead of classification head.
    )

    # Backbone outputs a list of feature maps, use the last/deepest one.
    channels = model.feature_info.channels()[-1]
    print("    Backbone output channels:", channels)

    return model, channels


def load_config(config_path: str = "configs/default.yaml") -> DictConfig:
    """
    Load and parse YAML config file.

    Parameters
    ----------
    config_path: str
        Path to the YAML config file.

    Returns
    -------
    DictConfig
        Parsed configuration as a DictConfig object.
    """

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    return config
