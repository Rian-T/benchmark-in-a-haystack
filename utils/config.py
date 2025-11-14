import yaml
import random
import numpy as np
import torch


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_models_dir(config):
    """
    Extract the models directory from config with fallback to default.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String path to the models directory
    """
    return config.get("models", {}).get("offline_dir", "models")

