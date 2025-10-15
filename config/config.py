"""
Configuration module for the audio reprompt project.

This module provides functions to load and access configuration settings
from a YAML file located at the repository root.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from config.manager import ConfigManager
from config.types import Config, DataConfig, EnvironmentConfig, InferenceConfig, LoggingConfig, ModelConfig, \
    TrainingConfig, EvaluationConfig
from config import PROJECT_ROOT

# Global config manager instance
_config_manager = ConfigManager()


# Public API functions
def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to configuration file. If not provided,
                    will search for config.yaml in current directory and parents.
    
    Returns:
        Config object with all configuration settings.
    
    Raises:
        ConfigurationError: If configuration file cannot be found or loaded.
    """
    return _config_manager.load_config(config_path)


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Force reload configuration from YAML file.
    
    Args:
        config_path: Optional path to configuration file.
    
    Returns:
        Config object with all configuration settings.
    """
    return _config_manager.reload_config(config_path)


def get_data_config() -> DataConfig:
    """Get data configuration settings."""
    return load_config().data


def get_model_config() -> ModelConfig:
    """Get model configuration settings."""
    return load_config().model


def get_training_config() -> TrainingConfig:
    """Get training configuration settings."""
    return load_config().training


def get_inference_config() -> InferenceConfig:
    """Get inference configuration settings."""
    return load_config().inference


def get_logging_config() -> LoggingConfig:
    """Get logging configuration settings."""
    return load_config().logging


def get_environment_config() -> EnvironmentConfig:
    """Get environment configuration settings."""
    return load_config().environment


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration settings."""
    return load_config().evaluation


def get_raw_config() -> Dict[str, Any]:
    """Get raw configuration dictionary."""
    return load_config()._raw_config


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value using dot notation.
    
    Args:
        key_path: Dot-separated path to the configuration value (e.g., 'data.batch_size')
        default: Default value to return if key is not found
    
    Returns:
        Configuration value or default if not found.
    
    Examples:
        >>> get_config_value('data.batch_size')
        32
        >>> get_config_value('model.learning_rate')
        2e-05
        >>> get_config_value('nonexistent.key', 'default_value')
        'default_value'
    """
    config_dict = get_raw_config()
    keys = key_path.split('.')
    
    current = config_dict
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

# Convenience function for notebooks and scripts
def setup_project_paths():
    """
    Set up project paths based on configuration.
    Useful for notebooks and scripts to ensure proper path resolution.
    """
    config = load_config()
    
    # Create directories if they don't exist
    paths_to_create = [
        config.data.raw_data_path,
        config.data.cleaned_data_path,
        config.data.train_data_path,
        config.data.tracks_data_path,
        config.data.data_docs_path,
        config.data.data_prompts_path,
        config.data.data_clap_path,
        config.model.model_musicgen_path,
        config.training.checkpoint_dir,
        config.training.log_dir,
        config.inference.output_dir,
        os.path.dirname(config.logging.file)
    ]
    
    for path in paths_to_create:
        if path:
            Path(PROJECT_ROOT / path).mkdir(parents=True, exist_ok=True)
    
    return config