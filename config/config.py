"""
Configuration module for the audio reprompt project.

This module provides functions to load and access configuration settings
from a YAML file located at the repository root.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Configuration data classes for type safety and auto-completion
@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_data_path: str
    cleaned_data_path: str
    train_data_path: str
    tracks_data_path: str
    batch_size: int
    max_audio_length: int
    sample_rate: int


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    max_sequence_length: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    dropout_rate: float
    learning_rate: float
    warmup_steps: int
    max_epochs: int


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    seed: int
    train_split: float
    val_split: float
    test_split: float
    checkpoint_dir: str
    log_dir: str
    save_every_n_epochs: int
    early_stopping_patience: int


@dataclass
class InferenceConfig:
    """Inference configuration settings."""
    output_dir: str
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    file: str


@dataclass
class EnvironmentConfig:
    """Environment configuration settings."""
    device: str
    num_workers: int
    pin_memory: bool


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    logging: LoggingConfig
    environment: EnvironmentConfig
    _raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)


class ConfigurationError(Exception):
    """Raised when there's an issue with configuration loading or validation."""
    pass


class ConfigManager:
    """Manages configuration loading and caching."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Config] = None
    _config_file_path: Optional[Path] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _find_config_file(self) -> Path:
        """Find the config.yaml file starting from the current directory and going up."""
        current_path = Path.cwd()
        
        # First, try to find the repository root by looking for specific markers
        markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']
        
        for path in [current_path] + list(current_path.parents):
            for marker in markers:
                if (path / marker).exists():
                    config_file = path / 'config.yaml'
                    if config_file.exists():
                        return config_file
        
        # If not found via markers, try current directory and parents
        for path in [current_path] + list(current_path.parents):
            config_file = path / 'config.yaml'
            if config_file.exists():
                return config_file
        
        raise ConfigurationError("config.yaml not found in current directory or any parent directory")
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                if config_data is None:
                    raise ConfigurationError(f"Configuration file {config_path} is empty")
                return config_data
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {config_path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {config_path}: {e}")
    
    def _create_config_objects(self, config_data: Dict[str, Any]) -> Config:
        """Create typed configuration objects from raw configuration data."""
        try:
            data_config = DataConfig(**config_data.get('data', {}))
            model_config = ModelConfig(**config_data.get('model', {}))
            training_config = TrainingConfig(**config_data.get('training', {}))
            inference_config = InferenceConfig(**config_data.get('inference', {}))
            logging_config = LoggingConfig(**config_data.get('logging', {}))
            environment_config = EnvironmentConfig(**config_data.get('environment', {}))
            
            return Config(
                data=data_config,
                model=model_config,
                training=training_config,
                inference=inference_config,
                logging=logging_config,
                environment=environment_config,
                _raw_config=config_data
            )
        except TypeError as e:
            raise ConfigurationError(f"Configuration validation error: {e}")
    
    def load_config(self, config_path: Optional[str] = None) -> Config:
        """Load configuration from YAML file."""
        if config_path:
            config_file_path = Path(config_path)
        else:
            config_file_path = self._find_config_file()
        
        # Only reload if the config file has changed
        if self._config is None or self._config_file_path != config_file_path:
            config_data = self._load_yaml_config(config_file_path)
            self._config = self._create_config_objects(config_data)
            self._config_file_path = config_file_path
        
        return self._config
    
    def reload_config(self, config_path: Optional[str] = None) -> Config:
        """Force reload configuration from YAML file."""
        self._config = None
        self._config_file_path = None
        return self.load_config(config_path)


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
        config.training.checkpoint_dir,
        config.training.log_dir,
        config.inference.output_dir,
        os.path.dirname(config.logging.file)
    ]
    
    for path in paths_to_create:
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    return config