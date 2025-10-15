"""Configuration module for audio reprompt project."""

# Define the project root directory
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .manager import (
  ConfigurationError,
)

from .types import (
  Config,
  DataConfig,
  ModelConfig,
  TrainingConfig,
  InferenceConfig,
  LoggingConfig,
  EnvironmentConfig,
  EvaluationConfig,
)


from .config import (
    load_config,
    reload_config,
    get_data_config,
    get_model_config,
    get_training_config,
    get_inference_config,
    get_logging_config,
    get_environment_config,
    get_config_value,
    setup_project_paths,
    get_raw_config,
    get_evaluation_config
)

__all__ = [
    'PROJECT_ROOT',
    'load_config',
    'reload_config',
    'get_data_config',
    'get_model_config',
    'get_training_config',
    'get_inference_config',
    'get_logging_config',
    'get_environment_config',
    'get_config_value',
    'setup_project_paths',
    'get_raw_config',
    'get_evaluation_config',
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'LoggingConfig',
    'EnvironmentConfig',
    'ConfigurationError',
    'EvaluationConfig',
]
