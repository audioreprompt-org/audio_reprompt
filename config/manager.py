import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from config.types import Config, DataConfig, EnvironmentConfig, InferenceConfig, LoggingConfig, ModelConfig, TrainingConfig


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
