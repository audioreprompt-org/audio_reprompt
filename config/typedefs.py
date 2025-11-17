from dataclasses import dataclass, field
from typing import Any


# Configuration data classes for type safety and auto-completion
@dataclass
class DataConfig:
    """Data configuration settings."""

    raw_data_path: str
    cleaned_data_path: str
    train_data_path: str
    tracks_data_path: str
    tracks_base_data_path: str
    data_docs_path: str
    data_prompts_path: str
    data_clap_path: str
    embeddings_csv_path: str
    reprompts_csv_path: str
    batch_size: int
    max_audio_length: int
    sample_rate: int


@dataclass
class ModelConfig:
    """Model configuration settings."""

    name: str
    model_musicgen_path: str
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
    _raw_config: dict[str, Any] = field(default_factory=dict, repr=False)
