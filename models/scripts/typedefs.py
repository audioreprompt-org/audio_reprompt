from dataclasses import dataclass


# Tipos
@dataclass
class MusicGenData:
    """Estructura de datos para resultados de generación musical."""

    id: str
    prompt: str
    audio_path: str


@dataclass
class MusicGenCLAPResult(MusicGenData):
    """Extiende MusicGenData para incluir el CLAP Score."""

    clap_score: float
