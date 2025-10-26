import os
import pandas as pd
import numpy as np
import json
import random

import torch
import torchaudio
from laion_clap import CLAP_Module
from torch import serialization
from tqdm import tqdm

import laion_clap.clap_module.factory as factory
import laion_clap.hook as hook
from models.scripts.types import MusicGenCLAPResult, MusicGenData
from config import load_config, setup_project_paths, PROJECT_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")

print("Configurando las rutas del proyecto...")
setup_project_paths()

print("Cargando configuración...")
config = load_config()

tracks_base_data_path = PROJECT_ROOT / config.data.tracks_base_data_path
data_clap_path = (
    PROJECT_ROOT
    / config.data.data_clap_path
    / "results_with_clap_different_weights.csv"
)

laion_clap_path = PROJECT_ROOT / config.model.laion_clap_path

embeddings_csv_path = (
    PROJECT_ROOT
    / config.data.embeddings_csv_path
    / "music_prompt_embeddings_different_weights.csv"
)


def set_reproducibility(seed: int = 42):
    """
    Fija todas las semillas y configuraciones necesarias para que
    los resultados (embeddings, scores, etc.) sean reproducibles en CLAP o PyTorch.
    """
    print(f"Estableciendo modo determinista con semilla {seed}")

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modo determinista completo (puede afectar rendimiento, pero asegura reproducibilidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("Semillas y modo determinista configurados.\n")


def patched_create_model(amodel_name: str, *args, **kwargs):
    """
    Intercepta la llamada que crea el modelo y reemplaza el identificador "music_audioset" por
    "HTSAT-base" antes de que el código interno lo procese.
    """
    if amodel_name == "music_audioset":
        print("🎵 Usando modelo 'music_audioset' (alias de HTSAT-base)")
        amodel_name = "HTSAT-base"
    return (
        factory._create_model(amodel_name, *args, **kwargs)
        if hasattr(factory, "_create_model")
        else factory.create_model(amodel_name, *args, **kwargs)
    )


# Guardar referencia al original.
factory._create_model = getattr(factory, "create_model", None)

# Reemplazar en ambos lugares.
factory.create_model = patched_create_model
hook.create_model = patched_create_model

print(
    "Parche aplicado: CLAP_Module ahora reconoce 'music_audioset' como alias de 'HTSAT-base'"
)


def patched_load_state_dict(checkpoint_path, map_location="cpu"):
    """
    Modifica cómo se cargan los pesos (state_dict) del checkpoint.

    Patches factory.load_state_dict:
    1. Resuelve el error de seguridad (numpy global).
    2. Asegura que NO se salten los parámetros ('skip_params=False' implícito)
       para que se carguen los pesos de audio (HTSAT) y texto (RoBERTa).
    """

    # Define los globals requeridos por el checkpoint.
    safe_globals = ["numpy.core.multiarray.scalar"]

    print(f"Aplicando parche de seguridad para cargar el checkpoint: {checkpoint_path}")

    # 1. Usar el context manager para permitir los globals.
    # 2. Usar weights_only=False, como sugiere el error de PyTorch.
    with serialization.safe_globals(safe_globals):
        # Cargar el archivo de checkpoint completo.
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

    # Extraer el 'state_dict'. La mayoría de los checkpoints de PyTorch guardan los pesos aquí.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Si el checkpoint es solo el state_dict.
        state_dict = checkpoint

    return state_dict


# factory.load_state_dict = patched_load_state_dict
# print(
#    "Patch aplicado: factory.load_state_dict modificado para carga segura y completa de pesos."
# )

if not hasattr(np.random, "integers"):
    """
    Crea un alias integers → randint.
    """
    print("Aplicando parche de compatibilidad: np.random.integers -> np.random.randint")
    # Crear un alias para que las llamadas internas a 'integers' usen 'randint'.
    np.random.integers = np.random.randint


def compute_clap_scores(
    results: list[MusicGenData],
    device=None,
    save_embeddings=True,
) -> list[MusicGenCLAPResult]:
    """
    Calcula el CLAP Score (similaridad texto-audio) usando embeddings del modelo CLAP.
    """
    set_reproducibility(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}\n")

    # 1. Cargar modelo CLAP
    clap_model = CLAP_Module(
        enable_fusion=True,
        amodel="HTSAT-base",  # Modelo de 1024 dims, compatible con el checkpoint con los pesos.
    )  # Activa la modalidad combinada audio-texto del modelo.

    state_dict = factory.load_state_dict(laion_clap_path, map_location=device)
    clap_model.model.load_state_dict(state_dict, strict=False)

    # clap_model.load_ckpt(str(laion_clap_path))  # Descarga y carga los pesos preentrenados.
    clap_model.eval()  # Modo evaluación (desactiva dropout, gradientes, etc.).
    clap_model.to(device)

    print("Modelo CLAP cargado correctamente.\nCalculando CLAP Scores...\n")

    scored: list[MusicGenCLAPResult] = []
    embedding_records = []

    # 2. Iterar sobre los resultados
    for r in tqdm(results, desc="Procesando audios", ncols=80):
        try:
            audio, sr = torchaudio.load(r.audio_path)
            if sr != 48000:
                audio = torchaudio.functional.resample(audio, sr, 48000)
            audio = audio.to(device)

            with torch.no_grad():
                audio_emb = clap_model.get_audio_embedding_from_data(
                    audio, use_tensor=True
                )
                text_emb = clap_model.get_text_embedding(
                    [r.description], use_tensor=True
                )

                audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

                score = torch.nn.functional.cosine_similarity(
                    audio_emb, text_emb
                ).item()

            audio_emb_np = audio_emb.cpu().numpy().flatten().tolist()
            text_emb_np = text_emb.cpu().numpy().flatten().tolist()

            clap_score = round(float(score), 6)
            scored.append(
                MusicGenCLAPResult(
                    id=r.id,
                    taste=r.taste,
                    description=r.description,
                    instrument=r.instrument,
                    audio_path=r.audio_path,
                    clap_score=clap_score,
                )
            )

            if save_embeddings:
                embedding_records.append(
                    {
                        "taste": r.taste,
                        "description": r.description,
                        "audio_name": r.id,
                        "clap_score": clap_score,
                        "audio_emb": json.dumps(audio_emb_np),
                        "text_emb": json.dumps(text_emb_np),
                    }
                )

        except Exception as e:
            print(f"Error procesando {r.id}: {e}")

    if save_embeddings and embedding_records:
        df = pd.DataFrame(embedding_records)
        df.to_csv(embeddings_csv_path, index=False)
        print(f"\n Embeddings y CLAP Scores guardados en: {embeddings_csv_path}\n")
    return scored


# 1. Construir lista de audios generados existentes.
print(f"\nBuscando audios en: {tracks_base_data_path}")
audio_files = [f for f in os.listdir(tracks_base_data_path) if f.endswith(".wav")]

if not audio_files:
    raise FileNotFoundError(f"No se encontraron audios en {tracks_base_data_path}")

print(f"Se encontraron {len(audio_files)} archivos de audio.")

# Inferir metadata a partir del nombre de archivo.
results: list[MusicGenData] = []
for fname in audio_files:
    audio_path = os.path.join(tracks_base_data_path, fname)
    file_id = os.path.splitext(fname)[0]
    taste = file_id.split("_")[0] if "_" in file_id else "unknown"
    description = f"{taste} music, ambient for fine restaurant"

    results.append(
        MusicGenData(
            id=file_id,
            taste=taste,
            instrument="N/A",
            description=description,
            audio_path=audio_path,
        )
    )

print(f"Preparados {len(results)} registros para evaluación CLAP.\n")

# 2. Calcular CLAP Scores.
scored_results = compute_clap_scores(results, device=DEVICE)

# 3. Guardar resultados.
df = pd.DataFrame(scored_results)
# df.to_csv(data_clap_path, index=False)
print(f"\nPipeline completo: resultados guardados en {data_clap_path}")
