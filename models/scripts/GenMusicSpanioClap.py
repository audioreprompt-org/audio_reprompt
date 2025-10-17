import csv
import os
import pandas as pd
import json
import numpy as np

import torch
import torchaudio
import scipy.io.wavfile
from laion_clap import CLAP_Module
from torch.utils.data import Dataset
from torch import serialization
from tqdm import tqdm

from transformers import pipeline
from tqdm import tqdm
from dataclasses import dataclass, field
from config import load_config, setup_project_paths, PROJECT_ROOT

import laion_clap.clap_module.factory as factory
import laion_clap.hook as hook


def patched_create_model(amodel_name: str, *args, **kwargs):
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
    Patches factory.load_state_dict:
    1. Resuelve el error de seguridad (numpy global).
    2. Asegura que NO se salten los parámetros ('skip_params=False' implícito)
       para que se carguen los pesos de audio (HTSAT) y texto (RoBERTa).
    """

    # Define los globals requeridos por el checkpoint.
    safe_globals = ["numpy.core.multiarray.scalar"]

    print(
        f"✅ Aplicando parche de seguridad para cargar el checkpoint: {checkpoint_path}"
    )

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


factory.load_state_dict = patched_load_state_dict
print(
    "Patch aplicado: factory.load_state_dict modificado para carga segura y completa de pesos."
)

if not hasattr(np.random, "integers"):
    print("Aplicando parche de compatibilidad: np.random.integers -> np.random.randint")
    # Crear un alias para que las llamadas internas a 'integers' usen 'randint'.
    np.random.integers = np.random.randint


# Tipos
@dataclass
class MusicGenData:
    """Estructura de datos para resultados de generación musical."""

    id: str
    instrument: str
    description: str
    audio_path: str


@dataclass
class MusicGenCLAPResult(MusicGenData):
    """Extiende MusicGenData para incluir el CLAP Score."""

    clap_score: float


class LoadSpanioDataset(Dataset):
    """
    Clase `Dataset` para cargar, transformar y exportar descripciones musicales
    del conjunto de datos de Spanio (`taste-music-dataset`).

    Esta clase permite leer un archivo JSON con estructura de columnas o lista
    de registros, convertirlo a una lista de diccionarios individuales,
    acceder a sus elementos por índice y exportarlos a CSV.

    Attributes:
        json_file_path (str): La ruta al archivo JSON que contiene los datos de Spanio.
    """

    def __init__(self, json_file_path):
        """
        Inicializa la clase `LoadSpanioDataset` cargando el contenido del archivo JSON.

        Parameters:
            son_file_path (str): La ruta al archivo JSON que contiene los datos de Spanio.
        """
        super().__init__()
        self.json_file_path = json_file_path
        self.records = []
        self._load_data()

    def _load_data(self):
        """
        Carga los datos del archivo JSON y normaliza su estructura.
        """
        try:
            with open(self.json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalizar estructura (lista o dict).
                self.records = (
                    data
                    if isinstance(data, list)
                    else [{"id": k, **v} for k, v in data.items()]
                )
            print(f"Cargados {len(self.records)} registros desde {self.json_file_path}")
        except Exception as e:
            print(f"Error cargando {self.json_file_path}: {e}")

    def __len__(self):
        """
        Retorna el número total de registros (extractos) en el dataset.

        Returns:
            int: Número de registros disponibles.
        """
        return len(self.records)

    def __getitem__(self, idx):
        """
        Retorna un registro específico del dataset por índice.

        Parameters:
            idx (int): Índice del registro a retornar.

        Returns
            dict: Diccionario con las llaves `id`, `instrument` y `description`.
        """
        if not 0 <= idx < len(self.records):
            raise IndexError(
                f"indice {idx} fuera de rango para dataset: {len(self.records)}."
            )
        return self.records[idx]

    def map_records(self):
        """
        Mapea los registros de self.records() a un diccionario.

        Cada clave del diccionario es el 'id' del registro, y su valor es otro
        diccionario con el 'content' y 'title' del registro.

        Returns

        dict:
            - Diccionario donde las llaves son los `id` de los registros y los valores.
            - son diccionarios con `instrument` y `description`.

        """
        return {
            doc["id"]: {
                "instrument": doc["instrument"],
                "description": doc["description"],
            }
            for doc in self.records
        }

    def to_csv(self, output_path="spanio_prompts.csv"):
        """
        Exporta los registros del dataset a un archivo CSV con columnas:
        `id`, `instrument`, `description`.

        Parameters
            output_path (str): Ruta de salida donde se guardará el archivo CSV.

        """
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=["id", "instrument", "description"]
                )
                writer.writeheader()
                writer.writerows(self.records)
            print(f"CSV generado en: {output_path}")
        except Exception as e:
            print(f"Error exportando a CSV: {e}")


def generate_music_from_prompts(
    synthesiser, dataset, output_dir="generated_music", sample_rate=32000
) -> list[MusicGenData]:
    """
    Genera archivos de audio a partir de descripciones de texto usando el modelo tasty-musicgen-small.

    Parameters:
        synthesiser: Pipeline de Hugging Face para text-to-audio.
        dataset: Instancia de LoadSpanioDataset con prompts.
        output_dir (str): Carpeta donde guardar los .wav generados.
        sample_rate (int): Frecuencia de muestreo para los archivos de salida. MusicGen fue entrenado a 32 kHz.

    Returns:
        list[dict]: Lista con {'id', 'instrument', 'description', 'audio_path'} por cada generación.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[MusicGenData] = []

    print(f"Generando música para {len(dataset)} prompts...\n")

    for record in tqdm(dataset.records):
        text_prompt = record["description"]
        file_id = record["id"]

        try:
            # 1. Generar la música con el modelo.
            # output es un diccionario: audio(array NumPy con la señal de audio) y sampling_rate (frecuencia de muestreo del modelo).
            output = synthesiser(text_prompt, forward_params={"do_sample": True})

            # 2. Extraer datos del audio.
            audio_data = output["audio"]  # La onda de sonido (las muestras del audio).
            sr = output.get(
                "sampling_rate", sample_rate
            )  # La frecuencia de muestreo reportada por el modelo.

            # 3. Guardar el audio generado.
            output_path = os.path.join(output_dir, f"{file_id}.wav")
            scipy.io.wavfile.write(
                output_path, rate=sr, data=audio_data
            )  # Escribir el archivo .wav con la señal y la frecuencia.

            # 4. Registrar los resultados.
            results.append(
                MusicGenData(
                    id=file_id,
                    instrument=record["instrument"],
                    description=text_prompt,
                    audio_path=output_path,
                )
            )
        except Exception as e:
            print(f" Error generando {file_id}: {e}")
            continue

    print(f"\n {len(results)} archivos de audio generados en: {output_dir}")
    return results


def compute_clap_scores(
    results: list[MusicGenData], device=None
) -> list[MusicGenCLAPResult]:
    """
    Calcula el CLAP Score (similaridad texto-audio) usando embeddings del modelo CLAP.c

    Parameters:
        results (list[dict]): Lista de diccionarios con llaves 'audio_path' y 'description'.
        device (str, opcional): Dispositivo ('cuda' o 'cpu'). Si None, detecta automáticamente.

    Returns:
        list[dict]: Misma lista de entrada, agregando la llave 'clap_score' (float).
    """
    # 1. Configurar dispositivo.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}\n")

    # 2. Cargar modelo CLAP.
    clap_model = CLAP_Module(
        enable_fusion=True,
        amodel="HTSAT-base",  # Modelo de 1024 dims, compatible con el checkpoint con los pesos.
    )  # Activa la modalidad combinada audio-texto del modelo.

    state_dict = factory.load_state_dict(laion_clap_path, map_location=device)
    clap_model.model.load_state_dict(state_dict, strict=False)

    # clap_model.load_ckpt(laion_clap_path)  # Descarga y carga los pesos preentrenados.
    clap_model.eval()  # Modo evaluación (desactiva dropout, gradientes, etc.).
    clap_model.to(device)

    print("Modelo CLAP cargado correctamente.\nCalculando CLAP Scores...\n")

    scored: list[MusicGenCLAPResult] = []

    # 3. Iterar sobre los resultados.
    for r in tqdm(results, desc="Procesando audios", ncols=80):
        # 4. Carga y preprocesamiento del audio.
        try:
            audio, sr = torchaudio.load(
                r.audio_path
            )  # Carga el audio en un tensor y su frecuencia de muestreo.
            if sr != 48000:
                audio = torchaudio.functional.resample(
                    audio, sr, 48000
                )  # Si no está a 48 kHz, lo resamplea.
            audio = audio.to(device)  # Enviar a dispositivo.

            with torch.no_grad():
                # 5. Obtener embeddings.
                audio_emb = clap_model.get_audio_embedding_from_data(
                    audio, use_tensor=True
                )
                text_emb = clap_model.get_text_embedding(
                    [r.description], use_tensor=True
                )

                # 6. Normalización y cálculo de similitud.
                audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
                text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

                # Calcular similitud coseno.
                score = torch.nn.functional.cosine_similarity(
                    audio_emb, text_emb
                ).item()

            # 7. Guardar el resultado.
            clap_score = round(float(score), 6)
            scored.append(
                MusicGenCLAPResult(
                    id=r.id,
                    instrument=r.instrument,
                    description=r.description,
                    audio_path=r.audio_path,
                    clap_score=clap_score,
                )
            )

        except Exception as e:
            print(f"Error calculando CLAP para {r.id} ({r.audio_path}): {e}")
    return scored


# Variables y constantes.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")

print("Configurando las rutas del proyecto...")
setup_project_paths()

print("Cargando configuración...")
config = load_config()

data_docs_path = PROJECT_ROOT / config.data.data_docs_path / "descriptions.json"
data_prompts_path = PROJECT_ROOT / config.data.data_prompts_path / "spanio_prompts.csv"
model_musicgen_path = PROJECT_ROOT / config.model.model_musicgen_path
laion_clap_path = PROJECT_ROOT / config.model.laion_clap_path
tracks_data_path = PROJECT_ROOT / config.data.tracks_data_path
data_clap_path = PROJECT_ROOT / config.data.data_clap_path / "results_with_clap2.csv"


# Pipeline.
if __name__ == "__main__":
    dataset = LoadSpanioDataset(data_docs_path)
    dataset.records = dataset.records[3:4]

    synthesiser = pipeline(
        "text-to-audio",
        model=model_musicgen_path,
        device=DEVICE,
        trust_remote_code=True,
    )

    results = generate_music_from_prompts(synthesiser, dataset, tracks_data_path)

    print(results)
    print(type(results))
    scored_results = compute_clap_scores(results)

    df = pd.DataFrame(scored_results)
    df.to_csv(data_clap_path, index=False)
    print("\nPipeline completo: resultados guardados en csv")
