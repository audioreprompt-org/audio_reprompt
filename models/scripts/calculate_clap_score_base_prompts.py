import os
import pandas as pd

import torch

from models.clap_score.model import ClapModel, SPECIALIZED_WEIGHTS_URL
from models.clap_score.typedef import CLAPItem
from models.clap_score.utils import set_reproducibility, resolve_device
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
    / "results_with_clap_base_prompts_with_score.csv"
)


def compute_clap_scores(
    results: list[MusicGenData],
    device=None,
) -> list[MusicGenCLAPResult]:
    """
    Calcula SOLO el CLAP Score (similaridad texto-audio) usando el scorer por lotes.
    No guarda ni calcula embeddings.
    """
    set_reproducibility(42)
    device = resolve_device(device)
    print(f"\nUsando dispositivo: {device}\n")

    clap_score = ClapModel(device=device, enable_fusion=True, weights=SPECIALIZED_WEIGHTS_URL)

    if not results:
        return []

    # 1) Preparar los CLAPItem (mismo orden que 'results')
    items = [
        CLAPItem(
            id=r.id,
            prompt=r.prompt,
            audio_path=r.audio_path,
        )
        for r in results
    ]

    # 2) Calcular scores en batch con el backend seleccionado
    scored_batch = clap_score.calculate_scores(
        items,
    )  # list[CLAPScored]

    # 3) Mapear a MusicGenCLAPResult (redondeo y orden consistente)
    out: list[MusicGenCLAPResult] = []
    for r, sc in zip(results, scored_batch):
        clap_score = getattr(sc, "clap_score", None)
        out.append(
            MusicGenCLAPResult(
                id=r.id,
                prompt=r.prompt,
                audio_path=r.audio_path,
                clap_score=clap_score,
            )
        )
    return out


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
    prompt = f"{taste} music, ambient for fine restaurant"

    results.append(
        MusicGenData(
            id=file_id,
            prompt=prompt,
            audio_path=audio_path,
        )
    )

print(f"Preparados {len(results)} registros para evaluación CLAP.\n")

# 2. Calcular CLAP Scores (solo scores).
scored_results = compute_clap_scores(results, device=DEVICE)

# 3. Guardar resultados.
df = pd.DataFrame(scored_results)
df.to_csv(data_clap_path, index=False)
print(f"\nPipeline completo: resultados guardados en {data_clap_path}")
