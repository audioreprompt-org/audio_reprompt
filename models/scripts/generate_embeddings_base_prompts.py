import os
import pandas as pd
import json

import torch

from metrics.clap import get_audio_embeddings_from_paths, get_text_embeddings
from metrics.clap.backends.laion import MUSICGEN_WEIGHTS_URL
from metrics.clap.factory import calculate_scores_with_embeddings
from models.scripts.types import MusicGenCLAPResult, MusicGenData
from config import load_config, setup_project_paths, PROJECT_ROOT
from utils.device import resolve_device

from utils.seed import set_reproducibility


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
    / "results_with_clap_base_prompts_audioset_weights_disabled_fusion.csv"
)

embeddings_csv_path = (
    PROJECT_ROOT
    / config.data.embeddings_csv_path
    / "music_base_prompts_embeddings_audioset_weights_disabled_fusion.csv"
)


def compute_clap_scores(
    results: list[MusicGenData],
    device=None,
    save_embeddings=True,
    backend: str = "laion_module",
) -> list[MusicGenCLAPResult]:
    """
    Calcula el CLAP Score (similaridad texto-audio) usando embeddings del modelo CLAP.
    Procesa TODOS los audios y textos de una sola vez y usa calculate_scores_with_embeddings.
    """
    set_reproducibility(42)
    device = resolve_device(device)
    print(f"\nUsando dispositivo: {device}\n")

    if not results:
        return []

    # 1) Preparar listas en el mismo orden
    audio_paths = [r.audio_path for r in results]
    texts = [r.prompt for r in results]

    # 2) Obtener embeddings en batch (una sola llamada por tipo)
    audio_emb_list = get_audio_embeddings_from_paths(
        audio_paths,
        device,
        backend=backend,
        backend_cfg={"enable_fusion": False, "weights": MUSICGEN_WEIGHTS_URL},
    )
    text_emb_list = get_text_embeddings(
        texts,
        device,
        backend=backend,
        backend_cfg={"enable_fusion": False, "weights": MUSICGEN_WEIGHTS_URL},
    )

    # 3) Calcular similitudes con el helper (vectorizado)
    sims = calculate_scores_with_embeddings(audio_emb_list, text_emb_list)

    # 4) Construir resultados (sin tocar embeddings aún)
    scored: list[MusicGenCLAPResult] = []
    for i, r in enumerate(results):
        clap_score = round(float(sims[i]), 6)
        scored.append(
            MusicGenCLAPResult(
                id=r.id,
                prompt=r.prompt,
                audio_path=r.audio_path,
                clap_score=clap_score,
            )
        )

    # 5) (Opcional) Guardar embeddings normalizados y scores
    if save_embeddings:
        a_norm = torch.nn.functional.normalize(
            torch.stack(audio_emb_list, dim=0), dim=-1
        )
        t_norm = torch.nn.functional.normalize(
            torch.stack(text_emb_list, dim=0), dim=-1
        )

        embedding_records = []
        for i, r in enumerate(results):
            audio_emb_np = a_norm[i].detach().cpu().numpy().flatten().tolist()
            text_emb_np = t_norm[i].detach().cpu().numpy().flatten().tolist()
            embedding_records.append(
                {
                    "prompt": r.prompt,
                    "audio_name": r.id,
                    "clap_score": scored[i].clap_score,
                    "audio_emb": json.dumps(audio_emb_np),
                    "text_emb": json.dumps(text_emb_np),
                }
            )

        if embedding_records:
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
    prompt = f"{taste} music, ambient for fine restaurant"

    results.append(
        MusicGenData(
            id=file_id,
            prompt=prompt,
            audio_path=audio_path,
        )
    )

print(f"Preparados {len(results)} registros para evaluación CLAP.\n")

# 2. Calcular CLAP Scores.
scored_results = compute_clap_scores(results, device=DEVICE)

# 3. Guardar resultados.
df = pd.DataFrame(scored_results)
df.to_csv(data_clap_path, index=False)
print(f"\nPipeline completo: resultados guardados en {data_clap_path}")
