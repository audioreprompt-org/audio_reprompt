import os
import pandas as pd
import json

import torch
import torchaudio
from laion_clap import CLAP_Module
from tqdm import tqdm


from models.scripts.types import MusicGenCLAPResult, MusicGenData
from config import load_config, setup_project_paths, PROJECT_ROOT
from utils.seed import set_reproducibility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")

print("Configurando las rutas del proyecto...")
setup_project_paths()

print("Cargando configuración...")
config = load_config()

tracks_base_data_path = PROJECT_ROOT / config.data.tracks_base_data_path
data_clap_path = (
    PROJECT_ROOT / config.data.data_clap_path / "results_with_clap_normal_weights.csv"
)

embeddings_csv_path = (
    PROJECT_ROOT
    / config.data.embeddings_csv_path
    / "music_prompt_embeddings_normal_weights.csv"
)


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
    )  # Activa la modalidad combinada audio-texto del modelo.

    clap_model.load_ckpt()  # Descarga y carga los pesos preentrenados.
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
