import random
from pathlib import Path
import numpy as np
from scipy import linalg
import logging

from models.descriptors.model import clap_model

logger = logging.getLogger(__name__)


def load_audio_and_extract_embeddings(audio_dir, model, max_files=None):
    """Load audio files and extract CLAP embeddings."""
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    if max_files:
        audio_files = audio_files[:max_files]

    audio_paths = [str(f) for f in audio_files]
    embeddings = []

    for i in range(0, len(audio_paths), 10):
        batch_paths = audio_paths[i : i + 10]
        try:
            batch_embeddings = model.get_audio_embedding_from_filelist(
                x=batch_paths, use_tensor=False
            )
            embeddings.append(batch_embeddings)
            print(
                f"  Processed {min(i + 10, len(audio_paths))}/{len(audio_paths)} files"
            )
        except Exception as e:
            logger.warning(f"Error processing batch {i // 10}: {e}")
            continue

    return np.vstack(embeddings)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fad(path_real_audio, path_generated_audio, max_files=None):
    """Calculate FAD score using CLAP embeddings. Lower is better."""
    print("Loading CLAP model")
    model = clap_model()

    print(f"Extracting embeddings from real audio")
    real_embeddings = load_audio_and_extract_embeddings(
        path_real_audio, model, max_files
    )

    print(f"Extracting embeddings from RAG audio")
    gen_embeddings = load_audio_and_extract_embeddings(
        path_generated_audio, model, max_files
    )

    print("Calculating FAD")
    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)

    mu_gen = np.mean(gen_embeddings, axis=0)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)

    fad_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    return fad_score


def compare_paired_samples(real_path, gen_path, model):
    real_embed = model.get_audio_embedding_from_filelist([real_path], use_tensor=False)
    gen_embed = model.get_audio_embedding_from_filelist([gen_path], use_tensor=False)

    # Cosine similarity
    cosine_sim = np.dot(real_embed[0], gen_embed[0]) / (
        np.linalg.norm(real_embed[0]) * np.linalg.norm(gen_embed[0])
    )
    cosine_dist = 1 - cosine_sim

    return cosine_dist, cosine_sim


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    path_real = "data/tracks/guedes_music"
    path_generated = "data/tracks/rag_music"
    #
    # fad_score = calculate_fad(path_real, path_generated)
    # print(f"\nFAD Score: {fad_score:.4f}")

    for item in random.sample(range(1, 100), 10):
        cosine_dist, cosine_sim = compare_paired_samples(
            Path(path_real) / f"{item}.mp3",
            Path(path_generated) / f"{item}.wav",
            clap_model(),
        )

        print(f"item metrics: {item}")
        print(f"cosine distance: {cosine_dist}")
        print(f"cosine similarity: {cosine_sim}")
