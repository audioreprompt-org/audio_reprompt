import glob
import numpy as np
from scipy import linalg
import logging

from models.clap_score import ClapModel
from config import setup_project_paths, load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calcula la distancia de Fréchet entre dos distribuciones gaussianas."""
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


def calculate_fad(path_audio_samples: str| list[str], path_generated_samples: str|list[str], taste_paths: bool = False):
    """Calcula el puntaje FAD usando embeddings CLAP. Menor es mejor."""

    if taste_paths:
        prompt_audios = path_audio_samples
        reprompt_audios = path_generated_samples
    else:
        prompt_audios = [str(path) for path in glob.glob(f"{path_audio_samples}/*.wav")]
        reprompt_audios = [str(path) for path in glob.glob(f"{path_generated_samples}/**/*.wav")]

    logger.info(f"Extracting embeddings from source audio")
    real_embeddings = model.embed_audio(prompt_audios).numpy()

    logger.info(f"Extracting embeddings from generated audio")
    gen_embeddings = model.embed_audio(reprompt_audios).numpy()

    logger.info("Calculating FAD")

    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)

    mu_gen = np.mean(gen_embeddings, axis=0)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def compare_audio_taste_samples(spanio_track_dir: str, reprompt_track_dir: str, taste_value: str):
    """Calcula FAD para una categoría de sabor específica."""
    prompt_audios = [str(path) for path in glob.glob(f"{spanio_track_dir}/{taste_value}_*.wav")]
    reprompt_audios = [str(path) for path in glob.glob(f"{reprompt_track_dir}/{taste_value}/*.wav")]

    return calculate_fad(prompt_audios, reprompt_audios, taste_paths=True)


if __name__ == "__main__":
    setup_project_paths()
    config = load_config()
    AUDIOS_PATH = (PROJECT_ROOT / config.data.tracks_data_path).parent

    audios_without_reprompt = AUDIOS_PATH / "raw_prompts_audios"
    audios_with_reprompt = AUDIOS_PATH / "reprompt_audio_taste"
    audios_spanio = AUDIOS_PATH / "generated_base_music"

    model = ClapModel(device="auto", enable_fusion=True)

    # overall fad prompt vs reprompt audios
    print(calculate_fad(audios_without_reprompt, audios_with_reprompt))

    # overall fad spanio vs reprompt audios
    print(calculate_fad(audios_spanio, audios_with_reprompt))

    # comparison using taste for spanio vs reprompt audios
    print(compare_audio_taste_samples(audios_spanio, audios_with_reprompt, "sweet"))
    print(compare_audio_taste_samples(audios_spanio, audios_with_reprompt, "bitter"))
    print(compare_audio_taste_samples(audios_spanio, audios_with_reprompt, "salty"))
    print(compare_audio_taste_samples(audios_spanio, audios_with_reprompt, "sour"))
