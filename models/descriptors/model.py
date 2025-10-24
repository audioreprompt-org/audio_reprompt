import torch
import logging

from laion_clap import CLAP_Module

logger = logging.getLogger(__name__)

_CLAP_MODEL = None


def clap_model():
    global _CLAP_MODEL
    if _CLAP_MODEL is not None:
        return _CLAP_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAP_Module(enable_fusion=False)

    weights_url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, map_location=device, weights_only=False
        )
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error loading model weights from {weights_url}: {e}")
        raise

    model.to(device)
    model.eval()

    logger.info(f"Modelo CLAP cargado correctamente en {device}")
    _CLAP_MODEL = model
    return model
