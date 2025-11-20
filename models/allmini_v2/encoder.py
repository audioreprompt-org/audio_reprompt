from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def encode_text(items: list[str]) -> list[dict[str, str | list[float]]]:
    model = get_model()
    return [
        {"prompt": text, "text_embedding": emb.tolist()}
        for text, emb in zip(items, model.encode(items))
    ]
