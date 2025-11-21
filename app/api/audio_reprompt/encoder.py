from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        # Downloads model once (~80MB) and caches it
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def encode_text(items: list[str]) -> list[dict]:
    model = get_model()
    embeddings = model.encode(items)
    return [
        {"prompt": text, "text_embedding": emb.tolist()}
        for text, emb in zip(items, embeddings)
    ]
