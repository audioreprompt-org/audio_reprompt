from sentence_transformers import SentenceTransformer

# no ideal but
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def encode_text(items: list[str]) -> list[dict[str, str | list[float]]]:
    return [
        {"prompt": text, "text_embedding": emb.tolist()}
        for text, emb in zip(items, model.encode(items))
    ]
