# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=8
    )
    return embeddings.astype("float32")
