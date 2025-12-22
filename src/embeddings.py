<<<<<<< HEAD
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import numpy as np

# Usamos un modelo multilingual por compatibilidad con documentos en español
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    """
    Genera embeddings en float32 para una lista de textos.
    Devuelve un numpy array con dtype float32.
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=8
    )
    return embeddings.astype("float32")


def search_similar(query, documents, doc_embeddings, top_k=1):
    """
    Búsqueda semántica simple por similitud de producto punto (vectores L2 normalizados).
    """
    query_embedding = model.encode([query], convert_to_numpy=True)[0].astype("float32")

    # Asumimos embeddings ya normalizados; usamos producto punto como similitud.
    similarities = np.dot(doc_embeddings, query_embedding)

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [documents[i] for i in top_indices]
        texts,
