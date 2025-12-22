from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo de embeddings (RoBERTa)
MODEL_NAME = "sentence-transformers/all-roberta-large-v1"
model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    """
    Genera embeddings normalizados para una lista de textos
    """
    return model.encode(texts, normalize_embeddings=True)

def search_similar(query, documents, doc_embeddings, top_k=1):
    """
    Búsqueda semántica simple por similitud coseno
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    similarities = np.dot(doc_embeddings, query_embedding)

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [documents[i] for i in top_indices]
