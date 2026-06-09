# src/pipeline_rag.py
from ingest import load_text, chunk_text
from ner_spacy import extract_entities
from embeddings import embed_texts
from vector_store import VectorStore

if __name__ == "__main__":
    # 1. Load & chunk
    text = load_text("data/sample.txt")
    chunks = chunk_text(text)

    # 2. Embeddings
    embeddings = embed_texts(chunks)

    # 3. Vector store
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    # 4. Query
    query = "experiencia en inteligencia artificial y datos"
    query_embedding = embed_texts([query])

    results = store.search(query_embedding, k=2)

    print("\n--- QUERY ---")
    print(query)

    print("\n--- CONTEXTO RECUPERADO (RAG) ---")
    for r in results:
        print("-", r)

    # 5. NER sobre contexto recuperado
    print("\n--- ENTIDADES EN CONTEXTO ---")
    for r in results:
        ents = extract_entities(r)
        print(ents)
