# src/pipeline_rag_llama.py
from ingest import load_text, chunk_text
from embeddings import embed_texts
from vector_store import VectorStore
from llm_ollama import ask_llama
from prompts import ner_prompt

if __name__ == "__main__":
    text = load_text("data/sample.txt")
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    query = "experiencia profesional y tecnologías"
    query_embedding = embed_texts([query])

    context_chunks = store.search(query_embedding, k=2)
    context = "\n".join(context_chunks)

    prompt = ner_prompt(context)

    print("\n--- PROMPT ---")
    print(prompt)

    print("\n--- RESPUESTA LLaMA ---")
    response = ask_llama(prompt)
    print(response)
