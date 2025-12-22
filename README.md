# 🧠 NER + RAG + LLaMA (PoC local en español)

Proof of Concept (PoC) para **extracción de entidades nombradas (NER)**
en español, combinando:

-   🧩 spaCy (NER base)
-   🔎 RAG (Retrieval Augmented Generation)
-   🧠 Embeddings multilingües (Sentence Transformers)
-   📦 FAISS (vector store)
-   🦙 LLaMA local vía Ollama
-   💻 Ejecución 100% local (VS Code + Windows)

El objetivo es demostrar un **pipeline end-to-end** para enriquecer la
extracción de entidades usando contexto recuperado y un LLM.

------------------------------------------------------------------------

## 🏗️ Arquitectura

Texto → Chunking → Embeddings → FAISS\
Query → Embedding → Recuperación de contexto → LLaMA → JSON de entidades

------------------------------------------------------------------------

## 📂 Estructura del proyecto

ner-rag-poc/ ├── data/ │ └── sample.txt ├── src/ │ ├── ingest.py │ ├──
ner_spacy.py │ ├── embeddings.py │ ├── vector_store.py │ ├──
llm_ollama.py │ ├── prompts.py │ └── pipeline_rag_llama.py ├──
test_spacy.py └── README.md

------------------------------------------------------------------------

## ⚙️ Requisitos

-   Python 3.10
-   Conda
-   Git
-   VS Code
-   Ollama

------------------------------------------------------------------------

## 🐍 Setup rápido

conda create -n ner-rag python=3.10 -y\
conda activate ner-rag

pip install spacy==3.5.4 sentence-transformers faiss-cpu requests

------------------------------------------------------------------------

## 🦙 LLaMA local

ollama pull llama3:8b\
ollama serve

------------------------------------------------------------------------

## ▶️ Ejecución

python src/pipeline_rag_llama.py

------------------------------------------------------------------------

## 📤 Output esperado

{ "PERSONA": \["Hugo Villegas"\], "ORGANIZACION": \["YPF"\], "FECHA":
\["2024"\], "LUGAR": \[\], "OTROS": \["spaCy", "LLaMA"\] }

------------------------------------------------------------------------

## 👤 Autor

Leonardo Villegas\
https://github.com/Leonardorf
