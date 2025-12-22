import spacy
from embeddings import embed_texts, search_similar
from llm_ollama import ask_llama

# =========================
# 1. Cargar modelo spaCy ES
# =========================
nlp = spacy.load("es_core_news_md")

# =========================
# 2. Documentos judiciales (simulados)
# =========================
documents = [
    """
    En el expediente EXP-2021-00458723-GDEMSA, el Juzgado Civil y Comercial N° 3
    resolvió la causa iniciada por Pérez, Juan contra la Dirección de Registros Públicos.
    La resolución fue archivada en el Archivo Judicial Provincial.
    """,
    """
    Durante el año 2022, la Dra. María González intervino como magistrada
    en diversas causas tramitadas en la ciudad de Mendoza.
    """
]

# =========================
# 3. Enriquecer documentos con NER
# =========================
def enrich_with_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    enriched_text = text + "\n\nENTIDADES DETECTADAS:\n"
    for ent, label in entities:
        enriched_text += f"- {ent} ({label})\n"
    return enriched_text

enriched_docs = [enrich_with_entities(d) for d in documents]

# =========================
# 4. Generar embeddings
# =========================
embeddings = embed_texts(enriched_docs)

# =========================
# 5. Consulta del usuario
# =========================
query = "¿En qué expediente intervino la Dirección de Registros Públicos?"

# =========================
# 6. Recuperar contexto relevante
# =========================
top_docs = search_similar(query, enriched_docs, embeddings, top_k=1)

context = "\n".join(top_docs)

# =========================
# 7. Prompt controlado (ANTI alucinación)
# =========================
# prompt = f"""
# Respondé únicamente con la información contenida en el siguiente contexto.
# Si la respuesta no está explícitamente en el texto, indicá que no se encontró información.

# CONTEXTO:
# {context}

# PREGUNTA:
# {query}

# RESPUESTA:
# """

prompt = f"""
Respondé en una oración completa y clara, utilizando únicamente la información
contenida en el contexto. No agregues interpretaciones ni información externa.
Si la respuesta no está explícitamente en el texto, indicá: "No se encontró información suficiente".

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

# =========================
# 8. LLaMA responde
# =========================
response = ask_llama(prompt)

print("\n--- RESPUESTA LLaMA ---\n")
print(response)
