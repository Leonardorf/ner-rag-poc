import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("HTTPS_PROXY", "http://<PROXY_HOST>:<PUERTO>")
os.environ.setdefault("HTTP_PROXY", "http://<PROXY_HOST>:<PUERTO>")
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_ollama import ask_llama, MODEL as OLLAMA_MODEL
from ner_patterns import PATRONES_REGISTRALES, DESCRIPCION_PATRONES

from rank_bm25 import BM25Okapi

LABEL_ES = {
    # Etiquetas estándar de spaCy
    "PER":          "Persona",
    "ORG":          "Organización",
    "LOC":          "Lugar",
    "MISC":         "Otro",
    "GPE":          "Lugar",
    "DATE":         "Fecha",
    "TIME":         "Hora",
    "MONEY":        "Monto",
    "CARDINAL":     "Número",
    "LAW":          "Norma legal",
    "EVENT":        "Evento",
    # Etiquetas registrales (definidas en ner_patterns.py)
    "NRO_EXPEDIENTE": "N° Expediente/Registro",
    "NRO_PADRON":     "N° de Padrón",
    "NRO_DNI":        "DNI",
    "NRO_ESCRITURA":  "N° de Escritura",
}

DEFAULT_DOCS = [
    (
        "Doc 1 — Inscripción de dominio",
        """En el expediente REG-2023-00012847-MZA se inscribió la transferencia de dominio
del inmueble ubicado en calle San Martín 1542, Ciudad de Mendoza, Padrón N° 45.231,
a favor de Rodríguez, Carlos Alberto, mediante escritura N° 88 otorgada por la
Escribana Laura Fernández. Vendedor: Gómez, Marta Susana.""",
    ),
    (
        "Doc 2 — Hipoteca bancaria",
        """El Banco Nación Argentina constituyó hipoteca en primer grado sobre el inmueble
Padrón N° 45.231, calle San Martín 1542, Mendoza, en favor del crédito otorgado a
Rodríguez, Carlos Alberto por la suma de $18.500.000.
Inscripción N° HIP-2023-00441. Plazo: 240 meses.""",
    ),
    (
        "Doc 3 — Inhibición general de bienes",
        """Por resolución del Juzgado Civil N° 7 de Mendoza, causa EXP-2022-00985-JC7,
se dispuso inhibición general de bienes sobre Pérez Villalba, Diego Hernán, DNI 28.441.902.
La medida fue comunicada al Registro Inmobiliario el 14 de marzo de 2023
para su anotación preventiva.""",
    ),
    (
        "Doc 4 — Embargo inmobiliario",
        """En los autos caratulados "Torres S.A. c/ Constructora Del Sur s/ cobro de pesos",
EXP-2021-00334-JCC, el Juzgado Civil y Comercial N° 2 ordenó embargo sobre el inmueble
Padrón N° 71.008, ubicado en Godoy Cruz, Mendoza,
de titularidad de Constructora Del Sur S.R.L.""",
    ),
    (
        "Doc 5 — Declaratoria de herederos",
        """En el expediente sucesorio EXP-2022-00761-SUC tramitado ante el Juzgado Civil N° 4
de Mendoza, se dictó declaratoria de herederos de Morales, Ana Beatriz, DNI 12.334.567,
fallecida el 3 de julio de 2021. Se declararon herederos universales a sus hijos
Morales, Lucas Adrián y Morales, Sofía Inés. El acervo hereditario incluye el inmueble
Padrón N° 38.902, ubicado en Las Heras, Mendoza.""",
    ),
    (
        "Doc 6 — Constitución de usufructo",
        """Por escritura N° 142 del 10 de febrero de 2023, otorgada ante la Escribana
Paula Ríos, se constituyó usufructo vitalicio sobre el inmueble Padrón N° 38.902,
Las Heras, Mendoza, a favor de Morales, Ana Rosa, DNI 9.112.045.
La nuda propiedad quedó en cabeza de Morales, Lucas Adrián.
Inscripción registral: USF-2023-00089-MZA.""",
    ),
]


@st.cache_resource(show_spinner="Cargando modelo de lenguaje español...")
def load_nlp():
    nlp = spacy.load("es_core_news_md")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(PATRONES_REGISTRALES)
    return nlp


def tfidf_scores(query, docs):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs + [query])
    return cosine_similarity(matrix[-1], matrix[:-1])[0]


def bm25_scores(query, docs):
    tokenized_docs = [d.lower().split() for d in docs]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    max_score = scores.max()
    return scores / max_score if max_score > 0 else scores


# ── Página ────────────────────────────────────────────────────────
st.set_page_config(page_title="Consultor de Expedientes", layout="wide", page_icon="⚖️")

st.title("⚖️ Consultor de Expedientes Judiciales")
st.markdown(
    "Este sistema lee documentos judiciales, identifica personas y organizaciones "
    "mencionadas, y responde preguntas usando **solo la información de los documentos**. "
    "No inventa datos ni consulta internet."
)
st.markdown(
    "Arquitectura: **NER + RAG** &nbsp;|&nbsp; "
    "Recuperación: TF-IDF &nbsp;|&nbsp; "
    "Generación: LLaMA 3.1 (local, sin internet)",
    unsafe_allow_html=True,
)

with st.expander("¿Cómo funciona? Explicación de los términos técnicos"):
    st.markdown("""
**Procesamiento del Lenguaje Natural (PLN)**
Es un conjunto de técnicas informáticas que permiten a las computadoras analizar y entender texto escrito por humanos.
A diferencia de la inteligencia artificial generativa, el PLN clásico **no crea contenido nuevo**: solo lee, mide y clasifica lo que ya está escrito.
Ejemplos cotidianos: el corrector ortográfico, los filtros de spam o los buscadores de texto.

---

**NER — Reconocimiento de Entidades Nombradas** *(del inglés: Named Entity Recognition)*
Es una técnica de PLN que lee un texto y marca automáticamente los elementos importantes: nombres de personas, organizaciones, lugares, fechas y números.
Funciona como un subrayador automático: **no inventa ni interpreta, solo señala lo que ya está ahí**.
Por ejemplo, en el texto *"el Juzgado Civil N° 3 resolvió la causa de Pérez, Juan"* identifica que *"Juzgado Civil N° 3"* es una organización y *"Pérez, Juan"* es una persona.

---

**TF-IDF — Frecuencia de término / Frecuencia inversa de documento** *(del inglés: Term Frequency–Inverse Document Frequency)*
Es una técnica matemática que mide qué tan importante es una palabra para un documento comparado con el resto.
Le da más peso a las palabras poco frecuentes (como un número de expediente) y menos peso a palabras comunes (como "el", "la", "en").
Se usa para encontrar cuál documento es más relevante para una pregunta. **No genera texto: es un cálculo numérico puro.**

---

**BM25 — Mejor Coincidencia 25** *(del inglés: Best Match 25 / Okapi BM25)*
Es una evolución de TF-IDF que agrega dos mejoras importantes:

1. **Saturación de frecuencia**: en TF-IDF, si un término aparece 10 veces en un documento pesa el doble que si aparece 5. En BM25 la relevancia crece más lento y se estabiliza: evita que documentos con muchas repeticiones "engañen" al sistema.
2. **Normalización por longitud**: un documento corto que menciona el término buscado una vez puede ser más relevante que uno largo que lo menciona muchas veces, porque en el documento largo puede ser una mención marginal. BM25 ajusta el score según la longitud promedio.

**Ejemplo registral**: si buscamos *"embargo Padrón N° 71.008"* y un documento de 3 líneas lo menciona una vez, BM25 lo considera más relevante que uno de 20 líneas que lo menciona dos veces en un contexto más general.
BM25 es el algoritmo que usan internamente buscadores como **Elasticsearch** y **Apache Solr**. **No genera texto: es matemática pura, siempre reproducible y auditable.**

---

**Modelo de lenguaje generativo (Ollama — LLM local)**
Es el único componente del sistema que genera texto nuevo.
Recibe el documento más relevante encontrado por los pasos anteriores y formula una respuesta en lenguaje natural.
Corre localmente en la misma máquina: **los datos del expediente nunca salen de la red interna**.
Para reducir el riesgo de que "invente" información, se le indica explícitamente que responda solo con lo que está en el documento.

---

**RAG — Generación con Recuperación Aumentada** *(del inglés: Retrieval Augmented Generation)*
Es la arquitectura general de este sistema. Combina dos mundos:
1. **Recuperación (Retrieval):** antes de responder, el sistema *busca* el fragmento de texto más relevante usando TF-IDF y BM25, enriquecidos con las entidades detectadas por NER.
2. **Generación aumentada:** el modelo de lenguaje recibe ese fragmento como contexto adicional (*augmented*) y genera la respuesta basándose en él, no en su memoria interna.

Sin RAG, el modelo respondería desde su entrenamiento general, lo que aumenta el riesgo de inventar datos.
Con RAG, el modelo tiene delante el texto original del expediente y sus respuestas pueden verificarse contra ese documento.

---

**Reglas de dominio (EntityRuler)**
El modelo estadístico de spaCy reconoce entidades genéricas, pero no conoce los formatos propios del registro judicial mendocino.
Por eso se le agrega una capa de **reglas explícitas**: patrones escritos por personas con conocimiento del dominio que le indican al sistema exactamente qué formato tienen los números de expediente, padrón, DNI y escrituras.
Estas reglas se definen en el archivo `ner_patterns.py` y pueden ampliarse sin tocar el modelo ni el resto del sistema.
**¿Por qué es útil agregar reglas?** Porque un número como `REG-2023-00012847-MZA` no es una palabra del diccionario: el modelo estadístico por sí solo no sabe que es un identificador registral. Con la regla correspondiente, el sistema lo reconoce de forma exacta y reproducible en todos los documentos que usen ese formato.

---

**¿Por qué es relevante esto en el ámbito jurídico registral?**

Los expedientes judiciales y registrales contienen datos críticos: nombres de partes, números de expediente, fechas, organismos intervinientes y resoluciones.
Un sistema que *alucine* o invente información en este contexto puede tener consecuencias graves.

Este sistema está diseñado con esa preocupación en mente:

- El **NER** extrae entidades directamente del texto, sin inferir ni completar datos faltantes.
- Las **reglas de dominio** garantizan que los identificadores registrales sean reconocidos con precisión, sin depender del modelo estadístico.
- **TF-IDF y BM25** seleccionan documentos por coincidencia matemática de términos: deterministas, auditables, sin interpretación.
- El **modelo generativo** corre localmente y recibe solo el fragmento relevante, con instrucciones estrictas de no agregar información externa.

De esta forma, la respuesta siempre puede ser **rastreada hasta su fuente**: si el dato no está en el documento, el sistema lo indica explícitamente en lugar de inventarlo.
    """)

nlp = load_nlp()

st.divider()

# ── Documentos ────────────────────────────────────────────────────
st.subheader("Documentos disponibles")
st.caption("Podés editar el contenido de cada documento o reemplazarlo con texto propio.")

num_docs = st.number_input("Cantidad de documentos", min_value=1, max_value=10, value=len(DEFAULT_DOCS), step=1)

docs = []
cols = st.columns(int(num_docs))
for i, col in enumerate(cols):
    with col:
        default_title, default_text = DEFAULT_DOCS[i] if i < len(DEFAULT_DOCS) else (f"Documento {i+1}", "")
        st.markdown(f"**{default_title}**")
        text = st.text_area(
            f"doc_{i}",
            value=default_text,
            height=200,
            label_visibility="collapsed",
            key=f"doc_text_{i}",
        )
        docs.append(text.strip())

# ── Pregunta ──────────────────────────────────────────────────────
st.divider()
st.subheader("Tu pregunta")

PREGUNTAS_SUGERIDAS = [
    "¿Quién compró el inmueble en calle San Martín 1542?",
    "¿Qué banco constituyó la hipoteca y por qué monto?",
    "¿Sobre quién recae la inhibición general de bienes?",
    "¿Sobre qué inmueble se ordenó el embargo y quién es su titular?",
    "¿Quiénes son los herederos declarados y qué inmueble integra el acervo hereditario?",
    "¿A favor de quién se constituyó el usufructo y quién conserva la nuda propiedad?",
    "¿Qué escribana intervino en la transferencia de dominio?",
    "¿Cuál es el número de padrón del inmueble hipotecado?",
    "¿Qué juzgado ordenó el embargo sobre Constructora Del Sur?",
    "¿En qué expediente tramitó la sucesión de Morales, Ana Beatriz?",
]

if "query_input" not in st.session_state:
    st.session_state["query_input"] = PREGUNTAS_SUGERIDAS[3]

with st.expander("Preguntas sugeridas — hacé clic para usar como base"):
    cols_q = st.columns(2)
    for idx, pregunta in enumerate(PREGUNTAS_SUGERIDAS):
        with cols_q[idx % 2]:
            if st.button(pregunta, key=f"sugerida_{idx}", width='stretch'):
                st.session_state["query_input"] = pregunta

query = st.text_input(
    "Pregunta (podés editarla libremente):",
    key="query_input",
)

consultar = st.button("Consultar", type="primary")

# ── Pipeline ──────────────────────────────────────────────────────
if consultar:
    if not query.strip():
        st.warning("Por favor escribí una pregunta antes de consultar.")
        st.stop()

    active_docs = [d for d in docs if d]
    if not active_docs:
        st.warning("Agregá al menos un documento.")
        st.stop()

    st.divider()
    st.subheader("Cómo llegó el sistema a la respuesta")
    st.info(
        "Este sistema usa arquitectura **RAG (Retrieval Augmented Generation)**: "
        "primero *recupera* el documento más relevante (Pasos 1 y 2) y luego *genera* "
        "la respuesta usando solo ese documento como contexto (Paso 3). "
        "Los Pasos 1 y 2 son PLN clásico: no generan texto, no inventan, resultado siempre reproducible.",
        icon="ℹ️",
    )

    # ── PASO 1: NER ───────────────────────────────────────────────
    st.markdown("### Paso 1 — Reconocimiento de entidades nombradas (NER) `RAG: enriquecimiento`")
    st.caption(
        "Técnica de PLN que analiza el texto y etiqueta automáticamente nombres de personas, "
        "organizaciones, lugares y fechas, más identificadores registrales definidos por reglas de dominio. "
        "**No crea ni modifica información, solo la identifica. No puede alucinar porque no genera nada.**"
    )

    with st.expander("Reglas de dominio activas — identificadores registrales (ner_patterns.py)"):
        st.caption(
            "Estas reglas extienden el NER estadístico con formatos específicos del registro inmobiliario. "
            "Son deterministas: si el texto cumple el patrón, se etiqueta siempre igual. "
            "Para agregar un nuevo formato, editá el archivo `src/ner_patterns.py`."
        )
        rows = [
            {
                "Etiqueta": info["nombre"],
                "Formato reconocido": info["formato"],
                "Ejemplos": info["ejemplos"],
            }
            for info in DESCRIPCION_PATRONES.values()
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    enriched_docs = []
    for i, doc_text in enumerate(active_docs):
        spacy_doc = nlp(doc_text)
        entities = [(ent.text, LABEL_ES.get(ent.label_, ent.label_)) for ent in spacy_doc.ents]

        suffix = "\n\nENTIDADES DETECTADAS:\n" + "".join(
            f"- {t} ({l})\n" for t, l in entities
        ) if entities else ""
        enriched_docs.append(doc_text + suffix)

        with st.expander(f"Documento {i+1}", expanded=True):
            if entities:
                df = pd.DataFrame(entities, columns=["Texto encontrado", "Tipo de dato"])
                st.dataframe(df, width='stretch', hide_index=True)
            else:
                st.info("No se detectaron entidades en este documento.")

    # ── PASO 2: Comparación TF-IDF vs fastembed ───────────────────
    st.markdown("### Paso 2 — Búsqueda por relevancia `RAG: retrieval`")
    st.caption(
        "Se calculan los scores con dos métodos no generativos y se comparan lado a lado. "
        "TF-IDF pondera frecuencia de términos; BM25 además normaliza por longitud del documento. "
        "Ambos producen números, nunca texto. Ninguno puede alucinar."
    )

    with st.spinner("Calculando similitudes..."):
        scores_tfidf = tfidf_scores(query, enriched_docs)
        scores_bm25  = bm25_scores(query, enriched_docs)

    best_tfidf = int(scores_tfidf.argmax())
    best_bm25  = int(scores_bm25.argmax())
    best_idx   = best_tfidf

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**TF-IDF**")
        st.caption("Pondera términos por frecuencia relativa. Matemática pura, sin ML.")
        for i, score in enumerate(scores_tfidf):
            pct = max(0.0, float(score))
            sufijo = " ✓ seleccionado" if i == best_tfidf else ""
            st.markdown(f"Doc {i+1}{sufijo} — `{pct*100:.1f}%`")
            st.progress(pct)

    with col_b:
        st.markdown("**BM25** *(Okapi BM25)*")
        st.caption("Agrega normalización por longitud del documento. Usado en Elasticsearch. Sin ML.")
        for i, score in enumerate(scores_bm25):
            pct = max(0.0, float(score))
            sufijo = " ✓" if i == best_bm25 else ""
            st.markdown(f"Doc {i+1}{sufijo} — `{pct*100:.1f}%`")
            st.progress(pct)

    if best_tfidf != best_bm25:
        st.warning(
            f"Los métodos difieren: TF-IDF eligió Doc {best_tfidf+1}, "
            f"BM25 eligió Doc {best_bm25+1}. "
            f"Se usa TF-IDF para la respuesta."
        )
    else:
        st.success(f"Ambos métodos coinciden: **Documento {best_idx + 1}** es el más relevante.")

    # ── PASO 3: LLM ───────────────────────────────────────────────
    st.markdown(f"### Paso 3 — Formulación de la respuesta ({OLLAMA_MODEL}) `RAG: generation`")
    st.caption(
        "Recién aquí interviene un modelo de lenguaje generativo. "
        "Recibe **únicamente** el documento seleccionado en el Paso 2 y la pregunta, "
        "con una instrucción explícita de responder solo con lo que está en el texto "
        "e indicar 'No se encontró información suficiente' si la respuesta no está. "
        "El riesgo de alucinación se reduce al mínimo porque el contexto ya fue filtrado "
        "por las etapas anteriores."
    )

    context = enriched_docs[best_idx]
    prompt = f"""Respondé en una oración completa y clara, utilizando únicamente la información
contenida en el contexto. No agregues interpretaciones ni información externa.
Si la respuesta no está explícitamente en el texto, indicá: "No se encontró información suficiente".

CONTEXTO:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""

    with st.expander("Ver el texto exacto que recibió el modelo"):
        st.code(prompt, language=None)

    with st.spinner("El modelo está elaborando la respuesta..."):
        try:
            answer = ask_llama(prompt)
        except Exception as e:
            st.error(f"No se pudo conectar con Ollama: {e}")
            st.info("Verificá que Ollama esté ejecutándose. Abrí una terminal y ejecutá: `ollama serve`")
            st.stop()

    st.markdown("#### Respuesta:")
    st.success(answer)

    st.divider()
    st.caption(
        f"Resumen del proceso: spaCy NER → TF-IDF + BM25 (búsqueda, sin ML) → Ollama/{OLLAMA_MODEL} (generación)"
    )
