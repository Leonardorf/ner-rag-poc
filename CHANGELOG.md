# Registro de cambios y decisiones técnicas
## Sistema NER + RAG — Documentos Jurídico-Registrales

---

## v0.3 — Interfaz web y comparación de métodos de búsqueda

### Qué se agregó
- Interfaz web con Streamlit: documentos editables, preguntas sugeridas, resultados paso a paso
- Comparación lado a lado de TF-IDF vs BM25 en el Paso 2
- EntityRuler con reglas de dominio registral (`ner_patterns.py`)
- Glosario integrado: PLN, NER, TF-IDF, BM25, RAG y su importancia jurídica
- 6 documentos de ejemplo del registro inmobiliario
- 10 preguntas sugeridas clickeables con campo libre editable
- El nombre del modelo LLM se muestra dinámicamente desde `llm_ollama.py`

### Por qué BM25 además de TF-IDF

**TF-IDF** (Term Frequency–Inverse Document Frequency) asigna un peso a cada término
según dos factores:
- Qué tan frecuente es en el documento (`TF`)
- Qué tan raro es en el conjunto de documentos (`IDF`)

**BM25** (Best Match 25 / Okapi BM25) extiende TF-IDF con dos mejoras clave:

1. **Saturación de frecuencia**: en TF-IDF, si un término aparece 10 veces en un documento
   vale el doble que si aparece 5. En BM25, la relevancia crece más lentamente y se estabiliza.
   Esto evita que documentos muy largos con muchas repeticiones "engañen" al sistema.

2. **Normalización por longitud**: un documento corto que menciona el término buscado
   una sola vez puede ser más relevante que uno largo que lo menciona muchas veces
   porque en el texto largo puede ser una mención marginal. BM25 ajusta el score
   según el largo promedio de los documentos.

**Ejemplo práctico en contexto registral:**
Si buscamos "embargo Padrón N° 71.008" y un documento de 3 líneas lo menciona una vez,
BM25 lo considera más relevante que un documento de 20 líneas que lo menciona dos veces
pero en un contexto más general. TF-IDF podría elegir el segundo.

**Cuándo difieren TF-IDF y BM25:**
Con documentos de longitud similar (como los ejemplos de esta demo), los resultados
suelen coincidir. La diferencia se hace visible cuando los documentos tienen tamaños
muy distintos, que es el caso real de un repositorio de expedientes.

**Ambos son completamente no generativos**: producen un número (score de relevancia),
no texto. El resultado es determinista y reproducible para los mismos documentos y pregunta.
Son auditables: se puede rastrear exactamente por qué un documento obtuvo cierto puntaje.

---

## v0.2 — Cambio de motor de embeddings

### Qué cambió
- Se reemplazó `SentenceTransformers` + `all-roberta-large-v1` por TF-IDF (scikit-learn)
- Se eliminó PyTorch del entorno

### Por qué se eliminó PyTorch

PyTorch 2.x requiere el paquete redistribuible de Visual C++ 2022 para cargar sus DLLs.
En el entorno de trabajo (Windows 11, sin permisos de administrador) no fue posible instalarlo.
El error concreto fue:

```
OSError: [WinError 1114] Error en una rutina de inicialización de biblioteca de vínculos dinámicos (DLL).
Error loading "torch\lib\c10.dll"
```

Lo mismo ocurrió con `fastembed` (que usa `onnxruntime` internamente):

```
ImportError: DLL load failed while importing onnxruntime_pybind11_state
```

Ambas bibliotecas requieren VC++ 2022 que no se puede instalar sin admin.

### Por qué TF-IDF es suficiente (y mejor para el objetivo didáctico)

`SentenceTransformers` usa una red neuronal transformer entrenada con ML.
Si bien no es generativa, es una "caja gris": el resultado es difícil de explicar sin
conocimientos de deep learning.

TF-IDF y BM25 son fórmulas matemáticas que se pueden escribir en una pizarra.
Para el objetivo principal del proyecto —demostrar técnicas **no generativas** y
**auditables** en un contexto jurídico— son técnicamente superiores como ejemplo pedagógico.

La diferencia de calidad en la recuperación es mínima para documentos con terminología
estandarizada como los del registro inmobiliario, donde los mismos términos técnicos
aparecen en las preguntas y en los documentos.

---

## v0.1 — Pipeline inicial por consola

### Qué se construyó
- Pipeline NER + RAG ejecutable desde consola (`src/pipeline_rag_ner_judicial.py`)
- NER con spaCy `es_core_news_md`
- Embeddings con SentenceTransformers `all-roberta-large-v1` + FAISS
- LLM con Ollama `llama3.1:8b` local

### Entorno
- Python 3.11 (venv)
- Proxy corporativo: `<PROXY_HOST>:<PUERTO>` con certificado SSL autofirmado
- Variables requeridas: `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY=localhost`, `HF_HUB_DISABLE_SSL_VERIFICATION=1`

### Por qué Ollama en lugar de Claude API
Los documentos del registro inmobiliario contienen datos sensibles de personas físicas
y jurídicas. Usar una API externa implicaría enviar esos datos a servidores de terceros.
Ollama ejecuta el modelo localmente: los datos nunca salen de la red interna.

### Por qué llama3.2:3b en lugar de llama3.1:8b
Con 16 GB de RAM y la posibilidad de múltiples usuarios simultáneos en red,
`llama3.1:8b` (~5.5 GB de RAM) deja poco margen. `llama3.2:3b` (~2 GB de RAM)
es el sucesor directo, diseñado para respuestas cortas con instrucciones precisas.
Para RAG con contexto acotado la diferencia de calidad es mínima.
Para cambiar de modelo: editar `MODEL` en `src/llm_ollama.py`.

---

## Limitaciones conocidas

- **Concurrencia**: Ollama procesa una consulta a la vez. Con varios usuarios simultáneos,
  las consultas se encolan. Para producción considerar una cola de tareas.
- **Documentos en memoria**: Los documentos son texto fijo en la interfaz. No hay carga
  de archivos PDF ni conexión a base de datos. Siguiente paso natural.
- **Sin autenticación**: Cualquier persona en la red puede acceder. Para producción
  agregar autenticación básica de Streamlit.
- **BM25 sin stemming**: Los términos se comparan tal como aparecen. "embargado" y "embargo"
  son palabras distintas para el algoritmo. Agregar un stemmer en español mejoraría la cobertura.
