# NER + RAG — Registro Inmobiliario de Mendoza
**Dirección de Registros Públicos y Archivo Judicial**

---

## Captura del sistema

![Consultor de Expedientes Judiciales](Consultor%20de%20Expedientes%20Judiciales.png)

---

## Objetivo

Proof of Concept que demuestra cómo aplicar técnicas de **Procesamiento del Lenguaje Natural (PLN)** y **RAG (Retrieval Augmented Generation)** para consultar documentos del registro inmobiliario en lenguaje natural.

El sistema responde preguntas usando **exclusivamente** la información contenida en los documentos. No inventa datos, no consulta internet y procesa todo localmente dentro de la red interna.

---

## Arquitectura

```
Documentos → NER (spaCy) → Búsqueda TF-IDF + BM25 → LLM local (Ollama)
              ↑ no generativo ↑    ↑ no generativo ↑    ↑ generativo ↑
```

| Componente | Tecnología | Generativo |
|------------|-----------|-----------|
| Reconocimiento de entidades (NER) | spaCy `es_core_news_md` + EntityRuler | No |
| Reglas de dominio registral | `ner_patterns.py` (patrones explícitos) | No |
| Búsqueda por relevancia | TF-IDF + BM25 (scikit-learn / rank-bm25) | No |
| Respuesta en lenguaje natural | Ollama `llama3.2:3b` (local) | Sí |
| Interfaz web | Streamlit | — |

---

## Requisitos

- Python 3.11
- [Ollama](https://ollama.com) instalado y corriendo
- Modelo descargado: `ollama pull llama3.2:3b`
- Red con proxy corporativo (ver configuración más abajo)

---

## Instalación

### 1. Crear entorno virtual

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la ejecución:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt --proxy http://<PROXY_HOST>:<PUERTO> --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

### 3. Instalar modelo de español para spaCy

```powershell
pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_md-3.7.0/es_core_news_md-3.7.0-py3-none-any.whl --proxy http://<PROXY_HOST>:<PUERTO> --trusted-host github.com --trusted-host objects.githubusercontent.com --trusted-host releases.githubusercontent.com
```

### 4. Descargar modelo LLM

```powershell
ollama pull llama3.2:3b
```

---

## Uso

### Activar entorno y ver opciones

```powershell
.\activar_entorno.ps1
```

El script activa el venv, configura las variables de proxy y muestra los comandos disponibles.

### Interfaz web (recomendado)

```powershell
streamlit run src/app.py
```

Abre automáticamente en `http://localhost:8501`.

### Uso en red local (otras PCs)

```powershell
streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501
```

Requiere que el puerto 8501 esté habilitado en el firewall de Windows.

### Pipeline por consola

```powershell
cd src
python pipeline_rag_ner_judicial.py
```

---

## Variables de entorno requeridas

| Variable | Valor | Por qué |
|----------|-------|---------|
| `HTTPS_PROXY` | `http://<PROXY_HOST>:<PUERTO>` | Proxy corporativo |
| `HTTP_PROXY` | `http://<PROXY_HOST>:<PUERTO>` | Proxy corporativo |
| `NO_PROXY` | `localhost,127.0.0.1` | Evita que el proxy intercepte Ollama |
| `HF_HUB_DISABLE_SSL_VERIFICATION` | `1` | Certificado autofirmado del proxy |

El script `activar_entorno.ps1` las configura automáticamente.

---

## Estructura del proyecto

```
ner-rag-poc-main/
├── activar_entorno.ps1        # Script de arranque
├── requirements.txt           # Dependencias Python
├── CHANGELOG.md               # Historial de versiones y decisiones técnicas
├── src/
│   ├── app.py                 # Interfaz web Streamlit
│   ├── ner_patterns.py        # Reglas de dominio registral (EntityRuler)
│   ├── llm_ollama.py          # Cliente Ollama (cambiar modelo aquí)
│   ├── embeddings.py          # Embeddings originales (referencia)
│   └── pipeline_rag_ner_judicial.py  # Pipeline por consola
└── data/
    └── sample.txt
```

---

## Agregar nuevos formatos de expediente

Editá `src/ner_patterns.py` y agregá una entrada a `PATRONES_REGISTRALES`:

```python
{
    "label": "NRO_EXPEDIENTE",
    "pattern": [{"TEXT": {"REGEX": r"^TU-PREFIJO-\d{4}-\d{4,8}$"}}]
},
```

Y su descripción en `DESCRIPCION_PATRONES` para que aparezca en la interfaz.

---

## Cambiar modelo LLM

Editá la línea `MODEL` en `src/llm_ollama.py`:

```python
MODEL = "llama3.2:3b"   # rápido, ~2 GB RAM
# MODEL = "llama3.1:8b" # mayor calidad, ~5.5 GB RAM
# MODEL = "phi3:mini"   # muy rápido, respuestas cortas
```

---

## Ejemplo de consulta

**Pregunta:** ¿Sobre qué inmueble se ordenó el embargo y quién es su titular?

**Respuesta:** Se ordenó embargo sobre el inmueble Padrón N° 71.008, ubicado en Godoy Cruz, Mendoza, de titularidad de Constructora Del Sur S.R.L.

✔ Basada exclusivamente en el documento  
✔ Sin inferencias ni interpretación jurídica  
✔ Procesamiento 100% local — los datos no salen de la red interna  

---

## Principios de diseño

- El sistema **no toma decisiones legales**
- **No modifica** documentos originales
- Responde solo si la información está **explícitamente presente**
- Todo el procesamiento es **local** (Ollama corre en la misma máquina)
- Los pasos de búsqueda son **auditables y reproducibles**

---

## Autor

**Ing. Leonardo Villegas**  
GitHub: [https://github.com/Leonardorf](https://github.com/Leonardorf)  
Dirección de Registros Públicos y Archivo Judicial — Mendoza

---

> Este proyecto es un PoC técnico orientado a demostración. Ver `CHANGELOG.md` para el detalle de decisiones técnicas y limitaciones conocidas.
