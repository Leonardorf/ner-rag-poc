# NER + RAG Judicial PoC
Dirección de Registros Públicos y Archivo Judicial

------------------------------------------------------------------------

##  Objetivo del Proyecto

Este Proof of Concept (PoC) demuestra cómo aplicar Reconocimiento de Entidades Nombradas (NER) y RAG (Retrieval Augmented Generation) para facilitar la consulta inteligente de documentos judiciales archivados.

El sistema permite responder preguntas en lenguaje natural utilizando exclusivamente información existente en los expedientes, garantizando trazabilidad, control y ausencia de interpretaciones automáticas.

------------------------------------------------------------------------

## Pipeline principal (PASO 2)

Archivo clave Ejecución:

``` bash
python src/pipeline_rag_ner_judicial.py

```
Otro archivo de ejecucion para probar llama
``` bash
python src/pipeline_rag_llama.py 
```            
------------------------------------------------------------------------

##  Estructura del proyecto

![Estructura del proyecto](estructura.png)

Archivos principales: - `test_spacy.py`: prueba básica de NER con
spaCy - `src/`: código del pipeline RAG + LLaMA - `data/`: textos de
ejemplo

------------------------------------------------------------------------

##  Requisitos

-   Python 3.10
-   Conda
-   VS Code
-   Ollama

------------------------------------------------------------------------

##  Flujo funcional

1 Ingreso de textos judiciales

2Extracción de entidades con spaCy ES

    Personas

    Organismos

    Fechas

    Números de expediente

3 Enriquecimiento semántico de los documentos

4 Generación de embeddings (RoBERTa)

5 Búsqueda semántica (RAG)

6 Respuesta controlada con LLaMA (Ollama)


------------------------------------------------------------------------



##  Ejemplo de uso institucional
### Texto procesado

“En el expediente EXP-2021-00458723-GDEMZA, el Juzgado Civil y Comercial N° 3 resolvió la causa iniciada por Pérez, Juan contra la Dirección de Registros Públicos.”

Consulta en lenguaje natural

¿En qué expediente intervino la Dirección de Registros Públicos?

Respuesta generada

La Dirección de Registros Públicos intervino en el expediente EXP-2021-00458723-GDEMZA.

✔️ Respuesta breve
✔️ Basada exclusivamente en el archivo
✔️ Sin inferencias ni interpretación jurídica

 Principios clave para uso judicial

El sistema no toma decisiones legales

No modifica documentos originales

Responde solo si la información está presente

Procesamiento local

Auditoría y trazabilidad completas

------------------------------------------------------------------------


##  Autor

**Ing. Leonardo Villegas**\
GitHub: https://github.com/Leonardorf

------------------------------------------------------------------------

> Nota: este proyecto es un PoC técnico, no optimizado para
> producción.
> Nota Borrador para Windows  usando Vs code
 Habilitar conda en PowerShell.

1️⃣ Abrí Anaconda Prompt (una sola vez/Windows)

Ejecutá:

conda init powershell

2️⃣ Cerrá TODO

Cerrá VS Code

Cerrá PowerShell

3️⃣ Abrí VS Code de nuevo

Nueva terminal:

conda activate ner-rag
Listo para ejecutar los pipelines