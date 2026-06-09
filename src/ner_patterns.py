# ============================================================
# ner_patterns.py — Reglas de reconocimiento de entidades
#                   específicas del Registro Inmobiliario
# ============================================================
#
# ¿Por qué este archivo?
# El modelo estadístico de spaCy (es_core_news_md) reconoce
# entidades genéricas (personas, lugares, organizaciones).
# Pero NO conoce los formatos propios del registro judicial:
# números de expediente, padrón, DNI, escrituras, etc.
#
# Estas reglas le "enseñan" al sistema esos formatos de forma
# explícita y determinista: si el texto cumple el patrón,
# se etiqueta. Sin ML. Sin ambigüedad. Sin alucinación.
#
# ¿Cómo agregar un nuevo patrón?
# Copiá uno de los bloques de abajo y modificá:
#   - "label": nombre de la categoría (en mayúsculas)
#   - "pattern": lista de tokens que forman la entidad
#   - "REGEX": expresión regular que debe cumplir cada token
#
# Después agregá la etiqueta al dict DESCRIPCION_PATRONES
# y a LABEL_ES en app.py para que aparezca en la tabla.
# ============================================================

PATRONES_REGISTRALES = [

    # ── Números de expediente / registrales ──────────────────
    # Formato: PREFIJO-AAAA-NNNNNNN(-SUFIJO)?
    # Ejemplos: EXP-2022-00985-JC7  REG-2023-00012847-MZA
    #           HIP-2023-00441      USF-2023-00089-MZA
    {
        "label": "NRO_EXPEDIENTE",
        "pattern": [
            {
                "TEXT": {
                    "REGEX": r"^(EXP|REG|HIP|USF|SUC|EMB|JCC)-\d{4}-\d{4,8}(-[A-Z0-9]{2,6})?$"
                }
            }
        ],
    },

    # ── Número de padrón inmobiliario ─────────────────────────
    # Formato: Padrón N° NNNNN o Padron N° NNNNN
    # Ejemplos: Padrón N° 45.231    Padrón N° 71.008
    {
        "label": "NRO_PADRON",
        "pattern": [
            {"LOWER": {"IN": ["padrón", "padron"]}},
            {"TEXT": "N°"},
            {"TEXT": {"REGEX": r"^\d[\d\.]*$"}},
        ],
    },

    # ── Número de DNI ────────────────────────────────────────
    # Formato: DNI NNNNNNNN (con o sin puntos)
    # Ejemplos: DNI 28.441.902    DNI 12334567
    {
        "label": "NRO_DNI",
        "pattern": [
            {"TEXT": "DNI"},
            {"TEXT": {"REGEX": r"^\d[\d\.]*$"}},
        ],
    },

    # ── Número de escritura ───────────────────────────────────
    # Formato: escritura N° NNN
    # Ejemplo: escritura N° 88    escritura N° 142
    {
        "label": "NRO_ESCRITURA",
        "pattern": [
            {"LOWER": "escritura"},
            {"TEXT": "N°"},
            {"TEXT": {"REGEX": r"^\d+$"}},
        ],
    },
]

# ── Descripción legible de cada etiqueta ─────────────────────
# Se usa para mostrar en la interfaz qué reconoce cada regla.
DESCRIPCION_PATRONES = {
    "NRO_EXPEDIENTE": {
        "nombre":   "N° de Expediente / Registro",
        "formato":  "EXP-AAAA-NNNNNNN-XXX, REG-..., HIP-..., USF-..., SUC-...",
        "ejemplos": "EXP-2022-00985-JC7 · REG-2023-00012847-MZA · HIP-2023-00441",
    },
    "NRO_PADRON": {
        "nombre":   "N° de Padrón Inmobiliario",
        "formato":  "Padrón N° NNNNN",
        "ejemplos": "Padrón N° 45.231 · Padrón N° 71.008",
    },
    "NRO_DNI": {
        "nombre":   "Documento Nacional de Identidad",
        "formato":  "DNI NNNNNNNN",
        "ejemplos": "DNI 28.441.902 · DNI 12.334.567",
    },
    "NRO_ESCRITURA": {
        "nombre":   "N° de Escritura Pública",
        "formato":  "escritura N° NNN",
        "ejemplos": "escritura N° 88 · escritura N° 142",
    },
}
