# src/prompts.py
def ner_prompt(context: str) -> str:
    return f"""
Eres un sistema de extracción de entidades nombradas en español.

Extrae y devuelve SOLO un JSON con este esquema:

{{
  "PERSONA": [],
  "ORGANIZACION": [],
  "FECHA": [],
  "LUGAR": [],
  "OTROS": []
}}

Texto:
\"\"\"
{context}
\"\"\"

No expliques nada.
"""
