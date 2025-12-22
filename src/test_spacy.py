import spacy
from spacy.matcher import Matcher

# Cargar modelo spaCy en español
nlp = spacy.load("es_core_news_md")

# Texto judicial de ejemplo (realista)
texto = """
En la ciudad de Mendoza, a los 15 días del mes de marzo de 2022,
el Juzgado Civil y Comercial N° 3 resolvió en el expediente
EXP-2021-00458723-GDEMSA, caratulado "Pérez, Juan c/ Dirección
de Registros Públicos s/ Acción Declarativa".

Intervino como magistrada la Dra. María González.
La resolución fue archivada en el Archivo Judicial Provincial.
"""

doc = nlp(texto)

print("\n--- ENTIDADES DETECTADAS (spaCy) ---\n")
for ent in doc.ents:
    print(f"{ent.text:<40} {ent.label_}")

# Matcher para patrones jurídicos específicos (ej: expediente)
matcher = Matcher(nlp.vocab)

patron_expediente = [
    {"TEXT": {"REGEX": "EXP[-–]\\d{4}[-–]\\d{8}[-–]?[A-Z]*"}}
]

matcher.add("NUMERO_EXPEDIENTE", [patron_expediente])

matches = matcher(doc)

print("\n--- ENTIDADES JURÍDICAS PERSONALIZADAS ---\n")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"{span.text:<40} NUM_EXPEDIENTE")

