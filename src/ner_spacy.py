# src/ner_spacy.py
import spacy
from collections import defaultdict

nlp = spacy.load("es_core_news_md")


def extract_entities(text: str) -> dict:
    doc = nlp(text)
    entities = defaultdict(set)

    for ent in doc.ents:
        entities[ent.label_].add(ent.text)

    return {label: list(values) for label, values in entities.items()}
