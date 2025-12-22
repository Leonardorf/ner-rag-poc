# src/pipeline.py
from ingest import load_text, chunk_text
from ner_spacy import extract_entities

if __name__ == "__main__":
    text = load_text("data/sample.txt")

    print("\n--- TEXTO ---")
    print(text)

    print("\n--- ENTIDADES (spaCy) ---")
    entities = extract_entities(text)
    for label, values in entities.items():
        print(f"{label}: {values}")

    chunks = chunk_text(text)
    print(f"\n--- CHUNKS ({len(chunks)}) ---")
    for i, c in enumerate(chunks):
        print(f"[{i}] {c}")
