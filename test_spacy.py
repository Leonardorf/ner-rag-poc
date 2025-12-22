import spacy

nlp = spacy.load("es_core_news_md")

text = "Hugo Villegas trabajó en YPF en 2024"
doc = nlp(text)

print([(e.text, e.label_) for e in doc.ents])
