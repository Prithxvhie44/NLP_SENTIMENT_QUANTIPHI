"""
POS Tagging and Named Entity Recognition
----------------------------------------
Uses spaCy to extract Part-of-Speech tags and Named Entities
from the cleaned reviews.
Saves structured output in /data/processed/pos_ner_reviews.json
"""

import os
import json
import spacy
from tqdm import tqdm

# Load the small English spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_pos_ner(text):
    """Extract POS tags (NOUN, ADJ, VERB) and Named Entities from text"""
    doc = nlp(text)
    pos_tags = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"]]
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, named_entities

def analyze_reviews(input_json, output_json):
    print(f"🔍 Reading cleaned reviews from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="🧠 Analyzing POS & Entities"):
        text = item.get("cleaned_text", "")
        pos_tags, named_entities = extract_pos_ner(text)

        results.append({
            "id": item.get("ID", ""),
            "user": item.get("Username", ""),
            "rating": item.get("Rating", ""),
            "text": text,
            "pos_tags": pos_tags,
            "named_entities": named_entities
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved POS & NER data → {output_json}")
    print(f"📊 Total reviews processed: {len(results)}")
    return results


if __name__ == "__main__":
    input_path = "data/processed/cleaned_reviews.json"
    output_path = "data/processed/pos_ner_reviews.json"
    analyze_reviews(input_path, output_path)
