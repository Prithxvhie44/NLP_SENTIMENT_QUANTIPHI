"""
Topic Modeling Module
---------------------
Performs topic extraction from cleaned reviews using TF-IDF + LSA (Latent Semantic Analysis).
Outputs discovered topics and representative keywords.
"""

import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

def extract_topics(input_json, output_json, n_topics=5, n_words=10):
    """
    Extracts topics from the cleaned reviews using TF-IDF + LSA
    Args:
        input_json: cleaned reviews JSON file path
        output_json: output file to save topics
        n_topics: number of topics to extract
        n_words: number of words per topic
    """
    print(f"🔍 Reading cleaned reviews from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Combine all reviews
    texts = [d.get("cleaned_text", "") for d in data if d.get("cleaned_text")]

    print("⚙️ Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_df=0.8, 
        min_df=2, 
        stop_words='english',
        max_features=5000
    )
    X = vectorizer.fit_transform(texts)

    print("🧠 Performing LSA Topic Modeling...")
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    svd.fit(X)

    terms = vectorizer.get_feature_names_out()
    topics = {}

    for i, comp in enumerate(svd.components_):
        terms_in_topic = [terms[idx] for idx in comp.argsort()[:-n_words - 1:-1]]
        topics[f"Topic_{i+1}"] = terms_in_topic

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)

    print(f"✅ Extracted {len(topics)} topics → {output_json}")
    for t, words in topics.items():
        print(f"{t}: {', '.join(words)}")
    return topics


if __name__ == "__main__":
    input_path = "data/processed/cleaned_reviews.json"
    output_path = "data/processed/topics.json"
    extract_topics(input_path, output_path, n_topics=5, n_words=10)
