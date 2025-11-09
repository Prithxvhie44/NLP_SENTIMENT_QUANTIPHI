"""
Summarization and Insight Generation Module
-------------------------------------------
Clusters similar reviews using TF-IDF + cosine similarity
and extracts representative review summaries for each sentiment.
Outputs summary JSON report.
"""

import os, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def summarize_reviews(input_json, output_json, n_sentences=3):
    print(f"🔍 Reading sentiment data from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentiments = {"Positive": [], "Neutral": [], "Negative": []}
    for r in data:
        label = r.get("label", "Neutral")
        sentiments[label].append(r.get("text", ""))

    summaries = {}
    for label, reviews in sentiments.items():
        if not reviews:
            summaries[label] = []
            continue

        print(f"🧠 Summarizing {label} reviews ({len(reviews)}) ...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vectorizer.fit_transform(reviews)

        sim = cosine_similarity(X)
        centrality = np.sum(sim, axis=1)
        top_idx = np.argsort(-centrality)[:n_sentences]

        summaries[label] = [reviews[i] for i in top_idx]

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"✅ Summaries saved → {output_json}")
    return summaries


if __name__ == "__main__":
    input_path = "data/processed/sentiment_reviews.json"
    output_path = "data/reports/summary.json"
    summarize_reviews(input_path, output_path)
