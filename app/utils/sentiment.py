"""
Sentiment Analysis Module
-------------------------
Performs sentiment scoring on cleaned reviews using NLTK's VADER.
Outputs labeled data in /data/processed/sentiment_reviews.json
"""

import os
import json
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon (only first time)
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(text):
    """Return sentiment polarity (pos, neu, neg, compound)"""
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def sentiment_pipeline(input_json, output_json):
    """Apply sentiment analysis to all reviews"""
    print(f"🔍 Reading cleaned reviews from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sia = SentimentIntensityAnalyzer()
    results = []

    for review in tqdm(data, desc="💬 Analyzing Sentiment"):
        text = review.get("cleaned_text", "")
        rating = review.get("Rating", "")
        user = review.get("Username", "")
        sentiment_scores = sia.polarity_scores(text)

        compound = sentiment_scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "id": review.get("ID", ""),
            "user": user,
            "rating": rating,
            "text": text,
            "compound": round(compound, 3),
            "label": label
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Sentiment analysis completed → {output_json}")
    print(f"📊 Total reviews analyzed: {len(results)}")
    return results


if __name__ == "__main__":
    input_path = "data/processed/cleaned_reviews.json"
    output_path = "data/processed/sentiment_reviews.json"
    sentiment_pipeline(input_path, output_path)
