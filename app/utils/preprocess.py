"""
Preprocessing Module for Amazon Review Dataset
----------------------------------------------
Cleans raw text, merges review fields, removes rating artifacts like
'out of stars', punctuation, emojis, and other noise.
Performs tokenization and lemmatization.
Outputs processed data to /data/processed/cleaned_reviews.json
"""

import pandas as pd
import os
import re
import nltk
import ftfy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import json

# ✅ Download necessary NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

# ------------------------------------------------------------
# 🧹 Text Cleaning Function
# ------------------------------------------------------------
def clean_text(text):
    """Cleans review text by removing artifacts like 'out of stars', URLs, emojis, etc."""
    if not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    text = text.lower()

    # Remove Amazon rating artifacts
    text = re.sub(r"\b\d+(\.\d+)?\s*out\s*of\s*\d*\s*stars?\b", " ", text)
    text = re.sub(r"\bout\s*of\s*stars?\b", " ", text)

    # Remove URLs, HTML tags, punctuation, digits
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------
# 🧠 Lemmatization + Stopword Removal
# ------------------------------------------------------------
def preprocess_text(text):
    """Tokenizes, removes stopwords, lemmatizes words."""
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)

# ------------------------------------------------------------
# 🚀 Main Preprocessing Function
# ------------------------------------------------------------
def preprocess_reviews(input_csv, output_json):
    """
    Reads the review dataset, merges text fields, cleans text,
    lemmatizes it, and saves processed JSON.
    """
    print(f"🔍 Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # Merge both text fields if they exist
    text_fields = []
    for col in df.columns:
        if 'review' in col.lower() or 'content' in col.lower():
            text_fields.append(col)

    if not text_fields:
        raise ValueError("❌ Could not find review text columns in CSV.")

    print(f"🧩 Found review columns: {text_fields}")
    df["combined_review"] = df[text_fields].fillna("").astype(str).agg(" ".join, axis=1)

    tqdm.pandas(desc="🧼 Cleaning and Lemmatizing")
    df["cleaned_text"] = df["combined_review"].progress_apply(clean_text).progress_apply(preprocess_text)

    # Keep essential columns only
    df_final = df[["ID", "Username", "Rating", "cleaned_text"]].dropna()
    df_final = df_final[df_final["cleaned_text"].str.strip() != ""]

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    df_final.to_json(output_json, orient="records", force_ascii=False, indent=2)

    print(f"✅ Cleaned {len(df_final)} reviews → {output_json}")
    return df_final

# ------------------------------------------------------------
# 🧾 Run Standalone
# ------------------------------------------------------------
if __name__ == "__main__":
    input_path = "data/raw/review.csv"
    output_path = "data/processed/cleaned_reviews.json"
    preprocess_reviews(input_path, output_path)
