# ============================================
# PHASE 2 — Syntactic & Semantic Analysis
# Quantiphi NLP Sentiment Project
# ============================================

import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm
tqdm.pandas()

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
df = pd.read_csv("data\processed\processed_reviews.csv")
df['cleaned_text'] = df['cleaned_text'].astype(str)
print(f"✅ Loaded {len(df)} reviews")

# ---------------------------
# 2️⃣ POS Tagging & NER
# ---------------------------
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm", disable=["parser"])

def extract_pos_ner(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities

df[['pos_tags', 'entities']] = df['cleaned_text'].progress_apply(
    lambda x: pd.Series(extract_pos_ner(x))
)
print("🧩 POS Tagging & NER complete")

# POS frequency distribution
all_pos = [tag for tags in df['pos_tags'] for tag in tags]
pos_counts = Counter(all_pos).most_common(10)
print("📊 Top POS tags:", pos_counts)

# ---------------------------
# 3️⃣ TF-IDF Vectorization
# ---------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['cleaned_text'])
print("✅ TF-IDF Matrix shape:", X_tfidf.shape)

# Top 10 TF-IDF terms
tfidf_means = np.array(X_tfidf.mean(axis=0)).ravel()
top_idx = tfidf_means.argsort()[::-1][:10]
top_terms = [(tfidf.get_feature_names_out()[i], tfidf_means[i]) for i in top_idx]
print("🔝 Top 10 keywords by TF-IDF:", top_terms)

# ---------------------------
# 4️⃣ Word2Vec Embeddings
# ---------------------------
sentences = [t.split() for t in df['cleaned_text']]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
print("✅ Word2Vec model trained with vocab size:", len(w2v.wv))

# Similar words example
for feature in ["battery", "camera", "screen"]:
    if feature in w2v.wv:
        print(f"💬 Similar to '{feature}':", w2v.wv.most_similar(feature, topn=5))

# ---------------------------
# 5️⃣ Lexicon-based Sentiment (VADER)
# ---------------------------
analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    s = analyzer.polarity_scores(text)
    return s['compound']

df['sentiment_score'] = df['cleaned_text'].progress_apply(vader_score)
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
)

print("🧠 Sentiment distribution:")
print(df['sentiment_label'].value_counts())

# ---------------------------
# 6️⃣ Topic Modeling (LSA)
# ---------------------------
lsa = TruncatedSVD(n_components=5, random_state=42)
lsa_matrix = lsa.fit_transform(X_tfidf)

terms = tfidf.get_feature_names_out()
topics = []
for i, comp in enumerate(lsa.components_):
    terms_idx = np.argsort(comp)[::-1][:10]
    topics.append([terms[t] for t in terms_idx])
    print(f"\n🗂️ Topic {i+1}: {', '.join(terms[t] for t in terms_idx)}")

# ---------------------------
# 7️⃣ Semantic Similarity (Example)
# ---------------------------
def top_similar_reviews(keyword, topn=5):
    vec = tfidf.transform([keyword])
    sims = cosine_similarity(vec, X_tfidf).ravel()
    top_idx = sims.argsort()[::-1][:topn]
    return df.iloc[top_idx][['ID','cleaned_text','sentiment_label']]

print("\n🔍 Top similar reviews for 'battery life':")
print(top_similar_reviews("battery life"))

# ---------------------------
# 8️⃣ Save Outputs
# ---------------------------
df.to_csv("phase2_output.csv", index=False)
print("💾 Saved results → phase2_output.csv")
