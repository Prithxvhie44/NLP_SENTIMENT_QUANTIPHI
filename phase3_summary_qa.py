# ============================================
# PHASE 3 — Review Summarization & QA Synthesis
# Quantiphi NLP Sentiment Project
# ============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import json

# ---------------------------
# 1️⃣ Load Phase 2 Output
# ---------------------------
df = pd.read_csv("phase2_output.csv")
text_col = "cleaned_text" if "cleaned_text" in df.columns else "processed_text"

print(f"✅ Loaded {len(df)} reviews from phase2_output.csv")

# ---------------------------
# 2️⃣ TF-IDF Vectorization
# ---------------------------
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X_tfidf = tfidf.fit_transform(df[text_col])
print("✅ TF-IDF vectorization complete.")

# ---------------------------
# 3️⃣ Compute Similarity Matrix
# ---------------------------
sim_matrix = cosine_similarity(X_tfidf)
print("📈 Cosine similarity matrix computed.")

# ---------------------------
# 4️⃣ Cluster Reviews (K-Means)
# ---------------------------
k = 5 if len(df) > 200 else 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_tfidf)

df["cluster"] = clusters
print(f"🧩 Created {k} review clusters.")

# ---------------------------
# 5️⃣ Representative Reviews per Cluster
# ---------------------------
summaries = []
for i in range(k):
    cluster_indices = df[df["cluster"] == i].index
    if len(cluster_indices) == 0:
        continue
    cluster_vecs = X_tfidf[cluster_indices]
    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
    sims = cosine_similarity(cluster_vecs, centroid).ravel()
    best_idx = cluster_indices[sims.argmax()]
    summaries.append({
        "cluster_id": i,
        "representative_review": df.loc[best_idx, text_col],
        "sentiment": df.loc[best_idx, "sentiment_label"]
            if "sentiment_label" in df.columns else "N/A"
    })

# ---------------------------
# 6️⃣ Generate Overall Summary
# ---------------------------
sentiment_counts = df["sentiment_label"].value_counts().to_dict() \
    if "sentiment_label" in df.columns else {}

summary_text = {
    "total_reviews": len(df),
    "sentiment_distribution": sentiment_counts,
    "clusters": summaries
}

print("\n📝 Review Summary:")
for s in summaries:
    print(f"Cluster {s['cluster_id']} ({s['sentiment']}): {s['representative_review'][:120]}...")

# ---------------------------
# 7️⃣ Generate Q&A Synthesis
# ---------------------------
# ---------------------------
# 7️⃣ Generate Q&A Synthesis (Updated Topics)
# ---------------------------

questions = []
topics = ["price", "grip", "weight balance", "rubber quality", "delivery", "packaging"]

for t in random.sample(topics, min(5, len(topics))):
    related_reviews = df[df[text_col].str.contains(t.split()[0], case=False, na=False)]
    if not related_reviews.empty:
        pos = len(related_reviews[related_reviews["sentiment_label"] == "Positive"])
        neg = len(related_reviews[related_reviews["sentiment_label"] == "Negative"])
        total = len(related_reviews)
        if total == 0:
            continue

        # compute positivity %
        sentiment_ratio = round((pos / total) * 100, 1)
        if sentiment_ratio >= 75:
            summary = f"Most users ({sentiment_ratio}%) mention positive feedback about {t}."
        elif 50 <= sentiment_ratio < 75:
            summary = f"About {sentiment_ratio}% of users are satisfied with {t}, though some mention issues."
        else:
            summary = f"Only {sentiment_ratio}% of users express positive sentiment about {t}, showing mixed reviews."

        question = f"Q: How do customers feel about {t}?"
        questions.append({"question": question, "answer": summary})

summary_text["qa_pairs"] = questions


# ---------------------------
# 8️⃣ Save Outputs
# ---------------------------
with open("phase3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_text, f, indent=2, ensure_ascii=False)

pd.DataFrame(questions).to_csv("phase3_qa.csv", index=False)
print("\n💾 Saved summary → phase3_summary.json")
print("💾 Saved Q&A     → phase3_qa.csv")

print("\n✅ Phase 3 Completed Successfully!")
