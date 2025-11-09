# ============================================
# Streamlit Q&A App — Customer Insights
# Quantiphi NLP Sentiment Project
# ============================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Customer Sentiment Q&A", page_icon="🧠", layout="wide")

# ---------------------------
# 1️⃣ Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("phase2_output.csv")
    if 'cleaned_text' not in df.columns:
        st.error("❌ 'cleaned_text' column not found! Please use Phase 2 output.")
        st.stop()
    df.dropna(subset=['cleaned_text'], inplace=True)
    return df

df = load_data()



# ---------------------------
# 2️⃣ Page Setup
# ---------------------------



st.title("🧠 Customer Review Q&A System")
st.markdown("""
Ask any question related to customer opinions — for example:  
- *"What do people say about price?"*  
- *"How is the grip quality?"*  
- *"Are customers happy with packaging?"*  
""")

# ---------------------------
# 3️⃣ TF-IDF Vectorization
# ---------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(df['cleaned_text'])

# ---------------------------
# 4️⃣ Q&A Section
# ---------------------------
user_query = st.text_input("🔍 Ask a Question:")

if user_query:
    query_vec = tfidf.transform([user_query])
    sims = cosine_similarity(query_vec, X_tfidf).ravel()
    top_idx = sims.argsort()[::-1][:5]

    st.subheader("📊 Most Relevant Reviews")
    for idx in top_idx:
        review = df.iloc[idx]['cleaned_text']
        sentiment = df.iloc[idx].get('sentiment_label', 'Unknown')
        st.markdown(f"**Sentiment:** {sentiment}")
        st.write(f"💬 {review}")
        st.write("---")

    # Summarize overall sentiment
    sentiments = df.iloc[top_idx]['sentiment_label'].value_counts().to_dict()
    if sentiments:
        most_common = max(sentiments, key=sentiments.get)
        st.success(f"✅ Overall, customers express **{most_common.lower()}** sentiment about this topic.")
    else:
        st.warning("Not enough sentiment data to summarize.")

# ---------------------------
# 5️⃣ Negative Feedback Highlights
# ---------------------------
# ---------------------------
# 📉 Negative Feedback Highlights (Enhanced)
# ---------------------------
import re
from collections import Counter

st.subheader("📉 Negative Feedback Highlights")

if 'sentiment_label' not in df.columns:
    st.warning("No sentiment labels found — please run Phase 2 first.")
else:
    neg_df = df[df['sentiment_label'] == 'Negative']

    if neg_df.empty:
        st.info("🎉 No negative feedback found — customers are mostly satisfied!")
    else:
        # Clean repetitive or short reviews
        neg_df['cleaned_text'] = neg_df['cleaned_text'].apply(
            lambda x: re.sub(r'\s+', ' ', str(x)).strip().lower()
        )
        neg_df.drop_duplicates(subset=['cleaned_text'], inplace=True)

        st.write(f"Found {len(neg_df)} unique negative reviews. Here are a few examples:")
        for i, row in enumerate(neg_df.head(5).itertuples(), 1):
            st.markdown(f"**{i}.** {row.cleaned_text.capitalize()}")
            st.write("---")

        # Quick keyword summary
        all_words = " ".join(neg_df['cleaned_text']).split()
        common_words = Counter([w for w in all_words if len(w) > 3]).most_common(5)
        keywords = ", ".join([w for w, _ in common_words])

        st.success(f"🔎 Common issues reported: **{keywords}**")


# ============================================
# 🧭 DASHBOARD INSIGHTS SECTION
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

st.markdown("## 📈 Overall Insights Dashboard")

if 'sentiment_label' in df.columns:

    # 1️⃣ --- Basic Metrics
    total_reviews = len(df)
    avg_rating = df['Rating'].mean() if 'Rating' in df.columns else np.nan
    pos_count = len(df[df['sentiment_label'] == 'Positive'])
    neg_count = len(df[df['sentiment_label'] == 'Negative'])
    neu_count = len(df[df['sentiment_label'] == 'Neutral'])

    col1, col2, col3 = st.columns(3)
    col1.metric("🗂️ Total Reviews", f"{total_reviews}")
    col2.metric("⭐ Average Rating", f"{avg_rating:.1f}" if not np.isnan(avg_rating) else "N/A")
    col3.metric("😊 Positive Sentiment", f"{(pos_count / total_reviews) * 100:.1f}%")

    # 2️⃣ --- Donut Chart for Sentiment Distribution
    # 2️⃣ --- Donut Chart for Sentiment Distribution (Resized)
    st.markdown("### 🥧 Sentiment Distribution")

    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#4CAF50', '#FFB300', '#E53935']  # green, yellow, red

    fig, ax = plt.subplots(figsize=(3.2, 3.2))  # 👈 smaller chart size

    wedges, texts, autotexts = ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=150,
        colors=colors[:len(sentiment_counts)],
        textprops={'fontsize': 10, 'color': 'white'}
    )

    # Smaller donut hole for compactness
    centre_circle = plt.Circle((0, 0), 0.68, fc='#0E1117')  # 👈 radius slightly larger (thicker ring)
    fig.gca().add_artist(centre_circle)

    ax.set_title("Sentiment Breakdown", fontsize=12, color='white', pad=10)
    st.pyplot(fig)


    # 3️⃣ --- Keyword Analysis
    st.markdown("### 🔍 Frequent Keywords in Reviews")

    # Token extraction
    words = " ".join(df['cleaned_text']).split()
    common_words = Counter([w for w in words if len(w) > 3])
    common_df = pd.DataFrame(common_words.most_common(15), columns=["Word", "Frequency"])

    # Separate positive & negative words
    pos_words = " ".join(df[df['sentiment_label'] == 'Positive']['cleaned_text']).split()
    neg_words = " ".join(df[df['sentiment_label'] == 'Negative']['cleaned_text']).split()

    top_pos = Counter([w for w in pos_words if len(w) > 3]).most_common(10)
    top_neg = Counter([w for w in neg_words if len(w) > 3]).most_common(10)

    pos_df = pd.DataFrame(top_pos, columns=["Word", "Frequency"])
    neg_df = pd.DataFrame(top_neg, columns=["Word", "Frequency"])

    # 4️⃣ --- Dual Bar Charts
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### 👍 Top Positive Keywords")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.barplot(y="Word", x="Frequency", data=pos_df, palette="Greens_r", ax=ax1)
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col5:
        st.markdown("#### 👎 Top Negative Keywords")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.barplot(y="Word", x="Frequency", data=neg_df, palette="Reds_r", ax=ax2)
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("")
        st.pyplot(fig2)

else:
    st.warning("⚠️ Sentiment labels missing. Please run Phase 2 analysis first.")

# ---------------------------
# 6️⃣ Optional Filter Sidebar
# ---------------------------
# ============================================
# 🧩 Sentiment Cluster Visualization (with labeled axes)
# ============================================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("### 🧠 Sentiment Clusters (t-SNE Visualization)")

# Limit to avoid heavy computation
sample_size = min(500, len(df))
X_sample = X_tfidf[:sample_size].toarray()
sentiment_sample = df['sentiment_label'][:sample_size]

# Reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto')
X_embedded = tsne.fit_transform(X_sample)

# Build DataFrame
df_vis = pd.DataFrame(X_embedded, columns=['TSNE-X', 'TSNE-Y'])
df_vis['sentiment'] = sentiment_sample.values

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    data=df_vis,
    x='TSNE-X',
    y='TSNE-Y',
    hue='sentiment',
    palette={'Positive': '#4CAF50', 'Neutral': '#FFB300', 'Negative': '#E53935'},
    alpha=0.8,
    s=50,
    ax=ax
)

ax.set_title("Semantic Clusters of Reviews", fontsize=13, color='white', pad=10)
ax.set_xlabel("Latent Semantic Dimension 1", fontsize=11, color='white')
ax.set_ylabel("Latent Semantic Dimension 2", fontsize=11, color='white')
ax.legend(title='Sentiment', fontsize=9, title_fontsize=10, loc='best')
ax.set_facecolor('#0E1117')  # match dark theme
fig.patch.set_facecolor('#0E1117')

st.pyplot(fig)
