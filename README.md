# 🧠 NLP Sentiment Analysis — Quantiphi Project
### *Comprehensive Product Review Analysis for Customer Insights*

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📘 Project Overview
This project focuses on analyzing **Flipkart product reviews** using classical NLP techniques (no Transformer or generative AI).  
It aims to extract **sentiment**, detect **key product features**, and generate **actionable insights** for customers and product developers through a **Streamlit dashboard**.

### 🎯 Objectives
- Collect and preprocess customer reviews  
- Perform **POS tagging, NER, TF-IDF, Word2Vec**, and **sentiment analysis**  
- Extract topics using **Latent Semantic Analysis (LSA)**  
- Generate **Q&A summaries** automatically  
- Present results in an **interactive dashboard**

---

## 🧱 Repository Structure
NLP_SENTIMENT_QUANTIPHI/
│
├── app/
│ └── app.py # Streamlit interactive dashboard
│
├── data/
│ ├── raw/ # Raw scraped Flipkart data
│ ├── processed/ # Cleaned and analyzed files
│ │ ├── processed_reviews.json
│ │ ├── phase2_output.csv
│ │ ├── sentiment_reviews.json
│ │ ├── topics.json
│ │ └── summary.json
│
├── scripts/
│ ├── conversion.py # JSON → CSV converter
│ ├── flipkart_scraper.py # BeautifulSoup scraper
│ ├── phase2_analysis.py # POS, TF-IDF, Sentiment, Topic Modeling
│ └── phase3_summary_qa.py # Summarization & Q&A generator
│
├── models/
│ ├── word2vec.model # Trained Word2Vec model
│ ├── tfidf_vectorizer.joblib # TF-IDF vectorizer
│ └── lstm_sentiment_model.h5 # Optional sentiment classifier
│
├── reports/
│ ├── Quantiphi_Project_Report.md # Final Markdown report
│ ├── Quantiphi_Project_Report.pdf # PDF version
│ └── visuals/ # Images for documentation
│ ├── sentiment_donut.png
│ ├── cluster_tsne.png
│ ├── wordcloud_positive.png
│ └── dashboard_screenshot.png
│
├── .streamlit/
│ └── config.toml # Dark teal dashboard theme
│
├── requirements.txt # All dependencies
├── README.md # (This file)
├── LICENSE # MIT License
└── .gitignore # Ignored cache/checkpoints




---

## ⚙️ Installation & Setup


```bash
1️⃣ Clone the repository
git clone https://github.com/Prithxvhie44/NLP_SENTIMENT_QUANTIPHI.git
cd NLP_SENTIMENT_QUANTIPHI
2️⃣ Create and activate a virtual environment
python -m venv nlp_env
nlp_env\Scripts\activate     # Windows
source nlp_env/bin/activate  # macOS/Linux
3️⃣ Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

4️⃣ Run the Streamlit Dashboard
streamlit run app/app.py
---
```

🧠 Workflow Overview

| Step                | Description                                         | Output                  |
| ------------------- | --------------------------------------------------- | ----------------------- |
| **Data Conversion** | Converts scraped JSON → CSV                         | `processed_reviews.csv` |
| **Analysis**        | Performs POS, NER, TF-IDF, Word2Vec, Sentiment, LSA | `phase2_output.csv`     |
| **Summarization**   | Clusters reviews & extracts key feedback            | `phase3_summary.json`   |
| **Q&A Generation**  | Creates automatic Q&A pairs from insights           | `phase3_qa.csv`         |
| **Dashboard**       | Interactive visualization & question answering      | Web App                 |


📊 Dashboard Features

| Feature                            | Description                                             |
| ---------------------------------- | ------------------------------------------------------- |
| 🗂️ **Metrics Overview**           | Total reviews, average rating, sentiment ratio          |
| 🥧 **Sentiment Donut Chart**       | Distribution of Positive / Neutral / Negative reviews   |
| 📈 **Keyword Charts**              | Top keywords for positive & negative sentiments         |
| ☁️ **Word Cloud**                  | Visual representation of most frequent terms            |
| 🧩 **t-SNE Clusters**              | 2D visualization of semantic similarity between reviews |
| 📉 **Negative Feedback Extractor** | Displays top negative or mixed reviews                  |
| 💬 **Q&A Query Input**             | Users can ask: “What do customers say about price?”     |

🧾 Key Results

Metric	Observation
Total Reviews	380
Average Rating	4.2 / 5
Positive Sentiment	~85%
Neutral Sentiment	~10%
Negative Sentiment	~5%
Top Positive Keywords	grip, comfort, quality, packaging
Top Negative Keywords	rubber, sound, price, filling

🧩 Negative Feedback Highlights

“Rubber started peeling after a few days.”

“Gap inside one dumbbell makes rattling sound.”

“Price is slightly higher than expected.”

🧮 Visualizations

Sentiment Distribution Donut Chart
Top Positive & Negative Keywords (Bar Chart)
Word Cloud by Sentiment
t-SNE Semantic Clusters
Auto-generated Review Summaries

📸 Dashboard Preview


Dashboard Overview	
<img width="1878" height="610" alt="image" src="https://github.com/user-attachments/assets/5fec6ae0-00f9-47c0-a3f6-df512db52adf" />


Sentiment Donut Chart	
<img width="587" height="586" alt="image" src="https://github.com/user-attachments/assets/a28c5980-5703-49ef-9337-17293f259fe5" />

Word Cloud	

<img width="837" height="588" alt="image" src="https://github.com/user-attachments/assets/81b962de-f26d-406c-852f-67dc60d2c53e" />
<img width="809" height="588" alt="image" src="https://github.com/user-attachments/assets/50fdfb1b-a04c-485e-9a61-fef7fd279ef0" />




Cluster Visualization	
<img width="1125" height="798" alt="image" src="https://github.com/user-attachments/assets/9c59a2a7-f559-4d86-a599-2acb9d1555b3" />

