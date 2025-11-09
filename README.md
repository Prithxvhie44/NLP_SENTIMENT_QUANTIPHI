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

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Prithxvhie44/NLP_SENTIMENT_QUANTIPHI.git
cd NLP_SENTIMENT_QUANTIPHI
2️⃣ Create and activate virtual environment
bash
Copy code
python -m venv nlp_env
nlp_env\Scripts\activate     # Windows
source nlp_env/bin/activate  # macOS/Linux
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
python -m spacy download en_core_web_sm
4️⃣ Run the Streamlit Dashboard
bash
Copy code
streamlit run app/app.py

🧠 Workflow Overview
Step	Description	Output
Data Conversion	Converts scraped JSON → CSV	processed_reviews.csv
Analysis	Performs POS, NER, TF-IDF, Word2Vec, Sentiment, LSA	phase2_output.csv
Summarization	Clusters reviews & extracts key feedback	phase3_summary.json
Q&A Generation	Creates automatic Q&A pairs from insights	phase3_qa.csv
Dashboard	Interactive visualization & question answering	Web App
