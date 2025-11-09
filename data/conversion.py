import pandas as pd
import json

# ---- Load the processed JSON ----
with open("data\processed\cleaned_reviews.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Case 1: If JSON is a list of dicts ----
if isinstance(data, list):
    df = pd.DataFrame(data)

# ---- Case 2: If JSON has a top-level key like {"reviews": [...] } ----
elif isinstance(data, dict):
    # find list-like key automatically
    for k, v in data.items():
        if isinstance(v, list):
            df = pd.DataFrame(v)
            break
    else:
        raise ValueError("No list found inside JSON file.")

# ---- Save as CSV ----
df.to_csv("processed_reviews.csv", index=False)
print("✅ Converted processed_reviews.json → processed_reviews.csv")
print("📊 Columns:", list(df.columns))
print(df.head())
