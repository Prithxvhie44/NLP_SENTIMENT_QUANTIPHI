"""
Amazon Review Scraper (SerpAPI Version - Safe & Reliable)
---------------------------------------------------------
Fetches Amazon product reviews using SerpAPI.
Requires a free API key from https://serpapi.com/
"""

import os
import json
import re
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")


def get_asin_from_url(url: str):
    """Extract ASIN from product URL (for naming only)"""
    match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url)
    return match.group(1) if match else "unknown"


def clean_amazon_url(url: str):
    """Ensure the URL is in clean ASIN format for SerpAPI"""
    match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url)
    if match:
        asin = match.group(1)
        return f"https://www.amazon.in/dp/{asin}"
    return url


def scrape_amazon_reviews_serpapi(product_url: str, api_key: str, country="in"):
    """
    Fetch reviews using SerpAPI (Amazon Reviews Engine).
    Falls back to 'google_shopping' if no reviews are found.
    """

    product_url = clean_amazon_url(product_url)
    asin = get_asin_from_url(product_url)

    params = {
        "engine": "amazon_reviews",
        "amazon_domain": f"amazon.{country}",
        "product_url": product_url,
        "api_key": api_key,
    }

    print("🔍 Fetching reviews via SerpAPI...")
    search = GoogleSearch(params)
    results = search.get_dict()

    reviews = results.get("reviews", [])

    # ✅ Fallback if empty (optional but highly recommended)
    if not reviews:
        print("⚠️ No reviews found using 'amazon_reviews'. Trying Google Shopping...")
        params_fallback = {
            "engine": "google_shopping",
            "q": f"site:amazon.{country} {asin} reviews",
            "api_key": api_key,
        }
        fallback = GoogleSearch(params_fallback).get_dict()
        products = fallback.get("shopping_results", [])
        reviews = []

        for p in products:
            if "extensions" in p:
                reviews.append({
                    "title": p.get("title"),
                    "text": " | ".join(p["extensions"]),
                    "source": "GoogleShopping"
                })

    print(f"✅ Retrieved {len(reviews)} reviews from {product_url}")

    os.makedirs("data/raw", exist_ok=True)
    out_path = f"data/raw/amazon_{asin}_serpapi.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved reviews → {out_path}")
    return reviews


if __name__ == "__main__":
    api_key = API_KEY
    TEST_URL = "https://www.amazon.com/dp/B0BQLBNBV1"  # ✅ product with active reviews
    data = scrape_amazon_reviews_serpapi(TEST_URL, api_key,country="com")
    print(f"✅ Total reviews fetched: {len(data)}")
