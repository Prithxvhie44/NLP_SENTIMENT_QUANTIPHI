# flipkart_bs_scraper.py
# Usage:
# python flipkart_bs_scraper.py --url "<FLIPKART_REVIEW_URL>" --pages 10

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import argparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

def get_page_html(url):
    """Fetch HTML content for a page."""
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.text
    else:
        print(f"⚠️ Failed to fetch {url} — status code {response.status_code}")
        return None

def parse_reviews(html):
    """Extract reviews using BeautifulSoup."""
    soup = BeautifulSoup(html, 'html.parser')
    reviews_data = []

    # Flipkart review containers have class _16PBlm (as of late 2025)
    review_blocks = soup.find_all('div', {'class': '_16PBlm'})
    for block in review_blocks:
        try:
            rating = block.find('div', {'class': '_3LWZlK'}).text.strip()
        except:
            rating = None
        try:
            title = block.find('p', {'class': '_2-N8zT'}).text.strip()
        except:
            title = None
        try:
            review_text = block.find('div', {'class': 't-ZTKy'}).div.text.strip()
            review_text = review_text.replace('READ MORE', '').strip()
        except:
            review_text = None
        try:
            username = block.find('p', {'class': '_2sc7ZR'}).text.strip()
        except:
            username = None

        if review_text:
            reviews_data.append({
                'username': username,
                'rating': rating,
                'title': title,
                'review_text': review_text
            })

    return reviews_data

def scrape_flipkart_reviews(base_url, max_pages=5):
    """Scrape reviews from multiple pages."""
    all_reviews = []

    for page in range(1, max_pages + 1):
        page_url = f"{base_url}&page={page}"
        print(f"📄 Scraping page {page}: {page_url}")
        html = get_page_html(page_url)
        if not html:
            break

        reviews = parse_reviews(html)
        if not reviews:
            print("⚠️ No reviews found on this page.")
            break

        all_reviews.extend(reviews)
        time.sleep(1.5)  # polite delay

    return all_reviews

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Flipkart product reviews page URL")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    args = parser.parse_args()

    reviews = scrape_flipkart_reviews(args.url, args.pages)
    print(f"\n✅ Collected {len(reviews)} total reviews.")

    # Save results
    df = pd.DataFrame(reviews)
    df.to_csv("reviews.csv", index=False)
    with open("reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)

    print("💾 Saved: reviews.csv and reviews.json")

if __name__ == "__main__":
    main()
