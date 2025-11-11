import time
import json
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

def get_html_selenium(url):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)  
    html = driver.page_source
    driver.quit()
    return html

def parse_flipkart_review_html(html):

    soup = BeautifulSoup(html, "lxml")
    reviews = []

    for card in soup.select("div.col.EPCmJX.Ma1fCG"):
        title_element = card.select_one("div.row p, div.row strong, div.row span")
        rating_element = card.select_one("div.row")  
        text_element = card.select_one("div.ZmyHeo, div.row + div.row + div div")  
        if not text_element:
            
            divs = card.find_all("div")
            if divs:
                text_element = divs[-1]

        text = text_element.get_text(separator=" ").strip() if text_element else ""
        rating = rating_element.get_text(strip=True) if rating_element else ""
        title = title_element.get_text(strip=True) if title_element else ""

        if text:
            reviews.append({
                "rating": rating,
                "title": title,
                "review_text": text
            })
    return reviews

def scrape(product_reviews_url, max_pages=5, save_json="../data/raw/reviews_flipkart.json"):
    os.makedirs(os.path.dirname(save_json), exist_ok=True)

    all_reviews = []
    for page in tqdm(range(1, max_pages + 1)):
        paged_url = f"{product_reviews_url}&page={page}"
        html = get_html_selenium(paged_url)
        reviews = parse_flipkart_review_html(html)
        if not reviews:
            break
        all_reviews.extend(reviews)
        time.sleep(2)
    all_reviews = list({r["review_text"]: r for r in all_reviews}.values())
    with open(save_json, "w", encoding="utf8") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_reviews)} reviews {save_json}")
    return all_reviews

if __name__ == "__main__":
    PRODUCT_URL = "https://www.flipkart.com/aman-fit-best-quality-pvc-set-5kgs-x-2-pcs-1-pair-dumbbells-hex-fixed-weight-dumbbell/p/itm143e5291f79cb?pid=DBLGHZWGW3NZBKHR&lid=LSTDBLGHZWGW3NZBKHRUZ3IJR&marketplace=FLIPKART&q=dubmells&store=qoc%2Facb%2Fzuc&spotlightTagId=default_FkPickId_qoc%2Facb%2Fzuc&srno=s_1_5&otracker=search&otracker1=search&fm=Search&iid=179c18cc-d462-404c-a035-79bce45a583a.DBLGHZWGW3NZBKHR.SEARCH&ppt=sp&ppn=sp&ssid=mkeq625kcw0000001762662466465&qH=c7a243940c9d7410"
    scrape(PRODUCT_URL, max_pages=50)