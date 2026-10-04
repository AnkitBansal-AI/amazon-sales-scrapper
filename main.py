"""
Amazon Product Sales + Keyword Search microservice.

Given a keyword, in ONE browser session:
  1. PRODUCT SALES: fetches Amazon search-results page(s) and returns the
     estimated total monthly sales value, plus the top 5 products by
     sales value (title, price per unit, total sales value, image).
  2. KEYWORD SEARCH: reuses the SAME already-open page (no second page
     load) to type the keyword into Amazon's nav search box and read
     back the autocomplete suggestions.

Run locally (needs Chrome installed; webdriver-manager fetches a matching
chromedriver automatically):
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
    curl -X POST http://localhost:8000/scrape -H "Content-Type: application/json" -d '{"keyword": "wireless mouse"}'
"""

import logging
import os
import re
import time
import urllib.parse

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper-service")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AMAZON_DOMAIN = os.environ.get("AMAZON_DOMAIN", "amazon.in")

# Adaptive page-count logic: Amazon's page 1 itself can return anywhere
# from ~16 to ~48 organic tiles depending on the category, so rather than
# always fetching a fixed number of pages, decide how many to fetch based
# on how many products page 1 actually returned.
PAGE1_HIGH_THRESHOLD = 40
PAGE1_LOW_THRESHOLD = 20
PAGE_FETCH_DELAY_SECONDS = 2

TOP_PRODUCTS_COUNT = 5

SEARCH_RESULT_SELECTOR = '[data-component-type="s-search-result"]'
SCROLL_STABLE_ROUNDS_REQUIRED = 2
SCROLL_MAX_ROUNDS = 15
SCROLL_PAUSE_SECONDS = 1.2

REVIEWS_PATTERN = re.compile(r"^[\d,]+\s+ratings?$", re.IGNORECASE)
BOUGHT_PATTERN = re.compile(r"([\d,.]+)\s*([KMkm]?)\+?\s*bought", re.IGNORECASE)

AUTOCOMPLETE_TIMEOUT = 10
TYPING_WAIT_SECONDS = 1.0
SUGGESTION_COUNT = 10


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

def build_driver():
    """
    Create a headless Chrome driver configured to look like a normal
    browser (Amazon actively checks for signs of automation).
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        options.binary_location = chrome_bin

    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")
    service = Service(chromedriver_path) if chromedriver_path else Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_search_page(driver, keyword: str, page: int = 1) -> str:
    query = urllib.parse.quote_plus(keyword)
    url = f"https://www.{AMAZON_DOMAIN}/s?k={query}&page={page}"
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SEARCH_RESULT_SELECTOR))
        )
    except Exception:
        pass

    last_count = -1
    stable_rounds = 0
    for _ in range(SCROLL_MAX_ROUNDS):
        count = len(driver.find_elements(By.CSS_SELECTOR, SEARCH_RESULT_SELECTOR))
        if count == last_count:
            stable_rounds += 1
            if stable_rounds >= SCROLL_STABLE_ROUNDS_REQUIRED:
                break
        else:
            stable_rounds = 0
        last_count = count
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SECONDS)

    html = driver.page_source
    logger.info("Fetched page %d for %r: %d chars, %d tiles", page, keyword, len(html), last_count)
    return html


def is_blocked_page(html: str) -> bool:
    if not html:
        return True
    lowered = html.lower()
    markers = [
        "enter the characters you see below",
        "sorry, we just need to make sure you're not a robot",
        "api-services-support@amazon.com",
        "/errors/validatecaptcha",
    ]
    return any(marker in lowered for marker in markers)


# ---------------------------------------------------------------------------
# Parse search-results page: asin, title, price, image, bought_value
# ---------------------------------------------------------------------------

def parse_products(html: str) -> list[dict]:
    """
    Parse a rendered Amazon search-results page into a list of product
    dicts with just what we need: asin, title, price, image_url, and
    bought_value (estimated total sales value for that product).
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    cards = soup.find_all("div", {"data-component-type": "s-search-result"})

    products = []
    for card in cards:
        title_el = card.find("h2")
        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue  # not a real product tile (e.g. a widget/ad slot)

        asin = card.get("data-asin") or None

        img_el = card.find("img", class_="s-image")
        image_url = img_el.get("src") if img_el else None

        price_whole = card.find("span", {"class": "a-price-whole"})
        price_fraction = card.find("span", {"class": "a-price-fraction"})
        price = None
        if price_whole:
            whole = price_whole.get_text(strip=True).rstrip(".").replace(",", "")
            fraction = price_fraction.get_text(strip=True) if price_fraction else "00"
            price = f"{whole}.{fraction}"

        num_units_bought = None
        bought_el = card.find(string=re.compile(r"bought in (past|last) month", re.IGNORECASE))
        if bought_el:
            match = BOUGHT_PATTERN.search(bought_el)
            if match:
                number = float(match.group(1).replace(",", ""))
                suffix = match.group(2).upper()
                multiplier = {"K": 1_000, "M": 1_000_000}.get(suffix, 1)
                num_units_bought = int(number * multiplier)

        bought_value = None
        if num_units_bought is not None and price is not None:
            try:
                bought_value = num_units_bought * float(price)
            except (ValueError, TypeError):
                bought_value = None

        products.append({
            "asin": asin,
            "title": title,
            "price": price,
            "image_url": image_url,
            "bought_value": bought_value,
        })

    return products


# ---------------------------------------------------------------------------
# Autocomplete: reuse the already-open page's nav search box
# ---------------------------------------------------------------------------

def get_autocomplete_suggestions(driver, keyword: str) -> list[str]:
    """
    Type the keyword into the nav search box on whatever page is
    currently loaded (no separate navigation to the homepage) and read
    back the autocomplete dropdown suggestions.
    """
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.3)

    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "twotabsearchtextbox"))
        )
    except TimeoutException:
        logger.warning("Could not find Amazon search box for autocomplete on keyword=%r", keyword)
        return []

    search_box.click()
    search_box.send_keys(Keys.CONTROL, "a")
    search_box.send_keys(Keys.BACKSPACE)
    search_box.send_keys(keyword)

    time.sleep(TYPING_WAIT_SECONDS)

    def _extract() -> list[str]:
        selectors = [
            (By.CSS_SELECTOR, "div.s-suggestion"),
            (By.CSS_SELECTOR, "div[role='option']"),
            (By.CSS_SELECTOR, "[data-type='suggestion']"),
        ]
        for by, selector in selectors:
            candidates = driver.find_elements(by, selector)
            visible = [el for el in candidates if el.is_displayed()]
            if visible:
                seen = set()
                suggestions = []
                for el in visible:
                    text = " ".join(el.text.split()).strip()
                    if text and text.casefold() not in seen:
                        seen.add(text.casefold())
                        suggestions.append(text)
                return suggestions
        return []

    try:
        WebDriverWait(driver, AUTOCOMPLETE_TIMEOUT).until(lambda d: len(_extract()) > 0)
    except TimeoutException:
        pass

    return _extract()[:SUGGESTION_COUNT]


# ---------------------------------------------------------------------------
# Core combined logic
# ---------------------------------------------------------------------------

class ScraperBlockedError(Exception):
    """Raised when Amazon blocked the very first page (no data at all)."""


def _target_page_count(first_page_product_count: int) -> int:
    if first_page_product_count >= PAGE1_HIGH_THRESHOLD:
        return 1
    elif first_page_product_count >= PAGE1_LOW_THRESHOLD:
        return 2
    else:
        return 3


def run_product_search(driver, keyword: str) -> dict:
    products_by_asin: dict[str, dict] = {}
    pages_fetched = 0

    html = fetch_search_page(driver, keyword, page=1)

    if is_blocked_page(html):
        raise ScraperBlockedError(
            f"Amazon blocked the request for keyword '{keyword}' (page 1)."
        )

    page_products = parse_products(html)
    pages_fetched = 1
    for p in page_products:
        asin = p.get("asin")
        if asin:
            products_by_asin.setdefault(asin, p)

    target_pages = _target_page_count(len(page_products))
    logger.info(
        "Keyword=%r: page 1 returned %d products -> targeting %d page(s) total",
        keyword, len(page_products), target_pages,
    )

    for page_num in range(2, target_pages + 1):
        time.sleep(PAGE_FETCH_DELAY_SECONDS)
        html = fetch_search_page(driver, keyword, page=page_num)

        if is_blocked_page(html):
            break

        page_products = parse_products(html)
        if not page_products:
            break

        for p in page_products:
            asin = p.get("asin")
            if asin:
                products_by_asin.setdefault(asin, p)

        pages_fetched += 1

    products = list(products_by_asin.values())
    estimated_monthly_sales_value = sum(
        p["bought_value"] for p in products if p.get("bought_value") is not None
    )

    ranked = sorted(
        [p for p in products if p.get("bought_value") is not None],
        key=lambda p: p["bought_value"],
        reverse=True,
    )
    top_products = [
        {
            "title": p["title"],
            "price": p["price"],
            "total_sales_value": round(p["bought_value"], 2),
            "image_url": p["image_url"],
            "product_url": f"https://www.{AMAZON_DOMAIN}/dp/{p['asin']}" if p.get("asin") else None,
        }
        for p in ranked[:TOP_PRODUCTS_COUNT]
    ]

    return {
        "estimated_monthly_sales_value": round(estimated_monthly_sales_value, 2),
        "num_products_found": len(products),
        "pages_fetched": pages_fetched,
        "top_products": top_products,
    }


def run_combined_search(keyword: str) -> dict:
    driver = build_driver()
    start = time.time()

    try:
        product_result = run_product_search(driver, keyword)
        suggested_keywords = get_autocomplete_suggestions(driver, keyword)

        return {
            "keyword": keyword,
            "estimated_monthly_sales_value": product_result["estimated_monthly_sales_value"],
            "num_products_found": product_result["num_products_found"],
            "top_products": product_result["top_products"],
            "suggested_keywords": suggested_keywords,
            "pages_fetched": product_result["pages_fetched"],
            "elapsed_seconds": round(time.time() - start, 1),
        }
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Amazon Product Sales + Keyword Search Microservice")


class ScrapeRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)


class TopProduct(BaseModel):
    title: str
    price: str | None = None
    total_sales_value: float
    image_url: str | None = None
    product_url: str | None = None


class ScrapeResponse(BaseModel):
    keyword: str
    estimated_monthly_sales_value: float
    num_products_found: int
    top_products: list[TopProduct]
    suggested_keywords: list[str]
    pages_fetched: int
    elapsed_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scrape", response_model=ScrapeResponse)
def scrape(req: ScrapeRequest):
    keyword = req.keyword.strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="keyword must not be empty")

    logger.info("Scraping keyword=%r", keyword)
    try:
        result = run_combined_search(keyword)
    except ScraperBlockedError as e:
        logger.warning("Blocked: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Scrape failed for keyword=%r", keyword)
        raise HTTPException(status_code=500, detail=f"Scrape failed: {e}")

    logger.info("Done keyword=%r -> %d products, %d suggestions, %.1fs",
                keyword, result["num_products_found"], len(result["suggested_keywords"]), result["elapsed_seconds"])
    return result


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# This service is stateless by design - it does NOT store keyword/timestamp/
# result anywhere. That's left to the backend API, which calls POST /scrape
# and persists {keyword, timestamp, result} in its own database.
