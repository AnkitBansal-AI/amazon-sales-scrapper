"""
Amazon Estimated-Monthly-Sales microservice.

Given a keyword, fetches Amazon search-results page(s) (no product-page
visits at all) and returns the estimated total monthly sales value for
that keyword: sum of (units "bought in past month" x price) across the
unique products found.

This is a trimmed-down combination of the user's original search_phase.py
+ scraper_common.py, keeping ONLY what's needed to answer that one
question. Removed entirely:
  - CSV job files / SQLite database / any persistence (the caller's own
    backend is expected to store keyword/timestamp/result - see notes at
    bottom of this file)
  - seller_asin tracking / position-finding
  - top-product deep scrape (product detail page visit + all its parsing:
    breadcrumb, brand, BSR, returns policy, manufacturer, etc.)
  - research summary CSV export
  - the hardcoded os.chdir("C:/Users/Srikant/...") from both original
    files, which would crash outside that one machine

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
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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
# on how many products page 1 actually returned:
#   - page 1 has PAGE1_HIGH_THRESHOLD+ products  -> stop, 1 page is enough
#   - page 1 has fewer than PAGE1_LOW_THRESHOLD  -> fetch 2 more (3 total)
#   - otherwise (in between)                     -> fetch 1 more (2 total)
PAGE1_HIGH_THRESHOLD = 40
PAGE1_LOW_THRESHOLD = 20
PAGE_FETCH_DELAY_SECONDS = 2

# Amazon's search-results template hydrates extra product tiles in as you
# scroll, so grabbing page_source immediately after driver.get() only
# captures whatever rendered in the first second or two. Scroll to the
# bottom repeatedly, pausing to let new tiles hydrate, and stop once the
# tile count holds steady (or we hit the hard round cap).
SEARCH_RESULT_SELECTOR = '[data-component-type="s-search-result"]'
SCROLL_STABLE_ROUNDS_REQUIRED = 2
SCROLL_MAX_ROUNDS = 15
SCROLL_PAUSE_SECONDS = 1.2

REVIEWS_PATTERN = re.compile(r"^[\d,]+\s+ratings?$", re.IGNORECASE)
BOUGHT_PATTERN = re.compile(r"([\d,.]+)\s*([KMkm]?)\+?\s*bought", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

def build_driver():
    """
    Create a headless Chrome driver configured to look like a normal
    browser (Amazon actively checks for signs of automation).

    In Docker, set CHROME_BIN and CHROMEDRIVER_PATH env vars to point at
    apt-installed chromium/chromedriver (see Dockerfile) so this skips
    webdriver-manager's online version-matching lookup, which is both
    faster and more reliable in a container. Locally (no env vars set),
    it falls back to webdriver-manager, which auto-downloads a matching
    driver for whatever Chrome you have installed.
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

    # On a blocked/captcha page or a zero-result page this just times out
    # and falls through - is_blocked_page() / the "no products" handling
    # below takes it from there.
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
# Parse (trimmed to just what's needed for the sales estimate)
# ---------------------------------------------------------------------------

def _clean_number_string(value):
    return value.replace(",", "") if value else value


def parse_products(html: str) -> list[dict]:
    """
    Parse a rendered Amazon search-results page into a list of
    {"asin": ..., "bought_value": ...} dicts - the minimum needed to
    compute the estimated monthly sales value. No detail-page fields,
    no badges/images/delivery text/etc.
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

        products.append({"asin": asin, "bought_value": bought_value})

    return products


# ---------------------------------------------------------------------------
# Core scrape logic
# ---------------------------------------------------------------------------

class ScraperBlockedError(Exception):
    """Raised when Amazon blocked the very first page (no data at all)."""


def _target_page_count(first_page_product_count: int) -> int:
    """
    Amazon's search-results page inconsistently renders anywhere from
    ~16 to ~48 product tiles on page 1 depending on the keyword, so a
    fixed page count either wastes time re-fetching an already-dense page
    or undercounts a sparse one. Decide how many pages are worth fetching
    based on what page 1 actually returned:
      - 40+ products on page 1  -> that's already a full page, stop at 1
      - 20-39 products          -> fetch one more page (2 total)
      - <20 products            -> sparse page, fetch two more (3 total)
    """
    if first_page_product_count >= PAGE1_HIGH_THRESHOLD:
        return 1
    elif first_page_product_count >= PAGE1_LOW_THRESHOLD:
        return 2
    else:
        return 3


def get_estimated_monthly_sales(keyword: str) -> dict:
    driver = build_driver()
    start = time.time()
    products_by_asin: dict[str, dict] = {}
    unkeyed_bought_values: list[float] = []  # rare: card has no asin at all
    pages_fetched = 0

    def _collect(page_products: list[dict]) -> None:
        for p in page_products:
            asin = p.get("asin")
            if asin:
                products_by_asin.setdefault(asin, p)
            elif p.get("bought_value") is not None:
                unkeyed_bought_values.append(p["bought_value"])

    try:
        # Page 1 is always fetched first; it also tells us how many more
        # pages (if any) are worth fetching for this particular keyword.
        html = fetch_search_page(driver, keyword, page=1)

        if is_blocked_page(html):
            raise ScraperBlockedError(
                f"Amazon blocked the request for keyword '{keyword}' (page 1)."
            )

        page_products = parse_products(html)
        pages_fetched = 1
        _collect(page_products)

        target_pages = _target_page_count(len(page_products))
        logger.info(
            "Keyword=%r: page 1 returned %d products -> targeting %d page(s) total",
            keyword, len(page_products), target_pages,
        )

        for page_num in range(2, target_pages + 1):
            time.sleep(PAGE_FETCH_DELAY_SECONDS)
            html = fetch_search_page(driver, keyword, page=page_num)

            if is_blocked_page(html):
                break  # keep whatever was already collected from earlier pages

            page_products = parse_products(html)
            if not page_products:
                break  # ran out of results before hitting the target

            _collect(page_products)
            pages_fetched += 1

        products = list(products_by_asin.values())
        estimated_monthly_sales_value = sum(
            p["bought_value"] for p in products if p.get("bought_value") is not None
        ) + sum(unkeyed_bought_values)

        return {
            "keyword": keyword,
            "estimated_monthly_sales_value": round(estimated_monthly_sales_value, 2),
            "num_products_found": len(products) + len(unkeyed_bought_values),
            "pages_fetched": pages_fetched,
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

app = FastAPI(title="Amazon Sales Estimator Microservice")


class ScrapeRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)


class ScrapeResponse(BaseModel):
    keyword: str
    estimated_monthly_sales_value: float
    num_products_found: int
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
        result = get_estimated_monthly_sales(keyword)
    except ScraperBlockedError as e:
        logger.warning("Blocked: %s", e)
        # 503 so the caller (your backend) knows to retry later / back off,
        # rather than treating this as a permanent failure.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Scrape failed for keyword=%r", keyword)
        raise HTTPException(status_code=500, detail=f"Scrape failed: {e}")

    logger.info("Done keyword=%r -> %s", keyword, result)
    return result


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# This service is stateless by design - it does NOT store keyword/timestamp/
# result anywhere. That's intentionally left to your main backend API (the
# "job" layer from the architecture we discussed): it should call POST
# /scrape, then persist {keyword, timestamp, result} in its own database.
# Keeping this service stateless makes it trivial to scale horizontally
# (run N copies behind a load balancer) without any shared state to worry
# about here.
