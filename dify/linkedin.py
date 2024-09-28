from pydantic import BaseModel
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint
from typing import List, Optional

# Create Modal Image with required dependencies
image = Image.debian_slim().pip_install("playwright").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "playwright install-deps chromium",
    "playwright install chromium",
)

# Create Modal App
app = App(name="linkedin-job-scraper")

class JobRequest(BaseModel):
    location: str
    keywords: Optional[str] = None
    limit: Optional[int] = 10

@app.function(image=image)
def scrape_linkedin_jobs(location: str, keywords: Optional[str] = None, limit: int = 10) -> List[dict]:
    """
    Scrape LinkedIn job postings based on location and optional keywords
    """
    from playwright.sync_api import sync_playwright
    import re

    search_url = f"https://www.linkedin.com/jobs/search/?location={location}"
    if keywords:
        search_url += f"&keywords={keywords}"

    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        context.set_default_timeout(120000)
        page = context.new_page()
        page.goto(search_url)

        # Wait for job listings to load
        page.wait_for_selector(".jobs-search__results-list")

        # Scroll to load more jobs
        for _ in range(limit // 25 + 1):  # LinkedIn loads ~25 jobs per scroll
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        job_cards = page.query_selector_all(".jobs-search__results-list > li")

        for card in job_cards[:limit]:
            title_elem = card.query_selector(".base-search-card__title")
            company_elem = card.query_selector(".base-search-card__subtitle")
            location_elem = card.query_selector(".job-search-card__location")
            link_elem = card.query_selector("a.base-card__full-link")

            if title_elem and company_elem and location_elem and link_elem:
                job = {
                    "title": title_elem.inner_text(),
                    "company": company_elem.inner_text(),
                    "location": location_elem.inner_text(),
                    "link": link_elem.get_attribute("href"),
                }
                jobs.append(job)

        browser.close()

    return jobs

@app.function(image=image)
@web_endpoint(method="POST")
def get_linkedin_jobs(request: JobRequest):
    try:
        jobs = scrape_linkedin_jobs.remote(request.location, request.keywords, request.limit)
        return JSONResponse(content={"jobs": jobs})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    app.serve()