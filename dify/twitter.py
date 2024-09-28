from pydantic import BaseModel
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint

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
app = App(name="twitter-scraper")

class TweetRequest(BaseModel):
    url: str

@app.function(image=image)
def scrape_tweet(url: str) -> dict:
    """
    Scrape a single tweet page for Tweet thread
    Return parent tweet, reply tweets and recommended tweets
    """
    from playwright.sync_api import sync_playwright

    _xhr_calls = []

    def intercept_response(response):
        """capture all background requests and save them"""
        if response.request.resource_type == "xhr":
            _xhr_calls.append(response)
        return response

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        context.set_default_timeout(120000)
        page = context.new_page()

        # enable background request intercepting:
        page.on("response", intercept_response)
        # go to url and wait for the page to load
        page.goto(url)
        page.wait_for_selector("[data-testid='tweet']")

        # find all tweet background requests:
        tweet_calls = [f for f in _xhr_calls if "TweetResultByRestId" in f.url]
        for xhr in tweet_calls:
            data = xhr.json()
            return data['data']['tweetResult']['result']

@app.function(image=image)
@web_endpoint(method="POST")
def get_tweet(request: TweetRequest):
    try:
        tweet_data = scrape_tweet.remote(request.url)
        return JSONResponse(content={"tweet_data": tweet_data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
