import random
import re
import requests
from bs4 import BeautifulSoup


# unrelated to crawling but still useful for a demo
def getInternalLinks(bs, base_url):
    internalLinks = []
    for link in bs.find_all("a", href=True):
        href = link["href"]
        if re.match(r"^(/|#)", href):
            full_url = requests.compat.urljoin(base_url, href)
        elif href.startswith(base_url):
            full_url = href
        else:
            continue
        if full_url not in internalLinks:
            internalLinks.append(full_url)
    return internalLinks


def getExternalLinks(bs, base_url):
    externalLinks = []
    for link in bs.find_all("a", href=True):
        href = link["href"]
        full_url = requests.compat.urljoin(base_url, href)
        if full_url.startswith("http") and not full_url.startswith(base_url):
            if full_url not in externalLinks:
                externalLinks.append(full_url)

    return externalLinks



random.seed(42)
BASE_URL = "https://en.wikipedia.org"
HEADERS = {"User-Agent": "EducationalScraper/1.0 (contact: Daniil Khlebnikov)"}

def get_links(article_path):
    url = f"{BASE_URL}{article_path}"
    r = requests.get(url, headers=HEADERS, timeout=5)
    r.raise_for_status()

    bs = BeautifulSoup(r.text, "html.parser")
    content = bs.find("div", id="bodyContent")

    if content is None:
        return []
    pattern = re.compile(r"^(/wiki/)((?!:).)*$")
    return content.find_all("a", href=pattern)

links = get_links("/wiki/Nightcrawler_(film)")
print(f"\t0. Nightcrawler (film)")
for i in range(1, 11):
    new_article = random.choice(links)["href"]
    print(f"\t{i}. {new_article.split('/')[-1].replace('_', ' ')}")
    links = get_links(new_article)
