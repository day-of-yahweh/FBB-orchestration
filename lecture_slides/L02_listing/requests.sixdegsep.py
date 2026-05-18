import re
import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "EducationalScraper/1.0 (contact: Daniil Khlebnikov)"}
r = requests.get("https://en.wikipedia.org/wiki/Kevin_Bacon",
                 headers=headers)
bs = BeautifulSoup(r.text, "html.parser")
pattern = re.compile(r"^(/wiki/)((?!:).)*$")
content = bs.find("div", id="bodyContent")
for link in content.find_all("a", href=pattern):
    print(link["href"])
