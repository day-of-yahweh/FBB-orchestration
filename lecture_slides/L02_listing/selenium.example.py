import csv
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


def normalize(s):
    return " ".join(line.rstrip() for line in s.strip().splitlines())


def scroll_to_bottom(driver, pause=1):
    """Прокручивает страницу вниз до конца с паузами для подгрузки контента."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


options = Options()
# Инициализация Firefox-драйвера
options = Options()
firefox_binary_path = "/snap/firefox/current/usr/lib/firefox/firefox"
options.binary = FirefoxBinary(firefox_binary_path)


driver = webdriver.Firefox(service=Service(), options=options)
driver.get("https://www.amazon.com/s?k=labubu")
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-component-type='s-search-result']"))
)
scroll_to_bottom(driver, pause=1.5)
product_cards = driver.find_elements(By.CSS_SELECTOR, "div[data-component-type='s-search-result']")


results = []
for card in product_cards:
    title = card.find_element(By.CSS_SELECTOR, "h2 span").text
    title = normalize(title)
    try:
        price_whole = card.find_element(By.CSS_SELECTOR, ".a-price-whole").text
        price_whole = price_whole.replace(',', '')
        price_frac = card.find_element(By.CSS_SELECTOR, ".a-price-fraction").text
        price = f"{price_whole}.{price_frac}"
        results.append({"name": title, "price": price})
    except Exception as e:
        print("No selector for price available")
with open("labubu_prices.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "price"])
    writer.writeheader()
    writer.writerows(results)
driver.quit()
