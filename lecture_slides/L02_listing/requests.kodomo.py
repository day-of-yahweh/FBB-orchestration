import requests
from bs4 import BeautifulSoup

url = "https://kodomo.fbb.msu.ru"

try:
  with requests.Session() as s:
    r = s.get(url, timeout=5, allow_redirects=True)
    r.raise_for_status()
    data = r.text
    print(r.url, '\n--------------')
    print(data)
except requests.RequestException as e:
    print("Request failed:", e)


bs = BeautifulSoup(data, 'html.parser')
print(bs.find('h1'))
print(bs.find_all('h1'))
print(bs.select('div > a'))
