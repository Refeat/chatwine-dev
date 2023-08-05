import json
import argparse
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import requests

def get_driver():
    options = Options()

    # set up the proxy server settings
    # prox = Proxy()
    # prox.proxy_type = ProxyType.MANUAL
    # prox.http_proxy = "127.0.0.1:8080"
    # prox.socks_proxy = "127.0.0.1:8080"
    # prox.ssl_proxy = "127.0.0.1:8080"

    # capabilities = webdriver.DesiredCapabilities.CHROME
    # prox.add_to_capabilities(capabilities)

    # options.add_argument("--headless")  # Ensure GUI is off
    # options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # webdriver_service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(desired_capabilities=capabilities, options=options)
    driver.get('https://www.vivino.com')
    return driver

def vivino_search(driver, name, country=None, state=None, minPrice=None, maxPrice=None, noPriceIncluded=None, minRatings=None, maxRatings=None, minAverage=None, maxAverage=None):
    vinos = []
    index = 1
    # Go to the search page
    driver.get(f'https://www.vivino.com/search/wines?q={name}&start={index}')
    time.sleep(2)  # ensure page load

    # Collect items
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # print(soup)
    cards = soup.find_all('div', class_='default-wine-card')
    for card in cards:
        vino = {}
        name_elem = card.select_one('.wine-card__name a')
        try:
            vino['name'] = name_elem.text.strip() if name_elem else None
        except:
            continue

        try:
            vino['link'] = card.select_one('a[data-cartitemsource="text-search"]')['href']
        except:
            continue

        try:
            style = card.select_one('figure.wine-card__image')['style']
            # Extract URL from style string
            vino['image_link'] = "http:" + style.split('"')[1]
        except:
            continue



        try:
            region_elem = card.select_one('.wine-card__region a[data-item-type="country"]')
            vino['country'] = region_elem.text.strip() if region_elem else None
        except:
            vino['country'] = None

        try:
            price_elem = card.select_one('.wine-price-value')
            vino['price'] = None if price_elem.text == '—' else float(price_elem.text.replace(',', ''))
        except:
            vino['price'] = None

        try:
            average_rating_elem = card.select_one('.average__number')
            vino['average_rating'] = float(average_rating_elem.text) if average_rating_elem else None
        except:
            vino['average_rating'] = None

        try:
            ratings_elem = card.select_one('.text-micro')
            ratings = ratings_elem.text.split() if ratings_elem else None
            vino['ratings'] = int(ratings[0].replace(',', '')) if ratings else None
        except:
            vino['ratings'] = None




        # Check filters
        if (minPrice and vino['price'] and vino['price'] < minPrice) or \
            (maxPrice and vino['price'] > maxPrice) or \
            (minRatings and vino['ratings'] < minRatings) or \
            (maxRatings and vino['ratings'] > maxRatings) or \
            (minAverage and vino['average_rating'] < minAverage) or \
            (maxAverage and vino['average_rating'] > maxAverage):
            continue

        # If we reached here, it means that the vino passed the filters
        vinos.append(vino)

    # Save the vinos in a file
    with open(f'output/{name}.json', 'w') as f:
        json.dump(vinos, f, indent=2)




if __name__ == '__main__':
    # Run the scraper
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--country', default='US')
    parser.add_argument('--state', default='CA')
    parser.add_argument('--minPrice', type=float)
    parser.add_argument('--maxPrice', type=float)
    parser.add_argument('--noPriceIncluded', action='store_true')  
    parser.add_argument('--minRatings', type=int)
    parser.add_argument('--maxRatings', type=int)
    parser.add_argument('--minAverage', type=float)
    parser.add_argument('--maxAverage', type=float)

    args = parser.parse_args()

    driver = get_driver()

    vivino_search(driver, args.name, args.country, args.state, args.minPrice, args.maxPrice, args.noPriceIncluded,
        args.minRatings, args.maxRatings, args.minAverage, args.maxAverage)

