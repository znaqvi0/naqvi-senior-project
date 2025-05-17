import os
import time
from urllib.parse import urljoin
from selenium import webdriver
from bs4 import BeautifulSoup
import requests


directory_url = 'https://www.mcdonogh.org/directory?Grade=12&DoSearch=1&StudSearch=1'
driver = webdriver.Chrome()
driver.get(directory_url)  # opens login page

print('press ENTER after logging in')
input()

driver.get(directory_url)
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()


os.makedirs('mcd_faces', exist_ok=True)
# print(soup.prettify())
divs = soup.find_all('div', class_='bg')

for div in divs:
    img_url = div.get('style', '').split('\'')[1]  # split by single quotes surrounding relative image path
    # print(img_url)
    if img_url:
        full_url = urljoin('https://www.mcdonogh.org', img_url)
        filepath = os.path.join('mcd_faces', os.path.basename(full_url))

        # skip if the face is already downloaded
        if img_url.split('/')[-1] in os.listdir('mcd_faces'):
            print(f'already downloaded {full_url}')
            continue

        try:
            data = requests.get(full_url).content
            with open(filepath, 'wb') as f:
                f.write(data)
            print(f'downloaded {full_url}')
        except Exception as e:
            print(f'failed to download {full_url}')
    time.sleep(0.1)  # avoid attacking the servers