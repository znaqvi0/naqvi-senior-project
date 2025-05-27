import os
import time
from urllib.parse import urljoin
from selenium import webdriver
from bs4 import BeautifulSoup
import requests

from config import FACE_DIR_PATH

if __name__ == '__main__':
    login_url = 'https://www.mcdonogh.org/login'
    directory_urls = [f'https://www.mcdonogh.org/directory?Grade={grade}&DoSearch=1&StudSearch=1' for grade in range(9, 13)]
    directory_urls.append('https://www.mcdonogh.org/about/contact/faculty-staff-directory?browse')

    driver = webdriver.Chrome()
    driver.get(login_url)  # opens login page

    print('press ENTER after logging in')
    input()

    soups = []
    # parse HTML for each URL
    for directory_url in directory_urls:
        driver.get(directory_url)
        soups.append(BeautifulSoup(driver.page_source, 'html.parser'))

    driver.quit()

    os.makedirs(FACE_DIR_PATH, exist_ok=True)
    # print(soup.prettify())
    divs = []
    for soup in soups:
        divs.extend(soup.find_all('div', class_='bg'))  # each image looks like div class="bg" in the HTML

    for div in divs:
        # full div: <div class="bg" style="background-image: url('/images/students/{year}/{person ID}.jpg'); background-position: center center"></div>
        img_url = div.get('style', '').split('\'')[1]  # isolate relative image URL using single quotes
        if not img_url:
            continue

        full_url = urljoin('https://www.mcdonogh.org', img_url)
        filepath = os.path.join(FACE_DIR_PATH, os.path.basename(full_url))

        img_name = img_url.split('/')[-1]

        # skip if the image is not a face (not a numeric ID)
        if not img_name.split('.')[0].isnumeric():
            print(f'skipping {img_name}')
            continue

        # skip if the face is already downloaded
        if img_name in os.listdir(FACE_DIR_PATH):
            print(f'already downloaded {os.path.basename(full_url)}')
            continue

        # download the image
        try:
            data = requests.get(full_url).content
            with open(filepath, 'wb') as f:  # wb: write in binary mode to preserve image data
                f.write(data)
            print(f'downloaded {full_url}')
        except Exception as e:
            print(f'failed to download {full_url}')

        time.sleep(0.1)  # avoid attacking the servers