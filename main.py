import datetime
import json
import os

import cv2

from config import EMBEDDING_JSON_PATH, PHOTO_DIR_PATH, PEOPLE_DATA_JSON_PATH
from embedding_generator import load_embedding_dict
from recognition import compare_image_to_embeddings
from webcam_reader import init_webcam, close_webcam

embedding_dict = load_embedding_dict(EMBEDDING_JSON_PATH)

use_names = os.path.exists(PEOPLE_DATA_JSON_PATH)

# try loading {id, name} json file
id_name_dict = None
if use_names:
    with open(PEOPLE_DATA_JSON_PATH, 'r') as f:
        id_name_dict = json.load(f)
else:
    print(f"{PEOPLE_DATA_JSON_PATH} does not exist. Try running directory_scraper.py or creating your own json file of {{UserID, name}} key-value pairs. Resorting to image names for comparison output...")

camera = init_webcam('webcam')

# read frame from webcam
while True:
    ret, frame = camera.read()
    cv2.imshow('webcam', frame)

    if not ret:
        print('photo capture failed')
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:  # esc pressed
        print("esc pressed, closing...")
        break
    elif k % 256 == 32:  # space pressed
        # save frame
        os.makedirs(PHOTO_DIR_PATH, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dest = os.path.join(PHOTO_DIR_PATH, f'{timestamp}.jpg')
        cv2.imwrite(save_dest, frame)
        print(f'photo saved as {save_dest}')

        compare_image_to_embeddings(save_dest, embedding_dict, id_name_dict, lambda name: name.split('.')[0])

close_webcam(camera)