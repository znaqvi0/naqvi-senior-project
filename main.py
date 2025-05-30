import datetime
import os

import cv2

from config import EMBEDDING_JSON_PATH, PHOTO_DIR_PATH
from embedding_generator import load_embedding_dict
from recognition import compare_image_to_embeddings
from webcam_reader import init_webcam, close_webcam

embedding_dict = load_embedding_dict(EMBEDDING_JSON_PATH)

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

        compare_image_to_embeddings(save_dest, embedding_dict)

close_webcam(camera)