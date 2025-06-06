import datetime
import os
import sys

import cv2
from PIL import Image

from config import PHOTO_DIR_PATH

def take_photo(save_destination):
    # initialize webcam
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('webcam')

    if not camera.isOpened():
        print('failed to open camera')
        return None

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
            # save frame to save_destination
            cv2.imwrite(save_destination, frame)
            print(f'photo saved as {save_destination}')
            break

    # release the camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

    if not ret:
        return None
    return Image.open(save_destination).convert('RGB')


def save_with_timestamp():
    os.makedirs(PHOTO_DIR_PATH, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dest = os.path.join(PHOTO_DIR_PATH, f'{timestamp}.jpg')
    return take_photo(save_dest), save_dest


def init_webcam(window_title):
    camera = cv2.VideoCapture(0)
    cv2.namedWindow(window_title)

    if not camera.isOpened():
        print('failed to open camera')
        sys.exit()

    return camera


def close_webcam(camera):
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    save_with_timestamp()