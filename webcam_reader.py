import datetime
import os

import cv2
from PIL import Image

PHOTO_DIR = 'captures'

def take_photo(save_destination):
    # initialize webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print('failed to open camera')
        return None

    # read frame from webcam
    ret, frame = camera.read()

    if ret:
        # save frame to save_destination
        cv2.imwrite(save_destination, frame)
        print(f'photo saved as {save_destination}')
    else:
        print('photo capture failed')

    # release the camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

    if not ret:
        return None
    return Image.open(save_destination).convert('RGB')


def save_with_timestamp():
    os.makedirs(PHOTO_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dest = os.path.join(PHOTO_DIR, f'{timestamp}.jpg')
    return take_photo(save_dest), save_dest


if __name__ == '__main__':
    save_with_timestamp()