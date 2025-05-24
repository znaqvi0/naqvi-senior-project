import cv2
from PIL import Image


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

    return Image.open(save_destination)
