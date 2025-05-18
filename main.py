import os
import time

import torch
from PIL import Image

from directory_scraper import FACE_DIR
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # pretrained embedding generator
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40).to(device)  # pretrained face detector

# test speed of pretrained models
start_time = time.perf_counter()

for img_path in os.listdir(FACE_DIR):
    img = Image.open(os.path.join(FACE_DIR, img_path))

    img_cropped = mtcnn(img)
    embedding = resnet(img_cropped.unsqueeze(0))

end_time = time.perf_counter()
print(f'total time: {end_time - start_time}')
print(f'average time: {(end_time - start_time) / len(os.listdir(FACE_DIR))}')