import json
import os
import time

import torch
from PIL import Image

from directory_scraper import FACE_DIR
from recognition_utils import resnet, mtcnn

def generate_embeddings():
    embedding_dict = {}
    # test speed of pretrained models
    start_time = time.perf_counter()

    for img_path in os.listdir(FACE_DIR):
        img = Image.open(os.path.join(FACE_DIR, img_path)).convert('RGB')

        img_cropped = mtcnn(img)

        if img_cropped is None:
            print(f'no face detected in {img_path}')
            continue

        embedding = resnet(img_cropped.unsqueeze(0))
        embedding_dict[img_path] = embedding.tolist()

    end_time = time.perf_counter()
    print(f'total time: {end_time - start_time}')
    print(f'average time: {(end_time - start_time) / len(os.listdir(FACE_DIR))}')

    with open('embeddings.json', 'w') as f:
        json.dump(embedding_dict, f)


def load_embedding_dict():
    # load embeddings from json
    with open('embeddings.json', 'r') as f:
        embedding_dict = json.load(f)

    # convert each embedding into a pytorch tensor
    for person in embedding_dict:
        embedding_dict[person] = torch.tensor(embedding_dict[person])

    return embedding_dict


if __name__ == '__main__':
    generate_embeddings()