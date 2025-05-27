import json
import os
import time

import torch
from PIL import Image

from config import EMBEDDING_JSON_PATH, FACE_DIR_PATH
from recognition_utils import resnet, mtcnn

def generate_embeddings(face_path, output_path):
    embedding_dict = {}
    # test speed of pretrained models
    start_time = time.perf_counter()

    for img_path in os.listdir(face_path):
        img = Image.open(os.path.join(face_path, img_path)).convert('RGB')

        img_cropped = mtcnn(img)

        if img_cropped is None:
            print(f'no face detected in {img_path}')
            continue

        embedding = resnet(img_cropped.unsqueeze(0))
        embedding_dict[img_path] = embedding.tolist()

    end_time = time.perf_counter()
    print(f'total time: {end_time - start_time}')
    print(f'average time: {(end_time - start_time) / len(os.listdir(face_path))}')

    with open(output_path, 'w') as f:
        json.dump(embedding_dict, f)


def load_embedding_dict(filepath):
    # load embeddings from json
    with open(filepath, 'r') as f:
        embedding_dict = json.load(f)

    # convert each embedding into a pytorch tensor
    for person in embedding_dict:
        embedding_dict[person] = torch.tensor(embedding_dict[person])

    return embedding_dict


if __name__ == '__main__':
    generate_embeddings(FACE_DIR_PATH, EMBEDDING_JSON_PATH)