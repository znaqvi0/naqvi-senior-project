import json
import os
import random

import torch
from PIL import Image

from directory_scraper import FACE_DIR
from facenet_pytorch import MTCNN, InceptionResnetV1

# load embeddings from json
with open('embeddings.json', 'r') as f:
    embedding_dict = json.load(f)

# convert each embedding into a pytorch tensor
for person in embedding_dict:
    embedding_dict[person] = torch.tensor(embedding_dict[person])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40).to(device)

input_image_path = random.choice(os.listdir(FACE_DIR))  # test path
input_image = Image.open(os.path.join(FACE_DIR, input_image_path)).convert('RGB')

# process input image
input_image_cropped = mtcnn(input_image)
if input_image_cropped is None:
    print(f'no face detected in {input_image_path}')
input_embedding = resnet(input_image_cropped.unsqueeze(0))

# search embeddings for a match
for person in embedding_dict:
    # compare the cosine of the angle between the two embeddings
    similarity = torch.nn.functional.cosine_similarity(input_embedding, embedding_dict[person])
    if similarity > 0.8:
        print(f'{input_image_path} has a similarity of {similarity[0]} to {person}')