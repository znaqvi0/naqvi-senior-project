import os

import torch
from PIL import Image

from directory_scraper import FACE_DIR
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40).to(device)

def search_for_match(input_image_path, embedding_dict, cosine_threshold, euclidean_threshold, ignore_filenames=False):
    input_image = Image.open(input_image_path).convert('RGB')

    input_image_cropped = mtcnn(input_image)
    if input_image_cropped is None:
        print(f'no face detected in {input_image_path}')
        return

    input_embedding = resnet(input_image_cropped.unsqueeze(0))

    # search embeddings for a match
    for person in embedding_dict:
        if not ignore_filenames and person == os.path.basename(input_image_path):
            continue
        # compare the cosine of the angle between the two embeddings
        similarity = torch.nn.functional.cosine_similarity(input_embedding, embedding_dict[person])
        if similarity > cosine_threshold:
            print(f'{input_image_path} has a similarity of {similarity[0]} to {person}')

        # find the Euclidean distance between the two embeddings
        euclidean_distance = torch.linalg.vector_norm(embedding_dict[person] - input_embedding)
        if euclidean_distance < euclidean_threshold:
            print(f'{input_image_path} has a distance of {euclidean_distance} to {person}')


# compare all people; if anyone has more than 1 match, the threshold should be adjusted
def compare_all(embedding_dict, cosine_threshold, euclidean_threshold):
    for input_person in embedding_dict:
        input_image_path = os.path.join(FACE_DIR, input_person)  # test path
        search_for_match(input_image_path, embedding_dict, cosine_threshold, euclidean_threshold)
