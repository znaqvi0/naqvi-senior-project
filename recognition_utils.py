import os

import torch
from PIL import Image

from config import FACE_DIR_PATH, COSINE_THRESHOLD, EUCLIDEAN_THRESHOLD, DISTANCE_METRIC
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40).to(device)


def search_for_match(input_image_path, embedding_dict, ignore_filenames=False):
    # determine which metrics to use
    compare_euclidean, compare_cosine = False, False
    if DISTANCE_METRIC == 'euclidean' or DISTANCE_METRIC == 'both':
        compare_euclidean = True
    if DISTANCE_METRIC == 'cosine' or DISTANCE_METRIC == 'both':
        compare_cosine = True
    if not compare_cosine and not compare_euclidean:
        print('invalid distance metric; defaulting to both...')
        compare_euclidean, compare_cosine = True, True

    input_image = Image.open(input_image_path).convert('RGB')
    # process image
    input_image_cropped = mtcnn(input_image)
    if input_image_cropped is None:
        print(f'no face detected in {input_image_path}')
        return
    input_embedding = resnet(input_image_cropped.unsqueeze(0))

    # search embeddings for a match
    for person in embedding_dict:
        if not ignore_filenames and person == os.path.basename(input_image_path):
            continue

        if compare_cosine:
            # compare the cosine of the angle between the two embeddings
            similarity = torch.nn.functional.cosine_similarity(input_embedding, embedding_dict[person])
            if similarity > COSINE_THRESHOLD:
                print(f'{input_image_path} has a similarity of {similarity[0]} to {person}')

        if compare_euclidean:
            # find the Euclidean distance between the two embeddings
            euclidean_distance = torch.linalg.vector_norm(embedding_dict[person] - input_embedding)
            if euclidean_distance < EUCLIDEAN_THRESHOLD:
                print(f'{input_image_path} has a distance of {euclidean_distance} to {person}')


# compare all people; if anyone has more than 1 match, the threshold should be adjusted
def compare_all(embedding_dict):
    for input_person in embedding_dict:
        input_image_path = os.path.join(FACE_DIR_PATH, input_person)  # test path
        search_for_match(input_image_path, embedding_dict, ignore_filenames=False)
