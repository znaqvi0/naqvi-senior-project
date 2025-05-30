import os

import torch
from PIL import Image

from config import FACE_DIR_PATH, COSINE_THRESHOLD, EUCLIDEAN_THRESHOLD, DISTANCE_METRIC
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40).to(device)


def find_similarities(input_image_path, embedding_dict, ignore_filenames=False, do_print=True):
    # determine which metrics to use
    compare_euclidean, compare_cosine = False, False
    if DISTANCE_METRIC == 'euclidean' or DISTANCE_METRIC == 'both':
        compare_euclidean = True
    if DISTANCE_METRIC == 'cosine' or DISTANCE_METRIC == 'both':
        compare_cosine = True
    if not compare_cosine and not compare_euclidean:
        print('invalid distance metric; defaulting to both...')
        compare_euclidean, compare_cosine = True, True

    cosine_similarities: dict[str, torch.Tensor] = {}
    euclidean_distances: dict[str, torch.Tensor] = {}

    input_image = Image.open(input_image_path).convert('RGB')
    # process image
    input_image_cropped = mtcnn(input_image)
    if input_image_cropped is None:
        print(f'no face detected in {input_image_path}')
        return None, None
    input_embedding = resnet(input_image_cropped.unsqueeze(0))

    # search embeddings for a match
    for person in embedding_dict:
        if not ignore_filenames and person == os.path.basename(input_image_path):
            continue

        if compare_cosine:
            # compare the cosine of the angle between the two embeddings
            similarity = torch.nn.functional.cosine_similarity(input_embedding, embedding_dict[person])
            cosine_similarities[person] = similarity
            if similarity > COSINE_THRESHOLD and do_print:
                print(f'{input_image_path} has a similarity of {similarity[0]} to {person}')

        if compare_euclidean:
            # find the Euclidean distance between the two embeddings
            euclidean_distance = torch.linalg.vector_norm(embedding_dict[person] - input_embedding)
            euclidean_distances[person] = euclidean_distance
            if euclidean_distance < EUCLIDEAN_THRESHOLD and do_print:
                print(f'{input_image_path} has a distance of {euclidean_distance} to {person}')

    return cosine_similarities, euclidean_distances


def most_likely_match(cosine_similarities, euclidean_distances):
    data = {
        'cosine': {'name': None, 'value': None},
        'euclidean': {'name': None, 'value': None}
    }

    if cosine_similarities:
        max_key = max(cosine_similarities, key=cosine_similarities.get)
        data['cosine']['name'] = max_key
        data['cosine']['value'] = cosine_similarities[max_key][0]

    if euclidean_distances:
        min_key = min(euclidean_distances, key=euclidean_distances.get)
        data['euclidean']['name'] = min_key
        data['euclidean']['value'] = euclidean_distances[min_key]

    return data


def compare_image_to_embeddings(image_path, embedding_dict, id_name_dict, preprocess_name_function):
    cosine_similarities, euclidean_distances = find_similarities(image_path, embedding_dict, do_print=False,
                                                                 ignore_filenames=True)
    candidate_data = most_likely_match(cosine_similarities, euclidean_distances)

    cosine_data = candidate_data['cosine']
    euclidean_data = candidate_data['euclidean']

    # convert image names to person names
    if id_name_dict is not None:
        if cosine_data['name']:
            cosine_data['name'] = id_name_dict[preprocess_name_function(candidate_data['cosine']['name'])]
        if euclidean_data['name']:
            euclidean_data['name'] = id_name_dict[preprocess_name_function(candidate_data['euclidean']['name'])]

    if cosine_data['name']:
        similarity = cosine_data['value']
        match_type = 'probable' if similarity > COSINE_THRESHOLD else 'improbable'
        print(f'{image_path} has a similarity of {similarity:0.4f} to {cosine_data['name']}; {match_type} match')

    if euclidean_data['name']:
        distance = euclidean_data['value']
        match_type = 'probable' if distance < EUCLIDEAN_THRESHOLD else 'improbable'
        print(f'{image_path} has a distance of {distance:0.4f} to {euclidean_data['name']}; {match_type} match')


# compare all people; if anyone has more than 1 match, the threshold should be adjusted
def compare_all(embedding_dict):
    for input_person in embedding_dict:
        input_image_path = os.path.join(FACE_DIR_PATH, input_person)  # test path
        find_similarities(input_image_path, embedding_dict, ignore_filenames=False)
