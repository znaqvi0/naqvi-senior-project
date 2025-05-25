import os
import random

from directory_scraper import FACE_DIR
from embedding_generator import load_embedding_dict
from recognition_utils import search_for_match

embedding_dict = load_embedding_dict()
cosine_threshold = 0.85
euclidean_threshold = 0.55

input_image_path = random.choice(os.listdir(FACE_DIR))  # test path
input_image_path = os.path.join(FACE_DIR, input_image_path)

search_for_match(input_image_path, embedding_dict, cosine_threshold=cosine_threshold, euclidean_threshold=euclidean_threshold, ignore_filenames=True)
