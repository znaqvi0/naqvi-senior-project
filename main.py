import os
import random

from config import FACE_DIR_PATH, EMBEDDING_JSON_PATH
from embedding_generator import load_embedding_dict
from recognition_utils import search_for_match
from webcam_reader import save_with_timestamp

embedding_dict = load_embedding_dict(EMBEDDING_JSON_PATH)

# input_image_path = random.choice(os.listdir(FACE_DIR_PATH))  # test path
# input_image_path = os.path.join(FACE_DIR_PATH, input_image_path)
_, input_image_path = save_with_timestamp()

search_for_match(input_image_path, embedding_dict, ignore_filenames=True)
