# Face Recognition Project README

This project involves several scripts to perform face recognition. Follow the steps below to set up and run the project.

---
## Configuration

Before running any scripts, you need to configure the project settings:

1.  Open the `config.py` file.
2.  Modify the following parameters as needed:

    * `EMBEDDING_JSON_PATH`: **string**
        * Path to store the 512-dimensional face embeddings in a JSON file.
        * Each key-value pair looks like `{image_name: str, embedding: list}`.
        * Each **embedding** (the value) is converted to `torch.tensor` when loaded.
    * `PEOPLE_DATA_JSON_PATH`: **string**
        * Path to a JSON file mapping IDs (or image prefixes) to names.
        * Format: `{ID_or_image_prefix: str, name: str}`. This allows the program to output a person's name.
    * `FACE_DIR_PATH`: **string**
        * Path to the directory where an image of each person is stored.
    * `PHOTO_DIR_PATH`: **string**
        * Path to the directory where images captured by the webcam will be stored.
    * `COSINE_THRESHOLD`: **float**
        * Threshold for cosine similarity. Embeddings with similarity above this value are considered a 'match'.
    * `EUCLIDEAN_THRESHOLD`: **float**
        * Threshold for Euclidean distance. Embeddings with distance below this value are considered a 'match'.
    * `DISTANCE_METRIC`: **string**
        * The distance metric to use for comparison.
        * Options: `'cosine'`, `'euclidean'`, or `'both'`.

---
## Running the Project

Follow these steps in order to get the desired results:

1.  **Generate Face Directory**
    * Run `directory_scraper.py` to generate a directory containing an image of each student and faculty member at McDonogh.
    * Alternatively, you can create your own directory and set its path in `FACE_DIR_PATH` in `config.py`.
    * **Note:** If the face directory (specified by `FACE_DIR_PATH`) already exists and is up-to-date, you do not need to run `directory_scraper.py` unless you expect the source data (e.g., school directory photos) to have changed.

2.  **Generate Face Embeddings**
    * Run `embedding_generator.py`.
    * This script will generate a JSON file containing the face embeddings for each person in the directory specified by `FACE_DIR_PATH` in `config.py`.
    * **Note:** If the embedding JSON file (specified by `EMBEDDING_JSON_PATH`) already exists and the face image directory (`FACE_DIR_PATH`) has not been updated, you do not need to run this script again.

3.  **Run Main Application**
    * Once the configuration is set and the embeddings are generated, run `main.py`.

---
