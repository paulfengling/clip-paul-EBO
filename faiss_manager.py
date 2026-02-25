# Integration with FAISS

import faiss
import pickle
import os
TOP_K = 3
IMAGE_DIR = "./similary_imgs"
INDEX_DIR = "faiss_index"

FIASS_INDEX_FILE = os.path.join(INDEX_DIR, "index_file.index")
IMAGE_PATH_FILE = os.path.join(INDEX_DIR, "image_paths.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# embedding_dim = 512 # Clip ViT-B/32 outputs 512 dimensional vectors
embedding_dim = 768 # dinov2-base outputs 768 dimensional vectors
index_file = FIASS_INDEX_FILE
paths_file = IMAGE_PATH_FILE

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    index = faiss.IndexFlatL2(embedding_dim) # create empty index using L2 distance (Euclidean)

if os.path.exists(paths_file):
    with open(paths_file, "rb") as file:
        image_paths = pickle.load(file)
else:
    image_paths = []

print(f"Loaded {len(image_paths)} images from the index.")

# Function to save the index and image paths to current state.
def save_index():
    faiss.write_index(index, index_file)
    with open(paths_file, "wb") as file:
        pickle.dump(image_paths, file)

# Function to add image embedding to the index
def add_to_index(embedding, image_path):
    global image_paths
    index.add(embedding)
    image_paths.append(image_path)
    save_index()

# Function to search for similar images
def search_similar(embedding, k=3):
    if index.ntotal == 0:
        return []

    distance, indices = index.search(embedding, k)
    print(distance)
    similar_images = []
    for i in indices[0]:
        if i == -1:
            break
        similar_images.append(image_paths[i])
        print("matched file path", image_paths[i])

    return similar_images, distance

# Function to reset the index
def reset_index():
    # Delete files and images
    if os.path.exists(index_file):
        os.remove(index_file)

    global index, image_paths
    image_paths = []
    index = faiss.IndexFlatL2(embedding_dim)