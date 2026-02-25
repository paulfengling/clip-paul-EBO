# Integration with CLIP model
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import clip
from PIL import Image
from faiss_manager import add_to_index, search_similar, reset_index
import cv2 as cv

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_path):
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((224, 224))
    processed_image = preprocess(pil_image)  # Adds resize, crop, normalization (0, -1) range as per CLIP model needs.
    image_tensor = processed_image.unsqueeze(0)  # Adds a batch dimension to make it 4 dimensional
    image_tensor = image_tensor.to(device)  # Move image tensor to GPU or CPU

    with torch.no_grad():
        embedding_tensor = model.encode_image(image_tensor)  # returns 512 dimensional vector embedding
        # no_grad() saved memory and computations by stating that we dont need gradient calculations

    embedding_numpy = embedding_tensor.cpu().numpy()  # Moves back to cpu and return numpy array
    print(embedding_numpy.shape)
    return embedding_numpy


def get_text_embedding(text):
    text_tensor = clip.tokenize([text]).to(device)  # Tokenizes the text and moves to GPU or CPU
    with torch.no_grad():
        embedding_tensor = model.encode_text(text_tensor)

    embedding_numpy = embedding_tensor.cpu().numpy()  # Moves back to cpu and return numpy array
    return embedding_numpy


if __name__ == "__main__":
    root_dir = "./similary_imgs"
    files = os.listdir(root_dir)
    reset_index()
    for file in files:
        print("file path-->>", os.path.join(root_dir, file))
        embedding = get_image_embedding(os.path.join(root_dir, file))
        add_to_index(embedding, os.path.join(root_dir, file))

    text_query = "good"
    image_path = "./broken_cont.png"
    src = cv.imread(image_path)
    embedding = get_image_embedding(image_path)
    txt_embedding = get_text_embedding(text_query)
    similar_images, distances = search_similar(embedding, 4)
    min_score = 1000
    for i in range(len(similar_images)):
        dist_score = distances[0][i]
        if min_score > dist_score:
            min_score = dist_score
            print("min distance: ", min_score, similar_images[i])
            image = cv.imread(similar_images[i])
    print("异常得分：", min_score)
    if min_score > 4.5:
        cv.putText(src, "NG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv.putText(src, "OK", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv.imshow("detection result", src)
    cv.imwrite("D:/detect_ad_bad.png", src)
    cv.waitKey(0)
    cv.destroyAllWindows()



