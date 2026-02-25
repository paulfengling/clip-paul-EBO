from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2 as cv
import os
import torch
from faiss_manager import add_to_index, search_similar, reset_index

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
print("dinov2 model loaded~~~~~")

def get_image_embedding(image_path):
    image1 = Image.open(image_path)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)
    feature_numpy = image_features1.cpu().numpy()  # Moves back to cpu and return numpy array
    print(feature_numpy.shape)
    return feature_numpy


if __name__ == "__main__":
    root_dir = "./similary_imgs"
    files = os.listdir(root_dir)
    reset_index()
    for file in files:
        print("file path-->>", os.path.join(root_dir, file))
        embedding = get_image_embedding(os.path.join(root_dir, file))
        add_to_index(embedding, os.path.join(root_dir, file))

    text_query = "good"
    image_path = "./broken_small.png"
    src = cv.imread(image_path)
    embedding = get_image_embedding(image_path)
    similar_images, distances = search_similar(embedding, 2)
    min_score = 1000
    for i in range(len(similar_images)):
        dist_score = distances[0][i]
        if min_score > dist_score:
            min_score = dist_score
            print("min distance: ", min_score, similar_images[i])
            image = cv.imread(similar_images[i])
    print("异常得分：", min_score)
    if min_score > 60:
        cv.putText(src, "NG", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv.putText(src, "OK", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv.imshow("detection result", src)
    cv.imwrite("D:/detect_ad_bad.png", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

