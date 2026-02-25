import os
import cv2 as cv
import torch
from PIL import Image
import open_clip
import numpy as np
label_names = ["girl", "dog", "cat", "bird"]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def searchText():
    image = preprocess(Image.open("text_image_ds/glod_dog.jpg")).unsqueeze(0)
    text = tokenizer(["a girl or girls", "a dog or dogs", "a cat or cats", "a bird or birds"])

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        print("label probs: ", text_probs)
    max_idx = np.argmax(text_probs.cpu().numpy(), 1)[0]
    print(max_idx)
    print("predicted Label: ", label_names[max_idx])

def searchImage():
    matched_im_file = []
    text = tokenizer(["a girl or girls", "a dog or dogs", "a cat or cats", "a bird or birds"])
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    files = os.listdir("./text_image_ds")
    for f in files:
        im_file = os.path.join("./text_image_ds/", f)
        print(im_file)

        image = preprocess(Image.open(im_file)).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            print("label probs: ", text_probs)
        max_idx = np.argmax(text_probs.cpu().numpy(), 1)[0]
        if max_idx == 3:
            print(max_idx)
            print("predicted Label: ", label_names[max_idx])
            src = cv.imread(im_file)
            cv.imshow("matched search result", src)
            cv.waitKey(0)

if __name__ == "__main__":
    searchImage()