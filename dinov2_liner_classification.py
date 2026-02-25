import numpy as np
import torch
import cv2 as cv
from PIL import Image
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l, get_tokenizer
from torchvision import  transforms

tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

img_pil = Image.open("example.png").convert("RGB")
image_tensor = tf(img_pil).unsqueeze(0).cuda()
print(image_tensor.size())

model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l().cuda()
tokenizer = get_tokenizer()

texts = ["photo of dogs", "photo of a chair", "photo of a bowl", "photo of a tupperware"]
class_names = ["dog", "chair", "bowl", "tupperware"]
tokenized_texts_tensor = tokenizer.tokenize(texts).cuda()
with torch.autocast('cuda', dtype=torch.float):
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokenized_texts_tensor)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (
    text_features.cpu().float().numpy() @ image_features.cpu().float().numpy().T
)
idx = np.argmax(similarity)
print("predict label: ", class_names[idx], ", max score: ", similarity[idx][0])
src = cv.imread("example.png")
cv.putText(src, class_names[idx], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv.imshow("dinov3-text predict", src)
cv.waitKey(0)
cv.destroyAllWindows()