import cv2 as cv
import torch
import torchvision
from torchvision import transforms
import numpy as np

model = torchvision.models.vit_b_16(pretrained=True).eval().cuda()
tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

with open('imagenet_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
src = cv.imread("D:/images/avar.jpg") # aeroplane.jpg
image = cv.resize(src, (224, 224))
image = np.float32(image) / 255.0
image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
image = image.transpose((2, 0, 1))
input_x = torch.from_numpy(image).unsqueeze(0)
print(input_x.size())
pred = model(input_x.cuda())
pred_index = torch.argmax(pred, 1).cpu().detach().numpy()
print(pred_index)
print("current predict class name : %s"%labels[pred_index[0]])
cv.putText(src, labels[pred_index[0]], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv.imshow("input", src)
cv.imwrite("D:/vit_demo.png", src)
cv.waitKey(0)
cv.destroyAllWindows()