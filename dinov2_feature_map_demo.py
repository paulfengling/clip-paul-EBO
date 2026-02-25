import numpy as np
from PIL import Image
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torch
from torchvision import  transforms


def feature_demo(image_file, img_size=224):
    assert img_size % 14 == 0, "The image size must be exactly divisible by 14"

    dinov2_vits14 = torch.hub.load('facebookresearch', "dinov2_vits14_reg", source='local')
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_vits14.eval()

    tf = transforms.Compose([
        transforms.Resize(img_size + int(img_size * 0.01) * 10),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    patch_h = patch_w = img_size // 14
    img_cnt = 1

    images = tf(Image.open(image_file))
    images = torch.FloatTensor(images).unsqueeze(0).cuda()

    with torch.no_grad():
        embeddings = dinov2_vits14.forward_features(images)
        x_norm_patchtokens = embeddings["x_norm_patchtokens"].cpu().numpy()
        print(x_norm_patchtokens.shape)

    x_norm_1616_patches = x_norm_patchtokens.reshape(img_cnt * patch_h * patch_w, -1)

    fg_pca = PCA(n_components=1)
    fg_pca_images = fg_pca.fit_transform(x_norm_1616_patches)
    fg_pca_images = minmax_scale(fg_pca_images)
    fg_pca_images = fg_pca_images.reshape(img_cnt, patch_h * patch_w)
    print(fg_pca_images.shape, fg_pca_images[0])

    feature_map = np.array(fg_pca_images[0]).reshape(32,32)
    gray_map = cv.resize(feature_map, (448, 448))
    gray_mask = (gray_map * 255).astype(np.uint8)
    ret, bin_mask = cv.threshold(gray_mask, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    colored_map = cv.applyColorMap(gray_mask, cv.COLORMAP_JET)
    src = cv.imread("tuzi.jpg")
    cv.imshow("input image", src)
    cv.imshow("DINOv3 feature mask", bin_mask)
    cv.imshow("DINOv3 feature map", colored_map)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    save_fg_mask = True
    img_size = 448
    feature_demo("tuzi.jpg", img_size)