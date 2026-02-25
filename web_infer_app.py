import gradio as gr
import os
import numpy as np
import logging
from glob import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2 as cv

from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from forward_utils import (
    get_adapted_single_class_text_embedding,
    calculate_similarity_map
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transform_x = transforms.Compose(
    [
        transforms.Resize((518, 518), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

setup_seed(111)
class_name = "object"
save_path = "ckpt/baseline"
# check save_path and setting logger
os.makedirs(save_path, exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(save_path, "opencv_xue_tang_test.log"),
    encoding="utf-8",
    level=logging.INFO,
)
# logger.info("args: %s", vars(args))
# set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# ========================================================
# load model
# set up model for testing
clip_model = create_model(
    model_name="ViT-L-14-336",
    img_size=518,
    device=device,
    pretrained="openai",
    require_pretrained=True,
)
clip_model.eval()
model = AdaptedCLIP(
    clip_model=clip_model,
    text_adapt_weight=0.1,
    image_adapt_weight=0.1,
    text_adapt_until=3,
    image_adapt_until=6,
    relu=False,
).to(device)
model.eval()
# load checkpoints if exists
text_file = glob(save_path + "/text_adapter.pth")
assert len(text_file) >= 0, "text adapter checkpoint not found"
if len(text_file) > 0:
    checkpoint = torch.load(text_file[0])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    adapt_text = True
else:
    adapt_text = False

print("loaded text adapter end!!!")
files = sorted(glob(save_path + "/image_adapter_20.pth"))
assert len(files) > 0, "image adapter checkpoint not found"
print("start to load image adapter->>", files[0])
checkpoint = torch.load(files[0])
model.image_adapter.load_state_dict(checkpoint["image_adapter"])
test_epoch = checkpoint["epoch"]
logger.info("-----------------------------------------------")
logger.info("load model from epoch %d", test_epoch)
logger.info("-----------------------------------------------")
print("loaded image adapter successfully->>", files[0])
# ========================================================

with torch.no_grad():
    if adapt_text:
        class_text_embeddings = get_adapted_single_class_text_embedding(
            model, "opencvxuetang", "object", device
        )
    else:
        class_text_embeddings = get_adapted_single_class_text_embedding(
            clip_model, "opencvxuetang", "object", device
        )

def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    device: str,
    img_size: int,
    image_path: str
):
    preds = []
    preds_image = []

    # get text
    image = Image.open(image_path).convert("RGB")
    image = transform_x(image).to(device)
    print("image shape-->>", image.shape)
    epoch_text_feature = class_text_embeddings

    # forward image
    patch_features, det_feature = model(image.unsqueeze(0))

    # calculate similarity and get prediction
    pred = det_feature @ epoch_text_feature
    pred = (pred[:, 1] + 1) / 2
    preds_image.append(pred.cpu().numpy())
    patch_preds = []
    for f in patch_features:
        # f: bs,patch_num,768
        patch_pred = calculate_similarity_map(
            f, epoch_text_feature, img_size, test=True, domain="Industrial"
        )
        patch_preds.append(patch_pred)
    patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
    preds.append(patch_preds)

    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return preds, preds_image

def image_infer(image, object_name):
    # ========================================================
    # testing
    image_file = "data/opencvxuetang/unknown/my_ok_ng_test.png"
    image.save(image_file)
    print("input object name : ", object_name)
    with torch.no_grad():
        pred_masks, preds_image = get_predictions(
            model=model,
            class_text_embeddings=class_text_embeddings,
            device=device,
            img_size=518,
            image_path = image_file
        )
        preds = pred_masks[0]
        print("preds: ", preds)
        print("preds_image: ", preds_image)
        if preds.max() != 1:
            pixel_preds = (preds - preds.min()) / (
                    preds.max() - preds.min()
            )
            gray_mask = (pixel_preds * 255).astype(np.uint8)
            print("prediction score -->> ", preds_image[0])

            colored_img = cv.applyColorMap(gray_mask, cv.COLORMAP_JET)
            ch, cw, cc = colored_img.shape
            print("colored_img-->> ", colored_img.shape)
            src = cv.imread(image_file, cv.IMREAD_COLOR)
            resized = cv.resize(src, (ch, cw))
            anomaly_mask = cv.addWeighted(resized, 0.8, colored_img, 0.6, 0)
            amap = Image.fromarray(cv.cvtColor(colored_img, cv.COLOR_BGR2RGB))
            amask = Image.fromarray(cv.cvtColor(anomaly_mask, cv.COLOR_BGR2RGB))
            return amask,amap,f"{preds_image[0]:.5f}"

# Gradio interface layout

demo = gr.Interface(
    fn=image_infer,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Class Name")
    ],
    outputs=[
        gr.Image(type="pil", label="anomaly Mask"),
        gr.Image(type="pil", label="anomaly Map"),
        gr.Textbox(label="Anomaly Score"),
    ],
    title="OpenCV学堂 -- 零样本异常检测大模型",
    description="Upload an image, enter class name, and select pre-trained datasets to do zero-shot anomaly detection"
)
# Launch the demo
demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=10002)