import cv2 as cv
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import TextStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

model_dir = "D:/LLMs/qwen2.5_3b/INT4"
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
model = OVModelForVisualCausalLM.from_pretrained(model_dir, device="CPU")
if processor.chat_template is None:
    tok = AutoTokenizer.from_pretrained("D:/LLMs/qwen2.5_3b")
    processor.chat_template = tok.chat_template


image = cv.imread("D:/1250.jpg")
question = "读取图像中的订单编号"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "D:/1250.jpg",
            },
            {"type": "text", "text": question},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("Question:")
print(question)
print("Answer:")

generated_ids = model.generate(**inputs, max_new_tokens=400,
        streamer=TextStreamer(processor.tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=True))
