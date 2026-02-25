
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

# OCR text output
# image = cv.imread("D:/images/1024_mask.png")
# question = "what‘s text in the image?"

# image classification
# image = cv.imread("D:/images/boogup.jpg")
# question = "what‘s text in the image?"

# object detection
image = cv.imread("./elephant2.jpg")
question = "Outline the position of each elephants and output all the coordinates in JSON format"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./elephants.jpg",
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

output_text = processor.batch_decode(generated_ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
boxes = []
lines = output_text[0].splitlines()
labels = []
for i, line in enumerate(lines):
    print("number of line: ", i ,  line)
    if line.strip().startswith("{\"bbox_2d\": ["):
        res = line.strip().split(", ")
        x1 = res[0].split("[")[-1]
        y1 = res[1]
        x2 = res[2]
        y2 = res[3][:len(res[3])-1]
        print("box: ", x1, y1, x2, y2)
        boxes.append([x1, y1, x2, y2])
        labels.append(res[-1].split(": ")[-1].replace("\"","").replace("}",""))

for i, box in enumerate(boxes):
    abs_x1 = int(box[0])
    abs_y1 = int(box[1])
    abs_x2 = int(box[2])
    abs_y2 = int(box[3])
    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
    cv.rectangle(image, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
    cv.putText(image, labels[i], (abs_x1, abs_y1), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv.imshow("QWen2.5-VL Multi-Models@gloomyfish-2025", image)
cv.waitKey(0)
cv.destroyAllWindows()
