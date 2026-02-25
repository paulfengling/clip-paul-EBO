from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompts = [
    "The future of large language model",
    "opencv is very important"
]

inputs = tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    max_length=100,
    temperature=0.7,
    do_sample=True
)

generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for prompt, text in zip(prompts, generated_texts):
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text}")


