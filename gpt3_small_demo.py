from transformers import GPT2Tokenizer, GPTNeoForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('minhtoan/gpt3-small-finetune-cnndaily-news')
model = GPTNeoForCausalLM.from_pretrained('minhtoan/gpt3-small-finetune-cnndaily-news')

text = "Ever noticed how plane seats appear to be getting smaller and smaller? "
input_ids = tokenizer.encode(text, return_tensors='pt')
max_length = 150

sample_outputs = model.generate(input_ids, do_sample=True, max_length=max_length,temperature = 0.8)

for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist())))
    print('\n---')