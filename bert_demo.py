from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
# tokens = tokenizer.tokenize("All work and no play makes Jack a dull boy!")
# print(tokens)

# sentence_a = "The weather is nice today."
# sentence_b = "I will go for a walk."

sentence_a = "The weather is nice today."
sentence_b = "The stock market crashed in 2008."

encoding = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')

outputs = model(**encoding)
logits = outputs.logits

prob = torch.softmax(logits, dim=1)
is_next_prob = prob[0][0].item()
not_next_prob = prob[0][1].item()

if is_next_prob > not_next_prob:
    print("Sentence B is likely to follow Sentence A (Is Next).")
    print("Prediction: Is Next (Probability: %.2f)"%is_next_prob)
else:
    print("Sentence B is NOT likely to follow Sentence A (Not Next).")
    print("Prediction: Not Next (Probability: %.2f)" % not_next_prob)
