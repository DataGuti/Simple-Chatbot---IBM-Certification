"""
This object will handle the following:
1) Connect to the open-source model
2) Handle Tokenization
3) Handle conversation history
4) Encode input and decode output
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

history_string = "\n".join(conversation_history)

input_text = "hello, how are you doing?"

inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

outputs = model.generate(**inputs,max_new_tokens=1000)

response = tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

print(response)

conversation_history.append(input_text)
conversation_history.append(response)
