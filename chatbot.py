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

