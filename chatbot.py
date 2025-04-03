"""
This object will handle the following:
1) Connect to the open-source model
2) Handle Tokenization
3) Handle conversation history
4) Encode input and decode output
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def initialize_model():
    """
    Function to fetch a model and tokenizer from HuggingFace"
    Returns a dictionary with a "model" and a "tokenizer"
    """
    model_name = "facebook/blenderbot-400M-distill"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return {"model" : model, "tokenizer" : tokenizer}

def handle_conversation(model_and_tokenizer):
    """
    Loop that handles conversation and stores a conversation history.
    Input is a dictionary with a model and tokenizer.
    Look function initialize_model for more details on input
    """
    conversation_history = []
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    # Chat loop. Open until closed from terminal.
    while True:
        
        #Parse conversation history list into a string
        history_string = "\n".join(conversation_history)

        input_text = input("> ")

        #Tokenize the input text AND the history for context.
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

        #Generate response from the model
        outputs = model.generate(**inputs,max_new_tokens=1000)

        #Decode the response into readable text
        response = tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

        print(response)

        #Update conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

def main():
    model_and_tokenizer = initialize_model()
    handle_conversation(model_and_tokenizer)

if __name__ == "__main__":
    main()


