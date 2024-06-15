# from huggingface_hub import login
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
# from flask import Flask, request, jsonify


# login(token = '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')
# tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
# model = AutoModelForCausalLM.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
# streamer = TextStreamer(tokenizer, skip_prompt=True)
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=2048,
#     temperature=0.6,
#     pad_token_id=tokenizer.eos_token_id,
#     top_p=0.95,
#     repetition_penalty=1.2,
#     streamer=streamer
# )

# messages = [
#     {"role": "user", "content": "你是誰?"},
# ]




# app = Flask(__name__)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message')
#     if not user_message:
#         return jsonify({"error": "No message provided"}), 400

#     messages = [{"role": "user", "content": user_message}]
#     response = pipe(messages)
#     return jsonify({"response": response[0]['generated_text']})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
from huggingface_hub import login
from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

login(token='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')

app = Flask(__name__)

# Load the model and tokenizer
model_name = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def stream_response(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 100,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    # Generate response using the model
    with torch.no_grad():
        generation = model.generate(**generation_kwargs)

    for token in streamer:
        yield token

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # Stream the response
    return Response(stream_response(prompt), content_type='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
