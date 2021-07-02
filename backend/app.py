from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import sentencepiece as spm
import random

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        text = ""
        if request.method=="POST":
            text = request.json.get("text","")
    except Exception as ex:
        prediction = str("Угтвар бичнэ үү")
        for r in request:
            print(request[r],r)
            print("#######")
        print(request.json,"#############", request)
        return jsonify(prediction)
    print(text)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor([tokens])
    outputs = model.generate(
        tokens, 
        max_length=150, 
        do_sample=True, 
        top_k=40, 
        top_p=0.95, 
        num_return_sequences=5,
        eos_token_id=2,
        pad_token_id=1,
        early_stopping=True)
    
    output = random.sample(outputs.tolist(), 1)
    out = tokenizer.decode(output[0])
    prediction = str(out)
    return {"prediction":prediction, "status":200}

if __name__ == "__main__":
    tokenizer = spm.SentencePieceProcessor(model_file="../model/tokenizer/mn.model")
    model_path = "../model/gpt2_model/model"
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    app.run(debug=True, host="0.0.0.0", port=8080)