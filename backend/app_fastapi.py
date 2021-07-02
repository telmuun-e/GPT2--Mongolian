import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import sentencepiece as spm
import random

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict():
    try:
        text = ""
        text = Request.json.get("text","")
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
    tokenizer = spm.SentencePieceProcessor(model_file="/home/app/tokenizer/mn.model")
    model_path = "/home/app/model/model"
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
    uvicorn.run(app, host="0.0.0.0", port=8080)