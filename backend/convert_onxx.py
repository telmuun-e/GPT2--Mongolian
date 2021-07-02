import torch.onnx
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(model_file="tokenizer/mn.model")
config = GPT2Config.from_pretrained("model/model")
model = GPT2LMHeadModel.from_pretrained("model/model", config=config)

text = "Монгол"
tokens = tokenizer.encode(text)
tokens = torch.tensor([tokens])
model.eval()

torch.onnx.export(
    model, 
    tokens,
    "model/model.onnx",
    input_names=["tokens"],
    output_names=["output"],
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
)
