import pickle
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_constant_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
import random
import sentencepiece as spm
from tqdm import tqdms
import numpy as np

class LMDataLoader(Dataset):

    def __init__(self, data, tokenizer, max_len, eos_token, pad_token):

        random.shuffle(data)
        self.dataset = self.process_text(data, tokenizer, max_len, eos_token, pad_token)

    def process_text(self, data, tokenizer, max_len, eos_token, pad_token):
        tokenized_text = tokenizer.encode(data)
        for i, tokens in enumerate(tokenized_text):
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
                tokens[-1] = eos_token
            else:
                tokens.append(eos_token)
                n = max_len - len(tokens)
                paddings = [pad_token] * n
                tokens.extend(paddings)

            tokenized_text[i] = tokens
        return tokenized_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return torch.tensor(self.dataset[item])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    SEED = 42
    BATCH_SIZE = 16
    DEVICE = "cuda"
    MAX_LEN = 256
    EOS_TOKEN = 2
    PAD_TOKEN = 1
    EPOCHS = 20
    LEARNING_RATE = 6e-5
    ADAM_EPSILON = 1e-6
    WARMUP_SAMPLES = 0

    DATAPATH = "../data/cleaned_data/clean_data"
    MODEL_PATH = "../model/gpt2_model/pretrained"

    set_seed(SEED)

    with open(DATAPATH, "rb") as wb:
        data = pickle.load(wb)

    tokenizer = spm.SentencePieceProcessor(model_file="tokenizer")
    dataset = LMDataLoader(data, tokenizer, MAX_LEN, EOS_TOKEN, PAD_TOKEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)


    config = GPT2Config.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, configt=config)
    model.to(DEVICE)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
    warmup_steps = WARMUP_SAMPLES / BATCH_SIZE
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    for epoch in EPOCHS:
        iterator = tqdm(dataloader)
        tr_loss = 0
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(iterator):
            inputs = batch.to(DEVICE)
            label = batch.to(DEVICE)
            attention_mask = torch.ones(batch.size[0], MAX_LEN)
            attention_mask[inputs == PAD_TOKEN] = 0
            attention_mask.to(DEVICE)
            model.train()
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            iterator.set_postfix(Perplexity=f"{(tr_loss/(i+1)).2f}")

        model.eval()
        outputs = model.generate(
            torch.tensor([tokenizer.encode("Монгол")]).to(DEVICE),
            max_length=150,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            num_return_sequences=5,
            eos_token_id=2,
            pad_token_id=1,
            early_stopping=True,
        )

        for output in outputs:
            output_text = tokenizer.decode(output.cpu().tolist())
            print(output_text)

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(f"../model/gpt2_model/checkpoint_{epoch}")

