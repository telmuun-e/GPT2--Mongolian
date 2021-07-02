import pickle
import random
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_constant_schedule_with_warmup

class LMDataLoader(Dataset):

    def __init__(self, data, tokenizer, max_len, eos_token, pad_token):

        random.shuffle(data)
        self.dataset = self.process_text(data, tokenizer, max_len, eos_token, pad_token)

    def process_text(self, data, tokenizer, max_len, eos_token, pad_token):
        tokenized_text = tokenizer.encode(data)

        for i, tokens in enumerate(tokenized_text):

            if len(tokens) >= max_len:
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

def save_state(model, step, tokenizer):
    model.eval()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(f"../model/gpt2_model/checkpoint_{step}")
    
    outputs = model.generate(
        torch.tensor([tokenizer.encode("Би")]).to(DEVICE),
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

def evaluate(model, tokenizer, dataset, eval_indices):

    eval_sampler = SubsetRandomSampler(eval_indices)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=BATCH_SIZE, drop_last=True
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in eval_dataloader:
        inputs, labels = (batch, batch)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        attention_mask = torch.ones((batch.shape[0], MAX_LEN))
        attention_mask[inputs == PAD_TOKEN] = 0
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            lm_loss = outputs[0]

            eval_loss += lm_loss.item()  # lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {"perplexity": perplexity, "loss": eval_loss}

    return result


if __name__ == "__main__":

    SEED = 42
    BATCH_SIZE = 12
    DEVICE = torch.device("cuda")
    MAX_LEN = 256
    EOS_TOKEN = 2
    PAD_TOKEN = 1
    EPOCHS = 5
    LEARNING_RATE = 6e-5
    ADAM_EPSILON = 1e-6
    WARMUP_SAMPLES = 0

    DATAPATH = "../data/cleaned_data/clean_data"
    MODEL_PATH = "../model/gpt2_model/general"

    writer = SummaryWriter()

    set_seed(SEED)

    with open(DATAPATH, "rb") as wb:
        data = pickle.load(wb)

    tokenizer = spm.SentencePieceProcessor(model_file="../model/tokenizer/mn.model")
    dataset = LMDataLoader(data, tokenizer, MAX_LEN, EOS_TOKEN, PAD_TOKEN)
  
    config = GPT2Config.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, config=config)
    model.to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
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

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.02 * dataset_size))
    random.shuffle(indices)
    train_indices, eval_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)
    
    step = 0
    for epoch in range(EPOCHS):
        iterator = tqdm(dataloader)
        tr_loss = 0
        print(f"Epoch: {epoch}")
        
        for i, batch in enumerate(iterator):
            inputs = batch.to(DEVICE)
            labels = batch.to(DEVICE)
            attention_mask = torch.ones((batch.shape[0], MAX_LEN))
            attention_mask[inputs == PAD_TOKEN] = 0
            attention_mask = attention_mask.to(DEVICE)
            model.train()

            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            loss = tr_loss/(i+1)
            perplexity = torch.exp(torch.tensor(tr_loss/(i+1))).item()
            writer.add_scalar("Loss", perplexity, step)
            iterator.set_postfix(Perplexity="{:.2f}".format(perplexity), Loss="{:.2f}".format(loss))
            
            step += 1

            if step % 10000 == 0:
                result = evaluate(model, tokenizer, dataset, eval_indices)
                print(result)
                save_state(model, step, tokenizer)
