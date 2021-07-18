from transformers import *
from data_utils.custom_dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import numpy as np
import torch
import json


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def run(args):
    print("Loading the tokenizer & model...")
    fix_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    special_tokens = {'sep_token': args.sep_token}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    model.resize_token_embeddings(args.vocab_size)
    print(tokenizer)
    print(model)
    
    args.eos_token = tokenizer.eos_token
    args.unk_token = tokenizer.unk_token
    args.pad_token  =tokenizer.pad_token
    args.eos_id = vocab[args.eos_token]
    args.unk_id = vocab[args.unk_token]
    args.pad_id = vocab[args.pad_token]
    args.sep_id = vocab[args.sep_token]
    
    args.src_max_len = min(args.src_max_len, model.config.n_positions)
    
    print("Loading & preprocessing data...")
    train_set = DstDataset(args.train_prefix, args, tokenizer)
    valid_set = DstDataset(args.valid_prefix, args, tokenizer)
    test_set = DstDataset(args.test_prefix, args, tokenizer)
    ppd = DSTPadCollate(pad_id=args.pad_id)
    
    with open(f"{args.data_dir}/{args.cached_dir}/{slot_descs_prefix}.json", 'r') as f:
        slot_descs = json.load(f)
    args.num_slots = len(slot_descs)
    
    fix_seed(args.seed)
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, pin_memory=True)
    
    print("Preparing for training...")
    save_dir = f"{args.save_dir}/{args.data_name}/{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpu}")
        
    num_batches = len(train_loader)
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_batches)
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_training_steps)
    
    writer = SummarWriter()
        
    print("Train starts.")
    
    print("Test starts.")
    

def train(args, model, train_loader, valid_loader optim, scheduler, writer):
    best_joint_goal_acc = 0.0
    for epoch in range(1, args.num_epochs+1):
        model.train()
        
        print("#" * 50 + f"Epoch: {epoch}" + "#" * 50)
        train_losses = []
        for b, batch in enumerate(tqdm(train_loader)):
            src_ids, trg_ids = batch  # (B, S, S_L), (B, S, T_L)
            src_ids, trg_ids = src_ids.to(args.device), trg_ids.to(args.device)
            
            outputs = model(input_ids=src_ids, labels=trg_ids)
            loss = outputs['loss']  # ()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            
            train_losses.append(loss)
            
        train_losses = [loss.item() for loss in train_losses]
        train_loss = np.mean(train_losses)
        print(f"Train Loss: {train_loss}")
        
        writer.add_scalar("train_loss", train_loss, epoch)
        
        evaluate(args, model, valid_loader, writer)


def evaluate(args, model, eval_loader, writer):
    pass
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cached_dir", type=str, default="cached")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_prefix", type=str, default="train")
    parser.add_argument("--valid_prefix", type=str, default="valid")
    parser.add_argument("--test_prefix", type=str, default="test")
    parser.add_argument("--slot_descs_prefix", type=str, default="slot_descs")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--src_max_len", type=int, default=512)
    parser.add_argument("--trg_max_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--sigmoid_threshold", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sep_token", type=str, default="<sep>")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--save_prefix", type=str, default="best_model")
    
    args = parser.parse_args()
    
    assert data_name in ["multiwoz"]
    assert model_name in ["t5-small, t5-base"]
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Please press Enter to proceed...")
    
    run(args)
    