from transformers import *
from data_utils.custom_dataset import *
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import numpy as np
import torch
import json


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
    print(args.vocab_size)
    
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
    ppd = DstPadCollate(pad_id=args.pad_id)
    
    with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{args.slot_descs_prefix}.json", 'r') as f:
        slot_descs = json.load(f)
    args.slot_list = [v for k, v in slot_descs.items()]
    
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
    
    model = model.to(args.device)
    
    num_batches = len(train_loader)
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_training_steps)
    
    writer = SummaryWriter()
        
    print("Train starts.")
    train(args, model, tokenizer, train_loader, valid_loader, optim, scheduler, save_dir, writer)
    
    print("Test starts.")
    test_joint_goal_acc, test_slot_accs = evaluate(args, model, tokenizer, test_loader)
    print(f"Test Joint Goal Accuracy: {test_joint_goal_acc}")
    print("Test slot Accuracy")
    print(test_slot_accs)
    

def train(args, model, tokenizer, train_loader, valid_loader, optim, scheduler, save_dir, writer):
    best_joint_goal_acc = 0.0
    for epoch in range(1, args.num_epochs+1):
        model.train()
        
        print("#" * 50 + f"Epoch: {epoch}" + "#" * 50)
        train_losses = []
        for b, batch in enumerate(tqdm(train_loader)):
            src_ids, trg_ids = batch  # (B, S_L), (B, T_L)
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
        
        print("Validation starts.")
        valid_joint_goal_acc, valid_slot_accs = evaluate(args, model, tokenizer, valid_loader)
        
        writer.add_scalar("valid_joint_goal_acc", valid_joint_goal_acc, epoch)
        writer.add_scalar("valid_slot_acc", valid_slot_accs['mean'], epoch)
        
        if valid_joint_goal_acc > best_joint_goal_acc:
            best_joint_goal_acc = valid_joint_goal_acc
            torch.save(model.state_dict(), f"{save_dir}/{args.save_prefix}_epoch{epoch}_joint{best_joint_goal_acc}.pt")
            print("Best model saved.")
        
        print(f"Best Valid Joint Goal Accuracy: {best_joint_goal_acc}")
        print(f"Valid Joint Goal Accuracy: {valid_joint_goal_acc}")
        print("Valid Slot Accuracy")
        print(valid_slot_accs)


def evaluate(args, model, tokenizer, eval_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b, batch in enumerate(tqdm(eval_loader)):
            src_ids, trg_ids = batch  # (B, S_L), (B, T_L)
            src_ids = src_ids.to(args.device)

            outputs = model.generate(input_ids=src_ids, max_length=args.trg_max_len)  # (B, T_L)
            
            preds.append(outputs)
            trues.append(trg_ids)
            
    preds, trues = make_strs(preds, tokenizer, args.pad_token, args.eos_token), make_strs(trues, tokenizer, args.pad_token, args.eos_token)
    
    assert len(preds) == len(trues)
    assert len(preds) % len(args.slot_list) == 0
    
    joint_goal_acc = get_joint_goal_acc(preds, trues, len(args.slot_list))
    slot_accs = get_slot_acc(preds, trues, args.slot_list)
    
    return joint_goal_acc, slot_accs


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
    
    assert args.data_name in ["multiwoz"]
    assert args.model_name in ["t5-small", "t5-base"]
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
#     input("Please press Enter to proceed...")
    
    run(args)
    