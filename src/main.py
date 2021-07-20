from data_utils.custom_dataset import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from train_module import *

import argparse
import os
import json


def run(args):
    print("Loading the tokenizer & model...")
    pt_module = TrainModule(args)
    
    args.src_max_len = min(args.src_max_len, pt_module.model.config.n_positions)
    
    print("Loading & preprocessing data...")
    train_set = DstDataset(args.train_prefix, args, pt_module.tokenizer)
    valid_set = DstDataset(args.valid_prefix, args, pt_module.tokenizer)
    test_set = DstDataset(args.test_prefix, args, pt_module.tokenizer)
    ppd = DstPadCollate(pad_id=args.pad_id)
    
    with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{args.slot_descs_prefix}.json", 'r') as f:
        slot_descs = json.load(f)
    args.slot_list = [v for k, v in slot_descs.items()]
    
    seed_everything(args.seed, workers=True)
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    
    num_batches = len(train_loader)
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)

    print("Setting pytorch lightning callback & trainer...")
    filename = "best_ckpt_{epoch}_{valid_joint_goal_acc:.4f}_{valid_slot_acc:.4f}"
    monitor = "valid_slot_acc"
    ckpt_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max',
        every_n_val_epochs=1,
        save_weights_only=True
    )

    print("Train starts.")
    trainer = Trainer(
        gpus=args.gpu,
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.max_grad_norm,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[ckpt_callback],
        default_root_dir=args.log_dir,
        accelerator="ddp",
    )
    trainer.fit(model=pt_module, train_dataloader=train_loader, val_dataloaders=valid_loader)
    
    print("Test starts.")
    trainer.test(test_dataloader=test_loader, ckpt_path='best')
    
    print("GOOD BYE.")


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
    parser.add_argument("--log_dir", type=str, default="./")
    parser.add_argument("--use_cached", action="store_true")
    
    args = parser.parse_args()
    
    assert args.data_name in ["multiwoz"]
    assert args.model_name in ["t5-small", "t5-base"]
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Please press Enter to proceed...")
    
    run(args)
    