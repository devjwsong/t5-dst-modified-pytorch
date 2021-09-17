from modified_dataset import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from modified_module import *

import argparse
import os
import json


def run(args):
    print("Loading the tokenizer & model...")
    pt_module = ModifiedModule(args)
    
    args.src_max_len = min(args.src_max_len, pt_module.model.config.n_positions)
    args.max_extras = min(args.max_extras, len(pt_module.tokenizer.additional_special_tokens)-1)
    
    print("Loading & preprocessing data...")
    # Zero-shot
    if args.trg_domain is not None:
        args.data_name = f"{args.data_name}/{args.trg_domain}"
    train_set = ModifiedDataset(args.train_prefix, args, pt_module.tokenizer)
    valid_set = ModifiedDataset(args.valid_prefix, args, pt_module.tokenizer)
    test_set = ModifiedDataset(args.test_prefix, args, pt_module.tokenizer)
    ppd = ModifiedPadCollate(pad_id=args.pad_id)
    
    with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{args.train_prefix}_{args.slot_descs_prefix}.json", 'r') as f:
        train_slot_descs = json.load(f)
    args.train_slot_list = [v for k, v in train_slot_descs.items()]
    with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{args.valid_prefix}_{args.slot_descs_prefix}.json", 'r') as f:
        valid_slot_descs = json.load(f)
    args.valid_slot_list = [v for k, v in valid_slot_descs.items()]
    with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{args.test_prefix}_{args.slot_descs_prefix}.json", 'r') as f:
        test_slot_descs = json.load(f)
    args.test_slot_list = [v for k, v in test_slot_descs.items()]
    
    seed_everything(args.seed, workers=True)
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    
    num_batches = len(train_loader)
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)
    
    print("Setting pytorch lightning callback & trainer...")
    filename = "best_ckpt_{epoch}_{valid_joint_goal_acc:.4f}_{valid_slot_acc:.4f}"
    monitor = "valid_joint_goal_acc"
    ckpt_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max',
        every_n_val_epochs=1,
        save_weights_only=True
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=True,
        mode='max',
    )

    print("Train starts.")
    trainer = Trainer(
        gpus=args.gpu,
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.max_grad_norm,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[ckpt_callback, early_stopping_callback],
        default_root_dir=args.log_dir,
        accelerator="ddp",
    )
    trainer.fit(model=pt_module, train_dataloader=train_loader, val_dataloaders=valid_loader)
    
    print("Test starts.")
    trainer.test(test_dataloaders=test_loader, ckpt_path='best')
    
    print("GOOD BYE.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--data_dir", type=str, default="data", help="The root directory for the entire data files.")
    parser.add_argument("--cached_dir", type=str, default="cached", help="The directory for cached files after processing.")
    parser.add_argument("--data_name", type=str, required=True, help="The data name to train/evaluate.")
    parser.add_argument("--trg_domain", type=str, required=False, help="The target domain to be excluded in zero-shot setting.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="The T5 model type.")
    parser.add_argument("--train_prefix", type=str, default="train", help="The train data prefix.")
    parser.add_argument("--valid_prefix", type=str, default="valid", help="The validation data prefix.")
    parser.add_argument("--test_prefix", type=str, default="test", help="The test data prefix.")
    parser.add_argument("--slot_descs_prefix", type=str, default="slot_descs", help="The slot description file prefix.")
    parser.add_argument("--num_epochs", type=int, default=10, help="The total number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="The batch size for train data loader.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="The batch size for evaluation data loader.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of subprocesses for data loading.")
    parser.add_argument("--src_max_len", type=int, default=512, help="The maximum length of the source sequence.")
    parser.add_argument("--trg_max_len", type=int, default=128, help="The maximum length of the target sequence.")
    parser.add_argument("--max_extras", type=int, default=5, help="The maximum number of slot types to include in one input.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio of warmup steps to total training steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="The maximum value of gradient.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="The minimum delta value for evaluation metric.")
    parser.add_argument("--patience", type=int, default=3, help="The number patience epochs before early stopping.")
    parser.add_argument("--sep_token", type=str, default="<sep>", help="The special token for separation.")
    parser.add_argument("--gpu", type=str, default="0", help="The indices of GPUs.")
    parser.add_argument("--log_dir", type=str, default="./", help="The location of lightning log directory.")
    parser.add_argument("--use_cached", action="store_true", help="Using cached data or not?")
    
    args = parser.parse_args()
    
    assert args.data_name in ["multiwoz_fullshot", "multiwoz_zeroshot"]
    assert args.model_name in ["t5-small", "t5-base"]
    if "zeroshot" in args.data_name:
        assert args.trg_domain in ["attraction", "hotel", "restaurant", "taxi", "train"]
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Please press Enter to proceed...")
    
    run(args)
    