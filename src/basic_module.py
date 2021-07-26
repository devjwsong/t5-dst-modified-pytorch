from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from utils import *
from pytorch_lightning import seed_everything
from argparse import Namespace

import torch
import pytorch_lightning as pl
import numpy as np


class BasicModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        
        seed_everything(args.seed, workers=True)
        
        self.args = args
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer.get_vocab()))

        special_tokens = {'sep_token': self.args.sep_token}
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.model.resize_token_embeddings(self.args.vocab_size)
        print(self.tokenizer)
        print(self.model)
        
        self.args.eos_token = self.tokenizer.eos_token
        self.args.unk_token = self.tokenizer.unk_token
        self.args.pad_token  =self.tokenizer.pad_token
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.unk_id = vocab[self.args.unk_token]
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.sep_id = vocab[self.args.sep_token]
        
        self.save_hyperparameters(args)
        
    def forward(self, src_ids):  # (B, S_L)
        outputs = self.model.generate(input_ids=src_ids, max_length=self.args.trg_max_len)  # (B, T_L)
    
        return outputs
    
    def training_step(self, batch, batch_idx):
        src_ids, trg_ids = batch  # (B, S_L), (B, T_L)
        outputs = self.model(input_ids=src_ids, labels=trg_ids)
        loss = outputs['loss']  # ()
            
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        train_losses = []
        for result in training_step_outputs:
            train_losses.append(result['loss'])
        
        train_losses = [loss.item() for loss in train_losses]
        train_loss = np.mean(train_losses)
        
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def validation_step(self, batch, batch_idx):
        src_ids, trg_ids = batch  # (B, S_L), (B, T_L)
        outputs = self.forward(src_ids)  # (B, T_L)
            
        return {'preds': outputs, 'trues': trg_ids}
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_preds = []
        valid_trues = []
        
        for result in validation_step_outputs:
            valid_preds.append(result['preds'])
            valid_trues.append(result['trues'])
        
        valid_preds = make_basic_strs(valid_preds, self.tokenizer, self.args.pad_token, self.args.eos_token)
        valid_trues = make_basic_strs(valid_trues, self.tokenizer, self.args.pad_token, self.args.eos_token)
        
        assert len(valid_preds) == len(valid_trues)
        assert len(valid_preds) % len(self.args.valid_slot_list) == 0
        
        valid_joint_goal_acc = get_joint_goal_acc(valid_preds, valid_trues, self.args.valid_slot_list, trg_domain=self.args.trg_domain)
        valid_slot_accs = get_slot_acc(valid_preds, valid_trues, self.args.valid_slot_list, trg_domain=self.args.trg_domain)
        
        print(f"Valid Joint Goal Acc: {valid_joint_goal_acc}")
        print("Valid Slot Acc")
        print(valid_slot_accs)
        
        self.log(f"valid_joint_goal_acc", valid_joint_goal_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"valid_slot_acc", valid_slot_accs['mean'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def test_step(self, batch, batch_idx):
        src_ids, trg_ids = batch  # (B, S_L), (B, T_L)
        outputs = self.forward(src_ids)  # (B, T_L)
            
        return {'preds': outputs, 'trues': trg_ids}
    
    def test_epoch_end(self, test_step_outputs):
        test_preds = []
        test_trues = []
        
        for result in test_step_outputs:
            test_preds.append(result['preds'])
            test_trues.append(result['trues'])
        
        test_preds = make_basic_strs(test_preds, self.tokenizer, self.args.pad_token, self.args.eos_token)
        test_trues = make_basic_strs(test_trues, self.tokenizer, self.args.pad_token, self.args.eos_token)
        
        assert len(test_preds) == len(test_trues)
        assert len(test_preds) % len(self.args.test_slot_list) == 0
        
        test_joint_goal_acc = get_joint_goal_acc(test_preds, test_trues, self.args.test_slot_list, trg_domain=self.args.trg_domain)
        test_slot_accs = get_slot_acc(test_preds, test_trues, self.args.test_slot_list, trg_domain=self.args.trg_domain)
        
        print(f"Test Joint Goal Acc: {test_joint_goal_acc}")
        print("Test Slot Acc")
        print(test_slot_accs)
        
        self.log(f"test_joint_goal_acc", test_joint_goal_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_slot_acc", test_slot_accs['mean'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.warmup_ratio < 0.0:
            return [optim]
        else:
            scheduler = {
                'scheduler': get_linear_schedule_with_warmup(
                                optim, 
                                num_warmup_steps=self.args.num_warmup_steps, 
                                num_training_steps=self.args.num_training_steps
                            ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1

            }

            return [optim], [scheduler]