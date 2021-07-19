from itertools import chain
from tqdm import tqdm

import torch
import random
import numpy as np


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    

def make_strs(tensor_list, tokenizer, pad_token, eos_token):
    trg_ids_list = []
    for tensor in tensor_list:
        trg_ids_list += tensor.tolist()
    
    results = []
    for trg_ids in trg_ids_list:
        decoded = tokenizer.decode(trg_ids)
        decoded = decoded.replace(pad_token, "").replace(eos_token, "").strip()
        results.append(decoded)
    
    return results


def get_joint_goal_acc(preds, trues, num_slots, round_num=4):
    N, corr = 0, 0
    for start in range(0, len(trues), num_slots):
        pred_state = preds[start:start+num_slots]
        true_state = trues[start:start+num_slots]
        
        N += 1
        if pred_state == true_state:
            corr += 1
            
    return round(corr / N, round_num)


def get_slot_acc(preds, trues, slot_list, round_num=4):
    results = {slot: 0 for slot in slot_list}
    
    for start in range(0, len(trues), len(slot_list)):
        pred_state = preds[start:start+len(slot_list)]
        true_state = trues[start:start+len(slot_list)]
        
        for i in range(len(true_state)):
            slot = slot_list[i]
            if pred_state[i] == true_state[i]:
                results[slot] += 1
    
    total_sum = 0
    for slot, corr_count in results.items():
        results[slot] = round(corr_count / (len(preds) // len(slot_list)), round_num)
        total_sum += results[slot]
        
    results['mean'] = round(total_sum / len(slot_list), round_num)
    
    return results
    