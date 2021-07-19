from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import pickle
import copy


class DstDataset(Dataset):
    def __init__(self, prefix, args, tokenizer):
        print(f"Loading {prefix} dataset...")
        
        with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{prefix}_utters.pickle", 'rb') as f:
            utters = pickle.load(f)
        with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{prefix}_states.pickle", 'rb') as f:
            states = pickle.load(f)
            
        self.src_ids = []  # (N, S_L)
        self.trg_ids = []  # (N, T_L)
        
        for d, utter_dialogue in enumerate(tqdm(utters)):
            state_dialogue = states[d]
            
            utter_hists, state_hists = [], []
            for u, utter in enumerate(utter_dialogue):
                state = state_dialogue[u]
                
                if utter.startswith("system"):
                    assert u % 2 == 1
                    user_utter_ids = tokenizer.encode(utter_dialogue[u-1])[:-1]
                    sys_utter_ids = tokenizer.encode(utter)[:-1]
                    utter_hists.append([user_utter_ids, sys_utter_ids])
                    state_hists.append(state)
                    
            for t, turn in enumerate(utter_hists):
                src_ids, trg_ids = self.make_seqs(
                    utter_hists[:t+1], state_hists[:t+1], args.pad_id, args.sep_id, args.eos_id, args.src_max_len, args.trg_max_len, tokenizer
                )
                
                if src_ids is not None and trg_ids is not None:
                    self.src_ids += src_ids
                    self.trg_ids += trg_ids
        
        assert len(self.src_ids) == len(self.trg_ids)
        
    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.src_ids[i], self.trg_ids[i]
                
    def make_seqs(self, utter_hists, state_hists, pad_id, sep_id, eos_id, src_max_len, trg_max_len, tokenizer):
        start_idx = 0
        while start_idx < len(utter_hists):
            partial_hists = utter_hists[start_idx:]
            partial_hists[-1] = partial_hists[-1][:-1]
            prefix_ids = list(chain.from_iterable(list(chain.from_iterable(partial_hists))))
            
            if start_idx > 0:
                result_state = self.cut_state(state_hists[start_idx-1], state_hists[-1])
            else:
                result_state = state_hists[-1]
            
            src_ids, trg_ids = [], []  # (S, S_L), (S, T_L)
            for slot_desc, value in result_state.items():
                slot_desc_ids = tokenizer.encode(slot_desc)
                src_ids.append(prefix_ids + [sep_id] + slot_desc_ids)
                value_ids = tokenizer.encode(value)
                trg_ids.append(value_ids)
                assert len(value_ids) <= trg_max_len
            
            all_pass = True
            for seq in src_ids:
                if len(seq) > src_max_len:
                    all_pass = False
            
            if all_pass:
                return src_ids, trg_ids
            
            start_idx += 1

        return None, None  
    
    def cut_state(prev_state, trg_state):
        result_state = copy.deepcopy(trg_state)
        for slot_desc, value in trg_state.items():
            prev_value = prev_state[slot_desc]
            if prev_value == value and prev_value != 'none' and value != 'none':
                result_state[slot_desc] = 'none'
                
        return result_state
    

class DstPadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        # Padding
        src_ids, trg_ids = [], []
        for idx, pair in enumerate(batch):
            src_ids.append(torch.LongTensor(pair[0]))
            trg_ids.append(torch.LongTensor(pair[1]))

        padded_src_ids = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=self.pad_id)
        padded_trg_ids = torch.nn.utils.rnn.pad_sequence(trg_ids, batch_first=True, padding_value=self.pad_id)

        # set contiguous for memory efficiency
        return padded_src_ids.contiguous(), padded_trg_ids.contiguous()
            