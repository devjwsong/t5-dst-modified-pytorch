from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import chain

import torch
import os
import pickle
import copy


class ModifiedDataset(Dataset):
    def __init__(self, prefix, args, tokenizer):
        print(f"Loading {prefix} dataset...")
        
        if not args.use_cached:
            print("Since you chose not to use cached data, preprocessing will start first.")
            total_src_ids, total_trg_ids = [], []
            
            with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{prefix}_utters.pickle", 'rb') as f:
                utters = pickle.load(f)
            with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/{prefix}_states.pickle", 'rb') as f:
                states = pickle.load(f)
        
            for d, utter_dialogue in enumerate(tqdm(utters)):
                state_dialogue = states[d]

                utter_hists, state_hists = [], []
                for u, utter in enumerate(utter_dialogue):
                    state = state_dialogue[u]

                    if utter.startswith("system"):
                        user_utter_ids = tokenizer.encode(utter_dialogue[u-1])[:-1]
                        sys_utter_ids = tokenizer.encode(utter)[:-1]
                        utter_hists.append([user_utter_ids, sys_utter_ids])
                        state_hists.append(state)
                
                for t, turn in enumerate(utter_hists):                    
                    src_ids, trg_ids = self.make_seqs(
                        utter_hists[:t+1], state_hists[:t+1], 
                        args.pad_id, args.sep_id, args.eos_id, args.max_extras, args.extra_ids, 
                        args.src_max_len, args.trg_max_len, tokenizer
                    )
                    
                    if src_ids is not None and trg_ids is not None:
                        total_src_ids += src_ids
                        total_trg_ids += trg_ids
        
            assert len(total_src_ids) == len(total_trg_ids)
            
            if not os.path.exists(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified"):
                os.makedirs(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified")
            
            with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified/{prefix}_src_ids.pickle", 'wb') as f:
                pickle.dump(total_src_ids, f)
            with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified/{prefix}_trg_ids.pickle", 'wb') as f:
                pickle.dump(total_trg_ids, f)

        with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified/{prefix}_src_ids.pickle", 'rb') as f:
            self.src_ids = pickle.load(f)
        with open(f"{args.data_dir}/{args.cached_dir}/{args.data_name}/modified/{prefix}_trg_ids.pickle", 'rb') as f:
            self.trg_ids = pickle.load(f)
            
        print(f"The # of {prefix} samples: {len(self.src_ids)}.")
        
    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, i):
        return self.src_ids[i], self.trg_ids[i]
                
    def make_seqs(self, utter_hists, state_hists, pad_id, sep_id, eos_id, max_extras, extra_ids, src_max_len, trg_max_len, tokenizer):
        start_idx = 0
        while start_idx < len(utter_hists):
            partial_hists = utter_hists[start_idx:]
            partial_hists[-1] = partial_hists[-1][:-1]
            prefix_ids = list(chain.from_iterable(list(chain.from_iterable(partial_hists))))
            
            if start_idx > 0:
                result_state = self.cut_state(state_hists[start_idx-1], state_hists[-1])
            else:
                result_state = state_hists[-1]
            
            src_ids, trg_ids = [], []  # (E, S_L), (E, T_L)
            slot_desc_ids_list, slot_value_ids_list, len_sums = [], [], [0]
            for s, (slot_desc, slot_value) in enumerate(result_state.items()):
                slot_desc_ids = tokenizer.encode(f"{slot_desc}:")[:-1]
                slot_desc_ids_list.append(slot_desc_ids)
                slot_value_ids_list.append(tokenizer.encode(slot_value)[:-1])
                len_sums.append(len_sums[-1] + len(slot_desc_ids))
            assert len(slot_desc_ids_list)+1 == len(len_sums)
            
            start = 0
            included = 0
            while start < len(slot_desc_ids_list):
                for num_extras in range(max_extras, 0, -1):
                    try:
                        if start + num_extras < len(len_sums):
                            len_sum = len_sums[start+num_extras] - len_sums[start]
                            sample_desc_ids_list, sample_value_ids_list = slot_desc_ids_list[start:start+num_extras], slot_value_ids_list[start:start+num_extras]
                        else:
                            len_sum = len_sums[-1] - len_sums[start]
                            sample_desc_ids_list, sample_value_ids_list = slot_desc_ids_list[start:], slot_value_ids_list[start:]
                            
                        if len(prefix_ids) + 2 * len(sample_desc_ids_list) + len_sum + 1 <= src_max_len:
                            included += len(sample_desc_ids_list)
                            postfix_ids = [[sep_id] + sample_desc_ids_list[i] + [extra_ids[i]] for i in range(len(sample_desc_ids_list))]
                            postfix_ids = list(chain.from_iterable(postfix_ids))
                            seq_ids = prefix_ids + postfix_ids + [eos_id]

                            value_ids = [[extra_ids[i]] + sample_value_ids_list[i] for i in range(len(sample_desc_ids_list))]
                            value_ids = list(chain.from_iterable(value_ids)) + [extra_ids[len(sample_desc_ids_list)], eos_id]

                            assert len(value_ids) <= trg_max_len

                            src_ids.append(seq_ids)
                            trg_ids.append(value_ids)

                            break
                    except IndexError as e:
                        print(e)
                        print(f"start: {start}")
                        print(f"num_extras: {num_extras}")
                        print(f"len(len_sums): {len(len_sums)}")
                        print(f"len_sums: {len_sums}")
                        print(f"len(slot_desc_ids_list): {len(slot_desc_ids_list)}")
                        print(f"len(slot_value_ids_list): {len(slot_value_ids_list)}")
                        exit()
                
                start += num_extras
                
            if included == len(slot_desc_ids_list):
                assert len(src_ids) == len(trg_ids)
                return src_ids, trg_ids
            
            start_idx += 1

        return None, None  
    
    def cut_state(self, prev_state, trg_state):
        result_state = copy.deepcopy(trg_state)
        for slot_desc, value in trg_state.items():
            prev_value = prev_state[slot_desc]
            if prev_value == value and prev_value != 'none' and value != 'none':
                result_state[slot_desc] = 'none'
                
        return result_state
    

class ModifiedPadCollate():
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
            