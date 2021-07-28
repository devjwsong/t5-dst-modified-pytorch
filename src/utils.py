def make_basic_strs(tensor_list, tokenizer, pad_token, eos_token):
    trg_ids_list = []
    for tensor in tensor_list:
        trg_ids_list += tensor.tolist()
    
    results = []
    for trg_ids in trg_ids_list:
        decoded = tokenizer.decode(trg_ids)
        decoded = decoded.replace(pad_token, "").replace(eos_token, "").strip()
        results.append(decoded)
    
    return results


def make_modified_strs(pred_list, true_list, tokenizer, pad_token, eos_token, extra_tokens):
    pred_ids_list,  true_ids_list = [], []
    for t in range(len(true_list)):
        pred_ids_list += pred_list[t].tolist()
        true_ids_list += true_list[t].tolist()
    
    pred_results, true_results = [], []
    for t in range(len(true_ids_list)):
        true_tokens = tokenizer.convert_ids_to_tokens(true_ids_list[t])
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids_list[t])
        
        true_tokens = [token for token in true_tokens if token != pad_token and token != eos_token]
        pred_tokens = [token for token in pred_tokens if token != pad_token and token != eos_token]
    
        def get_values(tokens):
            cur_tokens, values = [], []
            for token in tokens:
                if token in extra_tokens:
                    if len(cur_tokens) > 0:
                        values.append(tokenizer.convert_tokens_to_string(cur_tokens).replace(pad_token, "").replace(eos_token, "").strip())
                    cur_tokens = []
                else:
                    cur_tokens.append(token)
            
            return values
        
        true_values = get_values(true_tokens)
        pred_values = get_values(pred_tokens)
        
        if len(pred_values) >= len(true_values):
            pred_values = pred_values[:len(true_values)]
        else:
            pred_values += ["none"] * (len(true_values)-len(pred_values))
            
        assert len(pred_values) == len(true_values)
        
        pred_results += pred_values
        true_results += true_values
        
    return pred_results, true_results
    

def get_joint_goal_acc(preds, trues, slot_list, trg_domain=None, round_num=4):
    N, corr = 0, 0
    for start in range(0, len(trues), len(slot_list)):
        pred_state = preds[start:start+len(slot_list)]
        true_state = trues[start:start+len(slot_list)]
        
        if trg_domain is not None:
            pred_state = [pred_state[i] for i, slot in enumerate(slot_list) if trg_domain in slot]
            true_state = [true_state[i] for i, slot in enumerate(slot_list) if trg_domain in slot]
            assert len(pred_state) == len(true_state)
        
        N += 1
        if pred_state == true_state:
            corr += 1
            
    return round(corr / N, round_num)


def get_slot_acc(preds, trues, slot_list, trg_domain=None, round_num=4):
    counts = {slot: 0 for slot in slot_list}
    results = {slot: 0 for slot in slot_list}
    corr_count, total_count = 0, 0
    
    for start in range(0, len(trues), len(slot_list)):
        pred_state = preds[start:start+len(slot_list)]
        true_state = trues[start:start+len(slot_list)]
        
        for i in range(len(true_state)):
            slot = slot_list[i]
            counts[slot] += 1
            
            if trg_domain is None or (trg_domain is not None and trg_domain in slot):
                total_count += 1
                
            if pred_state[i] == true_state[i]:
                results[slot] += 1
                
                if trg_domain is None or (trg_domain is not None and trg_domain in slot):
                    corr_count += 1
    
#     total_sum = 0.0
#     num_slots = len(slot_list)
#     if trg_domain is not None:
#         num_slots = 0
#         for slot in slot_list:
#             if trg_domain in slot:
#                 num_slots += 1
    
    for slot, slot_count in results.items():
        try:
            results[slot] = round(slot_count / counts[slot], round_num)
        except ZeroDivisionError:
            results[slot] = 0.0
#         if trg_domain is None or (trg_domain is not None and trg_domain in slot):
#             total_sum += results[slot]
        
    results['mean'] = round(corr_count / total_count, round_num)
    
    return results
    