from tqdm import tqdm
from multiwoz_utils import *

import json
import os


def parse_data(args, from_dir, to_dir):
    with open(f"{from_dir}/valListFile.txt", 'r') as f:
        valid_list = f.readlines()
    with open(f"{from_dir}/testListFile.txt", 'r') as f:
        test_list = f.readlines()
    valid_list = [valid_name.strip() for valid_name in valid_list]
    test_list = [test_name.strip() for test_name in test_list]
        
    print("Parsing data...")
    with open(f"{from_dir}/data.json", 'r') as f:
        data = json.load(f)

    trg_domains = list(set([slot_desc.split('-')[0] for slot_desc in slot_descs]))
    for trg_domain in trg_domains:
        final_dir = f"{to_dir}/{trg_domain}"
        if not os.path.isdir(final_dir):
            os.makedirs(final_dir)

        id_maps = make_id_maps(args, data, valid_list, test_list, trg_domain=trg_domain)

        total_data = {
            f"{args.train_prefix}_utters": [],
            f"{args.train_prefix}_states": [],
            f"{args.valid_prefix}_utters": [],
            f"{args.valid_prefix}_states": [],
            f"{args.test_prefix}_utters": [],
            f"{args.test_prefix}_states": [],
        }
        
        train_slot_descs = {k: v for k, v in slot_descs.items() if trg_domain not in k}
        valid_slot_descs, test_slot_descs = copy.deepcopy(slot_descs), copy.deepcopy(slot_descs)

        print(f"Zero-shot: {trg_domain}")
        checked = {}
        for prefix, domains in id_maps.items():
            for domain, id_list in domains.items():
                print(f"{prefix}: {domain}")
                for dialogue_id in tqdm(id_list):
                    if dialogue_id in checked:
                        continue

                    obj = data[dialogue_id]
                    if prefix == args.train_prefix:
                        utters, states = parse_dialogue(args, obj, train_slot_descs)
                    elif prefix == args.valid_prefix:
                        utters, states = parse_dialogue(args, obj, valid_slot_descs)
                    elif prefix == args.test_prefix:
                        utters, states = parse_dialogue(args, obj, test_slot_descs)

                    if utters is not None:
                        total_data[f"{prefix}_utters"].append(utters)
                        total_data[f"{prefix}_states"].append(states)

                    checked[dialogue_id] = True

        print("Saving each utterance & state files...")
        save_files(final_dir, args.train_prefix, total_data[f"{args.train_prefix}_utters"], total_data[f"{args.train_prefix}_states"])
        save_files(final_dir, args.valid_prefix, total_data[f"{args.valid_prefix}_utters"], total_data[f"{args.valid_prefix}_states"])
        save_files(final_dir, args.test_prefix, total_data[f"{args.test_prefix}_utters"], total_data[f"{args.test_prefix}_states"])
        
        print("Saving the slot description json files...")
        with open(f"{final_dir}/{args.train_prefix}_{args.slot_descs_prefix}.json", 'w') as f:
            json.dump(train_slot_descs, f)
        with open(f"{final_dir}/{args.valid_prefix}_{args.slot_descs_prefix}.json", 'w') as f:
            json.dump(valid_slot_descs, f)
        with open(f"{final_dir}/{args.test_prefix}_{args.slot_descs_prefix}.json", 'w') as f:
            json.dump(test_slot_descs, f)

        print("Calculating additional infos...")
        num_train_utters = count_utters(total_data[f"{args.train_prefix}_utters"])
        num_valid_utters = count_utters(total_data[f"{args.valid_prefix}_utters"])
        num_test_utters = count_utters(total_data[f"{args.test_prefix}_utters"])

        print(f"<Total Spec: MultiWoZ 2.1 (Zero-shot: {trg_domain})>")
        print(f"The # of train dialogues: {len(total_data[f'{args.train_prefix}_utters'])}")
        print(f"The # of train utterances: {num_train_utters}")
        print(f"The # of valid dialogues: {len(total_data[f'{args.valid_prefix}_utters'])}")
        print(f"The # of valid utterances: {num_valid_utters}")
        print(f"The # of test dialogues: {len(total_data[f'{args.test_prefix}_utters'])}")
        print(f"The # of valid utterances: {num_test_utters}")
            

def make_id_maps(args, data, valid_list, test_list, trg_domain):
    id_maps = {
        args.train_prefix: {slot_desc.split('-')[0]: [] for slot_desc in slot_descs if slot_desc.split('-')[0] != trg_domain},
        args.valid_prefix: {slot_desc.split('-')[0]: [] for slot_desc in slot_descs if slot_desc.split('-')[0] == trg_domain},
        args.test_prefix: {slot_desc.split('-')[0]: [] for slot_desc in slot_descs if slot_desc.split('-')[0] == trg_domain}
    }
    
    for dialogue_id, obj in tqdm(data.items()):
        goal = obj['goal']
        domains = get_domains(goal)
                
        if dialogue_id in valid_list:
            prefix = args.valid_prefix
        elif dialogue_id in test_list:
            prefix = args.test_prefix
        else:
            prefix = args.train_prefix    
        
        if prefix == args.train_prefix:
            if trg_domain in domains:
                continue
        else:
            if trg_domain not in domains:
                continue
        
        for domain in domains:
            if domain in id_maps[prefix]:
                id_maps[prefix][domain].append(dialogue_id)
            
    return id_maps


def parse_dialogue(args, obj, slot_descs):
    goal = obj['goal']
    domains = get_domains(goal)
    
    state_template = {v: "none" for k, v in slot_descs.items()}
    
    log = obj['log']
    utters = [""] * len(log)
    states = [copy.deepcopy(state_template) for i in range(len(utters))]
    prev_speaker = ""
    for t, turn in enumerate(log):
        metadata = turn['metadata']
        if len(metadata) > 0:
            speaker = "system"
        else:
            speaker = "user"
            
        if prev_speaker == speaker:
            utters = utters[:t]
            states = states[:t]
            break

        if t == 0:
            assert speaker == "user"
            
        text = f"{speaker}:{normalize_text(turn['text'])}"
        utters[t] = text
        
        temp_state = {}
        for domain, state in metadata.items():
            if domain in domains:
                book = state['book']
                semi = state['semi']
                
                for slot, value in book.items():
                    if slot not in ['booked', 'ticket']:
                        slot_type = f"{domain}-book {slot}"
                        normalized_value = normalize_label(slot_type, value)
                        temp_state[slot_type] = normalized_value
                
                for slot, value in semi.items():
                    slot_type = f"{domain}-{slot}"
                    normalized_value = normalize_label(slot_type, value)
                    temp_state[slot_type] = normalized_value
        
        for slot_type, value in temp_state.items():
            states[t][slot_descs[slot_type]] = value
            
        prev_speaker = speaker
        
    assert len(utters) == len(states)
    
    return utters, states
