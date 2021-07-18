import argparse
import os
import process_multiwoz


data_names = ['multiwoz']


def process_data(args, cached_dir):
    print("Processing data for finetuning...")
    
    for data_name in data_names:
        print("#" * 100)
        print(f"Parsing {data_name}...")
        
        to_dir = f"{cached_dir}/{data_name}"
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
            
        if data_name == 'multiwoz':
            from_dir = f"{args.data_dir}/{args.raw_dir}/MultiWOZ_2.1"
            process_multiwoz.parse_data(args, from_dir, to_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--raw_dir", type=str, default="raw")
    parser.add_argument("--cached_dir", type=str, default="cached")
    parser.add_argument('--train_prefix', type=str, default="train")
    parser.add_argument('--valid_prefix', type=str, default="valid")
    parser.add_argument('--test_prefix', type=str, default="test")
    parser.add_argument('--slot_descs_prefix', type=str, default="slot_descs")
    
    args = parser.parse_args()
    
    cached_dir = f"{args.data_dir}/{args.cached_dir}"

    process_data(args, cached_dir)
    