import argparse
import os
import process_multiwoz_fullshot, process_multiwoz_zeroshot


data_names = ['multiwoz']


def run(args, cached_dir):    
    for data_name in data_names:
        print("#" * 100)
        print(f"Parsing {data_name}...")
            
        if data_name == 'multiwoz':
            from_dir = f"{args.data_dir}/{args.raw_dir}/MultiWOZ_2.1"
            
            print("Full-shot")
            to_dir = f"{cached_dir}/{data_name}_fullshot"
            if not os.path.isdir(to_dir):
                os.makedirs(to_dir)
            process_multiwoz_fullshot.parse_data(args, from_dir, to_dir)
            
            print("Zero-shot")
            to_dir = f"{cached_dir}/{data_name}_zeroshot"
            if not os.path.isdir(to_dir):
                os.makedirs(to_dir)
            process_multiwoz_zeroshot.parse_data(args, from_dir, to_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="The root directory for the entire data files.")
    parser.add_argument("--raw_dir", type=str, default="raw", help="The directory which contains raw data files.")
    parser.add_argument("--cached_dir", type=str, default="cached", help="The directory for cached files after processing.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The train data prefix.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The validation data prefix.")
    parser.add_argument('--test_prefix', type=str, default="test", help="The test data prefix.")
    parser.add_argument('--slot_descs_prefix', type=str, default="slot_descs", help="The slot description file prefix.")
    
    args = parser.parse_args()
    
    cached_dir = f"{args.data_dir}/{args.cached_dir}"

    run(args, cached_dir)
    