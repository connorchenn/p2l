import os
import json
from datasets import load_from_disk
from tqdm import tqdm
import argparse
import pandas as pd


def gen_tstamps(train_data, timestamps_file, batch_size, output_file):
    train_data = load_from_disk(train_data)
    df = pd.read_json(timestamps_file, lines=True)
    qid_ts = df.set_index('question_id')['tstamp'].to_dict()
    tstamps = {}
    largest = -float('inf')
    batch = 0
    for i in tqdm(range(0, len(train_data), batch_size)):
        for batch_idx in range(i, min(i + batch_size, len(train_data))):
            largest = max(largest, qid_ts[train_data[batch_idx]['question_id']])
        
        tstamps[batch] = {'end_tstamp': largest}
        batch += 1
        
    os.makedirs("timestamps", exist_ok=True)
    with open(f'timestamps/{output_file}', 'w') as file:
        json.dump(tstamps, file, indent=4)
            
    print("tstamps file saved")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chrono-train-data", type=str, default='/tmp/chrono_train_data'
    )
    parser.add_argument(
        "--tstamp-file", type=str, default='/tmp/quid_to_tstamp.jsonl'
    )
    parser.add_argument(
        "--batch-size", type=int, default=512
    )
    parser.add_argument(
        "--output-file", type=str, required=True
    )
    

    args = parser.parse_args()
    gen_tstamps(args.chrono_train_data, args.tstamp_file, args.batch_size, args.output_file)