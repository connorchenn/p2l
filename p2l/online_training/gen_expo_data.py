from datasets import load_from_disk
from collections import defaultdict
from tqdm import tqdm
import json
import os
import numpy as np
import argparse
import pandas as pd
from datasets import Dataset
import heapq


def generate_expo(chrono_train_data, batch_size, gamma, mu, tstamp_file, use_min, save_data, save_batch_every, save_folder, step_size):
    get_tstamp = bool(tstamp_file)
    
    if save_batch_every is not None:
        batch_tracker = {}
        batch_tracker_list = []
                
    if use_min:
        save_path = f'{save_folder}/expo_gamma_{gamma}_mu_{mu}_step_{step_size}_min'
    else:
        save_path = f'{save_folder}/expo_gamma_{gamma}_mu_{mu}_step_{step_size}'
        
    np.random.seed(42)

    chrono_data = load_from_disk(chrono_train_data)
    batch = defaultdict(list)
    num_data = len(chrono_data) 
    model_cnts = defaultdict(int)
    
    if get_tstamp:
        df = pd.read_json(tstamp_file, lines=True)
        qid_ts = df.set_index('question_id')['tstamp'].to_dict()
        tstamps = defaultdict(dict)

    curr_batch = 0
    full_batch = 0
    curr_tick = 0
    min_heap = []
    processed_count = 0
    
    for i in tqdm(range(num_data), desc="Generating delays and creating batches"):
        model_a = chrono_data[i]['model_a']
        model_b = chrono_data[i]['model_b']
        
        model_a_cnt = model_cnts.get(model_a, 0)
        model_b_cnt = model_cnts.get(model_b, 0)
        
        if use_min:
            n = min(model_a_cnt, model_b_cnt)
        else:
            n = model_a_cnt + model_b_cnt
        
        p = max(mu, gamma ** n)
        p = np.clip(p, None, 1-1e-10)
        
        lambda_ = -np.log(1 - p)
        delay = np.random.exponential(scale=1/lambda_)
        new_tick = curr_tick + delay
        curr_tick += step_size
        
        heapq.heappush(min_heap, (new_tick, i))
        model_cnts[model_a] += 1
        model_cnts[model_b] += 1
        
        while(min_heap and min_heap[0][0] <= curr_tick):
            _, original_idx = heapq.heappop(min_heap)
            
            while len(batch[curr_batch]) == batch_size:
                curr_batch += 1
                
            if get_tstamp:
                tstamps[curr_batch]['end'] = processed_count
                if 'start' not in tstamps[curr_batch]:
                    tstamps[curr_batch]['start'] = processed_count
                
            batch[curr_batch].append(original_idx)
        
            processed_count += 1
            
        while len(batch[full_batch]) == batch_size:
            full_batch += 1
            
        if save_batch_every is not None:
            if ((i + 1) % (save_batch_every * batch_size) == 0) or (i == num_data - 1):
                batch_tracker[(i+1) // batch_size] = full_batch
                batch_tracker_list.append(full_batch)
    
    end_idx = 0
    while True:
        assert len(batch[end_idx]) <= batch_size
        if len(batch[end_idx]) == batch_size:
            end_idx += 1
        if len(batch[end_idx]) < batch_size:
            end_idx -= 1
            break
        
    if save_batch_every is not None:
        tracker = {}
        tracker['dict'] = batch_tracker
        tracker['list'] = batch_tracker_list
        
        with open(f'{save_path}_batch_tracker.json', 'w') as file:
            json.dump(tracker, file)
    
    if get_tstamp:
        for batch_idx, times in tstamps.items():
            # Get original indices from the batch
            start_original_idx = batch[batch_idx][0] if batch[batch_idx] else 0
            end_original_idx = batch[batch_idx][-1] if batch[batch_idx] else 0
            
            times['start'] = qid_ts[chrono_data[start_original_idx]['question_id']]
            times['end'] = qid_ts[chrono_data[end_original_idx]['question_id']]
        
        os.makedirs("train_timestamps", exist_ok=True)
        with open(f'train_timestamps/{save_path}_tstamps.json', 'w') as file:
            json.dump(tstamps, file)
    
    if save_data:
        data = []
        for i in tqdm(range(end_idx+1), desc="Collecting final data"):
            for idx in batch[i]:
                data.append(idx)
                
        expo_train_data = [chrono_data[idx] for idx in tqdm(data, desc="Creating dataset")]
        expo_dataset = Dataset.from_pandas(pd.DataFrame(expo_train_data))
        expo_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chrono-train-data", type=str, default='/tmp/chrono_train_data'
    )
    parser.add_argument(
        "--batch-size", type=int, default=512
    )
    parser.add_argument(
        "--gamma", type=float, required=True
    )
    parser.add_argument(
        "--mu", type=float, required=True, help="minimum probability parameter for exponential delay"
    )
    parser.add_argument(
        "--tstamp-file", type=str, default=None, help="save start and end timestamps if true"
    )
    parser.add_argument(
        "--use-min", action="store_true", help="use min instead of sum"
    )
    parser.add_argument(
        "--no-save-data", 
        action="store_false", 
        dest="save_data", 
        help="disables saving the dataset if set"
    )
    parser.add_argument(
        "--save-batch-every", 
        type=int, 
        help="store the last full batch appearance for each args.save_batch_every batches (for evaluation)"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="folder to save the dataset"
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="step size for time progression"
    )

    args = parser.parse_args()
    generate_expo(args.chrono_train_data, args.batch_size, args.gamma, args.mu, args.tstamp_file, args.use_min, args.save_data, args.save_batch_every, args.save_folder, args.step_size) 