from datasets import load_from_disk
from collections import defaultdict
from tqdm import tqdm
import json
import os
import numpy as np
import argparse
import pandas as pd
from datasets import Dataset



def generate_geom(chrono_train_data, batch_size, gamma, eps, tstamp_file, use_min, save_data, save_batch_every, save_folder):
    get_tstamp = bool(tstamp_file)
    
    if save_batch_every is not None:
        batch_tracker = {}
        batch_tracker_list = []
                
    if use_min:
        save_path = f'{save_folder}/geom_gamma_{gamma}_eps_{eps}_min'
    else:
        save_path = f'{save_folder}/geom_gamma_{gamma}_eps_{eps}'
        
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
    
    for i in tqdm(range(num_data)):
        model_a = chrono_data[i]['model_a']
        model_b = chrono_data[i]['model_b']
        #p = success = non defered 
        if use_min:
            p = 1 - min(1-eps, 1-gamma**(min(model_cnts[model_a], model_cnts[model_b])))
        else:
            p = 1 - min(1-eps, 1-gamma**(model_cnts[model_a] + model_cnts[model_b]))
        
        while len(batch[curr_batch]) == batch_size:
            curr_batch += 1
            
        num = np.random.geometric(p) - 1 
        cnt = 0
            
        
        if get_tstamp:
            tstamps[curr_batch]['end'] = i
            if 'start' not in tstamps[curr_batch]:
                tstamps[curr_batch]['start'] = i
            
        while (len(batch[curr_batch + num + cnt]) == batch_size):
            cnt += 1
            
        batch[curr_batch + num + cnt].append(i)  
        
        #if we are at a checkpoint, save which is the lastest batch appearance (for checkpointing during training)
        
        #find latest full batch
        while len(batch[full_batch]) == batch_size:
            full_batch += 1
        
        if ((i + 1) % (save_batch_every * batch_size) == 0) or (i == num_data - 1):
            batch_tracker[(i+1) // batch_size] = full_batch
            batch_tracker_list.append(full_batch)
        
        if (num + cnt > 0) and get_tstamp:
            tstamps[curr_batch + num + cnt]['end'] = i
            if 'start' not in tstamps[curr_batch + num + cnt]:
                tstamps[curr_batch + num + cnt]['start'] = i

        model_cnts[model_a] += 1
        model_cnts[model_b] += 1
    
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
            times['start'] = qid_ts[chrono_data[times['start']]['question_id']]
            times['end'] = qid_ts[chrono_data[times['end']]['question_id']]
        
        os.makedirs("train_timestamps", exist_ok=True)
        with open(f'train_timestamps/{save_path}_tstamps.json', 'w') as file:
            json.dump(tstamps, file)
    
    if save_data:
        data = []
        for i in tqdm(range(end_idx+1)):
            for idx in batch[i]:
                data.append(idx)
                
        geom_train_data = [chrono_data[idx] for idx in tqdm(data)]
        geom_dataset = Dataset.from_pandas(pd.DataFrame(geom_train_data))
        geom_dataset.save_to_disk(save_path)


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
        "--eps", type=float, required=True
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

    args = parser.parse_args()
    generate_geom(args.chrono_train_data, args.batch_size, args.gamma, args.eps, args.tstamp_file, args.use_min, args.save_data, args.save_batch_every, args.save_folder)