if __name__ == "__main__":
    from datasets import load_from_disk
    from collections import defaultdict
    from tqdm import tqdm
    import random
    import numpy as np
    import pandas as pd
    from datasets import Dataset
    import os

    chrono_data_path = '/home/joseph/chrono_data/chrono_train_data'
    chrono_data = load_from_disk(chrono_data_path)

    def gen_geo(epsilon, gamma, save_path):
        curr_batch = 0
        batch = defaultdict(list)
        batch_size = 512
        num_data = len(chrono_data) 
        model_cnts = defaultdict(int)
        random.seed(42)
        
        for i in tqdm(range(num_data)):
            model_a = chrono_data[i]['model_a']
            model_b = chrono_data[i]['model_b']
            p = 1 - min(1 - epsilon, 1 - gamma ** (min(model_cnts[model_a], model_cnts[model_b])))
            
            while len(batch[curr_batch]) == batch_size:
                curr_batch += 1
                
            num = np.random.geometric(p) - 1
            cnt = 0
            while len(batch[curr_batch + num + cnt]) == batch_size:
                cnt += 1
            batch[curr_batch + num + cnt].append(i)

            model_cnts[model_a] += 1
            model_cnts[model_b] += 1

        num_batch = 0
        while True:
            if len(batch[num_batch]) > batch_size:
                raise ArithmeticError
            if len(batch[num_batch]) == batch_size:
                num_batch += 1
            if len(batch[num_batch]) < batch_size:
                num_batch -= 1
                print(num_batch)
                break

        data = []
        for i in tqdm(range(num_batch + 1)):
            for idx in batch[i]:
                data.append(idx)

        geom_train_data = [chrono_data[idx] for idx in tqdm(data)]
        df = pd.DataFrame(geom_train_data)
        geom_dataset = Dataset.from_pandas(df)
        geom_dataset.save_to_disk(save_path)

    gamma = [0.99, 0.995, 0.999, 0.9995, 0.9999]
    eps = [0.01, 0.005, 0.001, 0.0005, 0.0001]

    base = "/home/joseph/chrono_data/"
    redo = False

    for ep in eps:
        for gam in gamma:
            save = f'{base}/geom_gamma_{gam}_eps_{ep}_min'
            if not redo and os.path.exists(save):
                print(f"Already exists: {save}")
                continue
            gen_geo(ep, gam, save)
            print(f"Saved to {save}")