from tqdm import tqdm
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os
import random
from collections import defaultdict

random.seed(42)

def geometric_delay(chrono_data: "pd.DataFrame", model_cnts: dict[str, int], batch_assignments: dict[int, list[str]], batch_size: int, curr_batch: int, gamma: float, epsilon: float, minimum: bool) -> list[str]:
    num_data = len(chrono_data) 
    model_cnts = defaultdict(int)
    
    
    for i in tqdm(range(num_data)):
        model_a = chrono_data[i]['model_a']
        model_b = chrono_data[i]['model_b']
        if minimum:
            p = 1 - min(1 - epsilon, 1 - gamma ** (min(model_cnts[model_a], model_cnts[model_b])))
        else:
            p = 1 - min(1 - epsilon, 1 - gamma ** (model_cnts[model_a] + model_cnts[model_b]))
        
        while len(batch_assignments[curr_batch]) == batch_size:
            curr_batch += 1
            
        num = np.random.geometric(p) - 1
        cnt = 0
        while len(batch_assignments[curr_batch + num + cnt]) == batch_size:
            cnt += 1
        batch_assignments[curr_batch + num + cnt].append(i)

        model_cnts[model_a] += 1
        model_cnts[model_b] += 1

    num_batch = 0
    while True:
        if len(batch_assignments[num_batch]) > batch_size:
            raise ArithmeticError
        if len(batch_assignments[num_batch]) == batch_size:
            num_batch += 1
        if len(batch_assignments[num_batch]) < batch_size:
            num_batch -= 1
            print(num_batch)
            break

    data = []
    for i in tqdm(range(num_batch + 1)):
        for idx in batch_assignments[i]:
            data.append(idx)

    return data

def exponential_delay(counter: int, chrono_data: "pd.DataFrame", model_cnts: dict[str, int], gamma: float, mu: float, curr_tick: int, min_heap: list, minimum: bool) -> tuple[list[str], list]:
    new_data = []
    
    for idx, row in tqdm(chrono_data.iterrows(), total=len(chrono_data)):
        model_a, model_b = row['model_a'], row['model_b']
        model_a_cnt, model_b_cnt = model_cnts.get(model_a, 0), model_cnts.get(model_b, 0)
        if minimum:
            lambda_ = max(mu, gamma**(min(model_a_cnt, model_b_cnt)))
        else:
            lambda_ = max(mu, gamma**(model_a_cnt + model_b_cnt))
        delay = np.random.exponential(scale=1/lambda_)
        new_tick = curr_tick + delay
        curr_tick += 1
        
        heapq.heappush(min_heap, (new_tick, row['question_id']))
        while min_heap and min_heap[0][0] <= curr_tick:
            _, question_id = heapq.heappop(min_heap)
            new_data.append(question_id)
    
    return new_data, min_heap

def plot_model_occurrence_across_batches(qid_sequence: list[str], chrono_data: "pd.DataFrame", target_model: str, batch_size: int, gamma: float, mu: float, minimum: bool, save_dir: str = "plots"):
    batch_counts = []
    batch_numbers = []
    
    qid_to_model = {}
    print("Getting qid to model")
    for idx, row in tqdm(chrono_data.iterrows(), total=len(chrono_data)):
        qid_to_model[row['question_id']] = (row['model_a'], row['model_b'])
    
    for batch_idx in tqdm(range(0, len(qid_sequence), batch_size)):
        batch_qids = qid_sequence[batch_idx:batch_idx + batch_size]
        
        count = 0
        for qid in batch_qids:
            if qid in qid_to_model:
                model_a, model_b = qid_to_model[qid]
                if model_a == target_model or model_b == target_model:
                    count += 1
        
        batch_counts.append(count)
        batch_numbers.append(batch_idx // batch_size)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(batch_numbers, batch_counts, marker='o', linewidth=1, markersize=3)
    
    # Create title with hyperparameters
    min_type = "minimum" if minimum else "sum"
    title = f'Model "{target_model}" Occurrence (γ={gamma}, μ={mu}, {min_type})'
    plt.title(title)
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Occurrences in Batch')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    avg_count = np.mean(batch_counts)
    plt.axhline(y=avg_count, color='red', linestyle='--', alpha=0.7, 
                label=f'Average: {avg_count:.1f}')
    plt.legend()
    
    # Create save directory and filename
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = target_model.replace("/", "_").replace("-", "_")
    filename = f"{clean_model_name}_gamma_{gamma}_mu_{mu}_{min_type}.png"
    filepath = os.path.join(save_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Plot saved to: {filepath}")
    
    return batch_numbers, batch_counts

if __name__ == "__main__":
    batch_size = 512
    
    chrono_data = load_from_disk("seeded_data/chrono_train_data").to_pandas()
    
    # Create main plots directory
    base_save_dir = "delay_analysis_plots"
    os.makedirs(base_save_dir, exist_ok=True)
    
    for gamma in [0.9999, 0.9995, 0.999, 0.995, 0.99]:
        for mu in [0.0001, 0.0005, 0.001]:
            for minimum in [True, False]:
                print(f"\nRunning exponential_delay with γ={gamma}, μ={mu}, minimum={minimum}")
                new_qids, leftover = exponential_delay(
                    counter=0, 
                    chrono_data=chrono_data, 
                    model_cnts={}, 
                    gamma=gamma, 
                    mu=mu, 
                    curr_tick=0, 
                    min_heap=[], 
                    minimum=minimum
                )
                
                # Create subdirectory for this hyperparameter combination
                min_type = "minimum" if minimum else "sum"
                subdir = f"gamma_{gamma}_mu_{mu}_{min_type}"
                save_dir = os.path.join(base_save_dir, subdir)
                
                for model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "chatgpt-4o-latest-20240808"]:
                    plot_model_occurrence_across_batches(
                        new_qids, 
                        chrono_data, 
                        model, 
                        batch_size, 
                        gamma, 
                        mu, 
                        minimum, 
                        save_dir
                    )