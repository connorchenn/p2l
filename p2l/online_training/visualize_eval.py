import json
import torch
import ast
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('..')
from p2l.auto_eval_utils import registered_simple_metrics, registered_helpers
from p2l.model import HeadOutputs
from datasets import load_from_disk, Dataset
from tqdm import tqdm

process_head_output = registered_helpers["p2l"]["output_labels"]
loss_func = registered_simple_metrics["bag"]["Loss"]

def convert_to_list(label_str):
    if isinstance(label_str, str):
        cleaned = label_str.strip('[]').split()
        return [int(x) for x in cleaned if x]
    elif isinstance(label_str, list):
        return label_str
    else:
        return []
    
def parse_eval_output_data(
    path,
    loss_type,
):
    df = pd.read_json(path)
    df = df.rename(columns={"coefs": "betas"})

    df["eta"] = df["eta"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df["betas"] = df["betas"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df['labels'] = df['labels'].apply(convert_to_list)
    ret_df = df

    preprocess_func = registered_helpers[loss_type]["preprocess_data"]
    ret_df = preprocess_func(data=ret_df)
    return ret_df

def extract_loss(base, output_file):
    model_cp_results = {}
    for sub_path in tqdm(os.listdir(base)):
        checkpoint_folder = os.path.join(base, sub_path)
        if os.path.isdir(checkpoint_folder):
            cp_num = int(sub_path.split('-')[-1])
            model_cp_results[cp_num] = {}
            for base_file in os.listdir(checkpoint_folder):
                dataset_num = int(base_file.split('.')[0].split('-')[-1])    
                file_path = os.path.join(checkpoint_folder, base_file)
                df = parse_eval_output_data(file_path, "bag")
                head_out, labels = process_head_output(df)
                loss = loss_func(head_out, labels, "bag")
                model_cp_results[cp_num][dataset_num] = loss
            
    with open(output_file, 'w') as file:
        json.dump(model_cp_results, file)

def extract_accuracy(base, output_file):
    model_cp_results = {}
    # base = '/tmp/output_eps_0.016'
    for sub_path in tqdm(os.listdir(base)):
        checkpoint_folder = os.path.join(base, sub_path)
        if os.path.isdir(checkpoint_folder):
            cp_num = int(sub_path.split('-')[-1])
            model_cp_results[cp_num] = {}
            for base_file in os.listdir(checkpoint_folder):
                dataset_num = int(base_file.split('.')[0].split('-')[-1])    
                file_path = os.path.join(checkpoint_folder, base_file)
                df = parse_eval_output_data(file_path, "bag")
                head_out, labels = process_head_output(df)
                loss = accuracy_func(head_out, labels, "bag")
                model_cp_results[cp_num][dataset_num] = loss
            

    with open(output_file, 'w') as file:
        json.dump(model_cp_results, file)
        
        
        
def plot_cp_results(path, save_dir, model_name, title):
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, model_name)

    with open(path, 'r') as file:
        data = json.load(file)

    model_ckpts = sorted(int(k) for k in data.keys())
    dataset_ckpts = sorted(
        {int(inner_k) for inner_dict in data.values() for inner_k in inner_dict.keys()}
    )

    df = pd.DataFrame(index=model_ckpts, columns=dataset_ckpts, dtype=float)

    for model_size, datasets in data.items():
        for dataset_size, loss in datasets.items():
            df.loc[int(model_size), int(dataset_size)] = loss

    mask = df.isna()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="viridis_r",
        mask=mask, cbar_kws={'label': 'Loss'}
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Dataset Checkpoint', fontsize=14)
    plt.ylabel('Model Checkpoint', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_heatmap.png")
    plt.close()

    # Line plot
    plt.figure(figsize=(14, 8))
    for model_size in df.index:
        model_data = df.loc[model_size].dropna()
        plt.plot(model_data.index, model_data.values, 'o-', label=f'Model {model_size}')

    plt.xlabel('Dataset Checkpoint', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_lineplot.png")
    plt.close()
    print(f"Saved plot to {save_prefix}_lineplot.png and {save_prefix}_heatmap.png")

def visualize_eval(models, base, eval_plot_folder, accuracy):
    os.makedirs(eval_plot_folder, exist_ok=True)
    for model_info in models:
        if type(model_info) == str:
            model_eval = model_info
            title = f"Validation Loss Across Model and Dataset Checkpoints for {model_eval.split('/')[-1]}"
        elif len(model_info) == 2:
            model_eval, eps = model_info
            title = f'Replay Buffer (ε = {eps}) Validation Loss Across Model and Dataset Checkpoints'
        elif len(model_info) == 3:
            model_eval, gamma, sigma = model_info
            title = f'Geometric Buffer (γ = {gamma}, σ = {sigma}) Validation Loss Across Model and Dataset Checkpoints'
        elif len(model_info) == 4:
            model_eval, gamma, mu, step_size = model_info
            title = f'Exponential Buffer (γ = {gamma}, μ = {mu}, step size = {step_size}) Validation Loss Across Model and Dataset Checkpoints'
        else:
            raise ValueError(f"Invalid model info: {model_info}")
        if accuracy:
            data_base = os.path.join(base, 'accuracy')
        else:
            data_base = os.path.join(base, 'loss')
            
        os.makedirs(data_base, exist_ok=True)
        model_name = model_eval.split('/')[-1]
        model_file = f"{model_name}.json"
        output_file = os.path.join(data_base, model_file)
        if not os.path.exists(output_file):
            f'output file {output_file} not found'
            if accuracy:
                extract_accuracy(model_eval, output_file)
            else:
                extract_loss(model_eval, output_file)

        plot_cp_results(output_file, eval_plot_folder, model_name, title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', default=["outputs"], help='list of model directories to visualize')
    parser.add_argument("--base", type=str, default='outputs', help='base directory to save loss results')
    parser.add_argument("--eval-folder", type=str, default='eval_plots2', help='folder to save eval plots')
    parser.add_argument("--extract-accuracy", action='store_true')
    args = parser.parse_args()
    
    visualize_eval(args.models, args.base, args.eval_folder, args.extract_accuracy)