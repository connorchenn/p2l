import argparse
from p2l.model import get_p2l_model, P2LOutputs
from transformers import pipeline, TextClassificationPipeline, AutoTokenizer
from huggingface_hub import snapshot_download
from datasets import Dataset
import torch
from typing import Dict
import pandas as pd
import os
import json
import re
from tqdm.auto import tqdm
from torch.utils.data import Dataset as TorchDataset
from glob import glob
from p2l.eval import P2LPipeline, ListDataset

def extract_number_from_path(path, prefix):
    pattern = re.compile(f"{prefix}(\\d+)")
    match = pattern.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def get_sorted_files_with_numbers(directory, prefix):
    files = glob(os.path.join(directory, f"*"))
    result = []
    
    for file in files:
        num = extract_number_from_path(file, prefix)
        if num is not None:
            result.append((file, num))
    
    return sorted(result, key=lambda x: x[1])

def load_csv_dataset(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    local_repo_path = snapshot_download(
        repo_id=args.model_repo
    )
    
    checkpoints_with_nums = get_sorted_files_with_numbers(local_repo_path, args.checkpoint_prefix)
    
    val_files_with_nums = get_sorted_files_with_numbers(args.val_dir, "")
    
    val_datasets = []
    for val_file, val_num in val_files_with_nums:
        dataset = load_csv_dataset(val_file)
        val_datasets.append((dataset, val_num))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_list_path = os.path.join(local_repo_path, "model_list.json")

    with open(model_list_path) as file:
        model_list = json.load(file)
        
    if args.time_align:
        cp_frequency = checkpoints_with_nums[0][1]
        assert cp_frequency == checkpoints_with_nums[1][1] - checkpoints_with_nums[0][1]
    
    for checkpoint_dir, checkpoint_num in checkpoints_with_nums:
        checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint-{checkpoint_num}")
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        model_cls = get_p2l_model(args.model_type, args.loss_type, args.head_type)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        
        model = model_cls.from_pretrained(
            checkpoint_dir,
            CLS_id=tokenizer.cls_token_id,
            num_models=len(model_list),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation='flash_attention_2'
        )
        model.to(device)

        pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            pipeline_class=P2LPipeline,
        )
        
        if args.time_align:
            with open(args.train_time, 'r') as file: 
                train_times = json.load(file)
            with open(args.val_time, 'r') as file:
                val_times = json.load(file)
                
            begin_batch = checkpoint_num - (checkpoint_num % cp_frequency)
            if begin_batch == checkpoint_num:
                begin_batch -= cp_frequency
                
            largest_train_time = max(train_times[str(i)]['end_tstamp'] for i in range(begin_batch, checkpoint_num))
            val_sets = [(dataset, val_num) for dataset, val_num in val_datasets if val_times[str(val_num)]['end_tstamp'] < largest_train_time]
        else: 
            val_sets = [(dataset, val_num) for dataset, val_num in val_datasets if val_num <= checkpoint_num]
        
        for dataset, val_num in val_sets: 
            output_file = os.path.join(
                checkpoint_output_dir, 
                f"checkpoint-{checkpoint_num}-val-{val_num}.json"
            )
            
            if "prompt" in dataset.column_names:
                prompts = ListDataset(dataset["prompt"])
            else:
                print(f"ERROR: Dataset {val_num} does not have a 'prompt' column")

            with torch.no_grad():
                outputs = [
                    out
                    for out in tqdm(
                        pipe(prompts, 
                            batch_size=args.batch_size), 
                            total=len(prompts), 
                            desc = f"Checkpoint-{checkpoint_num}-val-{val_num}"
                    )
                ]

            df = dataset.to_pandas()
            outputs_df = pd.DataFrame.from_records(outputs)
            
            if "last_hidden_state" in outputs_df.columns:
                outputs_df = outputs_df.drop(columns=["last_hidden_state"])
                
            df = pd.concat((df, outputs_df), axis=1)

            df.to_json(output_file, orient="records", indent=4, force_ascii=False)
            print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-repo", "-m", type=str, required=True, 
        help="Huggingface model repository"
    )
    parser.add_argument(
        "--val-dir", "-v", type=str, required=True,
        help="Directory containing validation CSV files"
    )
    parser.add_argument(
        "--checkpoint-prefix", "-cp", type=str, default="checkpoint-",
        help="Prefix for checkpoint directories"
    )
    parser.add_argument(
        "--model-type", "-mt", type=str, default="qwen2",
        help="Model type (qwen2, llama, etc)"
    )
    parser.add_argument(
        "--head-type", "-ht", type=str, default="bt",
        help="Head type (Bradely Terry, Rao-Kupper, etc)"
    )
    parser.add_argument(
        "--loss-type", "-lt", type=str, default="bt",
        help="Loss type (Bradely Terry, Rao-Kupper, etc)"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=1, 
        help="Batch size"
    )
    parser.add_argument(
        "--output-dir", "-od", type=str, default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--time-align", action="store_true", help="include for time aligned evals"
    )
    parser.add_argument(
        "--train-time", type=str, help="file containing last times for each batch in train file"
    )
    parser.add_argument(
        "--val-time", type=str, help="file containing last times for each batch in val file"
    )

    args = parser.parse_args()

    if args.time_align and (not args.train_time or not args.val_time):
        parser.error("--train-time and --val-time required when --time-align is set.")
    
    main(args)