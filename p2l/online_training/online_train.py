import argparse
import os
import yaml
import json
import random
import torch
import deepspeed
import wandb
import torch.distributed as dist
from transformers import set_seed
from dataset import DataCollator, get_dataset, generate_online_labels
from model import get_p2l_model, get_tokenizer
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import datetime

def train_model(args):
    local_checkpoint_path = None
    config = None

    if args.checkpoint:
        print(f"Downloading checkpoint from Hugging Face hub: {args.checkpoint}")
        local_checkpoint_path = snapshot_download(repo_id=args.checkpoint)
        
        training_config_path = os.path.join(local_checkpoint_path, "training_config.json")
        
        with open(training_config_path) as f:
            config = json.load(f)
    else:
        with open(args.config_path) as f:
            config = yaml.safe_load(f)

    #project name
    proj_name = config["proj_name"]
    #get optimizer config
    optimizer_config = config["optimizer"]
    # Microbatch size
    batch_size = config["batch_size"]
    # HF data path
    train_data_path = config["train_data_path"]
    val_data_path = config["val_data_path"]
    output_dir = config["output_dir"]
    #get repo id
    pretrain_model_name = config["pretrain_model_name"]
    # Prompts will be truncted to this length
    max_length = config["max_length"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    # Deepspeed config choices can be found in the deepspeed directory
    deepspeed_config_path = config["deepspeed_config_path"]
    # Type of transformer, see model.py for options.
    model_type = config["model_type"]
    # Loss type (e.g, bt, rk), see model.py for options.
    loss_type = config["loss_type"]
    # The linear head type, see model.py for options.
    head_type = config["head_type"]

    #get wandb entity for logging
    wandb_entity = config.get("wandb_entity", None)
    chat_template = config.get("chat_template", None)
    # Downsize the rank of the classification head.
    linear_head_downsize_factor = config.get("linear_head_downsize_factor", None)
    # Whether to weight the loss. If this is true, it expects that the dataset has a "weight" column.
    weighted_loss = config.get("weighted_loss", False)
    # If the tokenizer/model does not already have a cls token, this will be used.
    cls_token_if_none = config.get("cls_token_if_none", "<|cls|>")
    # If the tokenizer/model does not already have a pad token, this will be used.
    pad_token_if_none = config.get("pad_token_if_none", "<|pad|>")
    # If using weighted loss, scalar reweight factor
    reweight_scale = config.get("reweight_scale", None)
    init_type = config.get("init_type", "reset_params")
    train_head_only = config.get("train_head_only", False)
    load_train_data_from_disk = config.get("load_train_data_from_disk", False)
    load_val_data_from_disk = config.get("load_val_data_from_disk", False)

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, proj_name)
    
    if not local_checkpoint_path:
        version = 1
        while os.path.exists(output_path):
            output_path = output_path.replace(f"_{version - 1}", "")
            output_path = output_path + f"_{version}"
            version += 1

    random.seed(42)
    set_seed(42)

    tokenizer = get_tokenizer(
        pretrain_model_name,
        chat_template,
        pad_token_if_none=pad_token_if_none,
        cls_token_if_none=cls_token_if_none,
    )

    train_data = get_dataset(
        train_data_path, "train", from_disk=load_train_data_from_disk
    )
    val_data = get_dataset(val_data_path, "train", from_disk=load_val_data_from_disk)
    
    if LOCAL_RANK <= 0:
        os.makedirs(output_path, exist_ok=False)

        with open(os.path.join(output_path, "training_config.json"), "w") as fout:
            json.dump(config, fout, indent=1)

        if wandb_entity:
            run = wandb.init(
                entity=wandb_entity,
                project='Online Training',
                name=proj_name,
                config=config
            )

    model_cls = get_p2l_model(
        model_type = model_type,
        loss_type = loss_type,
        head_type = head_type,
        init_type = init_type,
    )

    if local_checkpoint_path:
        print(f"Loading model from local checkpoint: {local_checkpoint_path}")
        model = model_cls.from_pretrained(
            local_checkpoint_path,
            CLS_id=tokenizer.cls_token_id,
            num_models=1000,
            linear_head_downsize_factor=linear_head_downsize_factor,
        )
        
        with open(os.path.join(local_checkpoint_path, "model_list.json"), "r") as f:
            model_list = json.load(f)
    else:
        model = model_cls.from_pretrained(
            pretrain_model_name,
            CLS_id=tokenizer.cls_token_id,
            num_models=1000,
            linear_head_downsize_factor=linear_head_downsize_factor,
        )

        model_list = []

    data_collator = DataCollator(
        tokenizer, max_length, weight=weighted_loss, reweight_scale=reweight_scale, include_labels=False
    )
        

    with open(deepspeed_config_path) as f:
        deepspeed_config = json.load(f)  

    #add optimizer, batch size, and grad accumulation to deepspeed config
    deepspeed_config["optimizer"] = optimizer_config
    deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size
    deepspeed_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    if model.config.vocab_size < len(tokenizer):
        print("WARNING: Resizing Token Embedding")
        model.resize_token_embeddings(len(tokenizer))

    if train_head_only:
        print("Freezing transformer, only training head.")
        model.freeze_transformer()

    train_data.set_format("torch")
    val_data.set_format("torch")

    val_dataloader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: x)

  

    train_dataloader = DataLoader(
        train_data,  
        batch_size=batch_size, 
        shuffle=True,   
        collate_fn=lambda x: x) 

    if local_checkpoint_path:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )
        model_engine.load_checkpoint(load_dir=local_checkpoint_path, load_optimizer_states=True)
    else:
        model_engine, optimizer, _, _ = deepspeed.initialize( 
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config,
        )
        
    train_loss = 0.0
    data_cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_engine.to(device)
    model_engine.train()
    for i, batch_data in enumerate(tqdm(train_dataloader)):
        data = data_collator(batch_data)

        #manually create the labels
        labels = generate_online_labels(model_list, batch_data)
        outputs = model_engine(data['input_ids'].to(device), data['attention_mask'].to(device), labels=labels.to(device))
        model_engine.backward(outputs.loss)
        train_loss += outputs.loss.cpu().item() * len(batch_data)
        data_cnt += len(batch_data)

        model_engine.step() 

        if model_engine.is_gradient_accumulation_boundary():
            local_loss = torch.tensor([train_loss], device=device)
            local_count = torch.tensor([data_cnt], device=device)

            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            
            global_loss = local_loss.item() / local_count.item()

            if model_engine.local_rank == 0:
                grads = [param.grad.detach().flatten() for param in model_engine.parameters() if param.grad is not None]
                grad_norm = torch.cat(grads).norm()


                print(f" Step: {i}, Global Train Loss: {global_loss}, Grad Norm: {grad_norm}")

                if wandb_entity:
                    run.log({"train/loss": global_loss, "train/step": i, "train/grad_norm": grad_norm})

            train_loss = 0.0
            data_cnt = 0
        
    if not args.no_eval:
        print("begin eval")
        model_engine.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader):
                data = data_collator(batch_data)
                labels = generate_online_labels(model_list, batch_data)

                outputs = model_engine(data['input_ids'].to(device), data['attention_mask'].to(device), labels=labels.to(device))
                outputs.loss = outputs.loss.cpu().item()
                val_loss += outputs.loss * len(batch_data)

        local_loss = torch.tensor([val_loss], device=device)
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

        if model_engine.local_rank == 0:
            global_loss = local_loss / len(val_data)
            print(f"Val Loss: {global_loss}")

    # Save model and optimizer states
    model_engine.save_checkpoint(output_path)
    
    if LOCAL_RANK <= 0:
        if wandb_entity:
            run.finish()

        with open(os.path.join(output_path, "model_list.json"), "w") as fout:
            json.dump(model_list, fout, indent=1)

        if args.push_to_hf:
            api = HfApi()
            # Add current date to repo name
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            repo_id = config.get("repo_id", f"p2el/{proj_name}-{today_str}")
            assert not api.repo_exists(
                repo_id=repo_id, repo_type="model"
            ), "repo already exists"

            api.create_repo(repo_id=repo_id, private=True, repo_type="model")
            api.upload_folder(
                folder_path=output_path,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=None  # include deepspeed
            )
            print("pushed to hub")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to checkpoint directory to resume training from",
        default=None,
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed"
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        default=True,
        help="True if push directly to huggingface",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="If flagged eval will not end at end of training loop.",
    )
    args = parser.parse_args()
    
    train_model(args)