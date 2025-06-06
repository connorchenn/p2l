import argparse
import gradio as gr
from model import get_tokenizer, get_p2l_model, HeadOutputs
from dataset import DataCollator
import json
import pandas as pd
from huggingface_hub import hf_hub_download
import yaml
import os
import torch
import time
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from auto_eval_utils import registered_helpers


def get_leaderboard(model, data_collator, model_list, prompt):
    processed_data = data_collator(prompt)
    device = next(model.parameters()).device
    out = model(processed_data['input_ids'].to(device), processed_data['attention_mask'].to(device))
    beta = out.coefs[0].cpu().detach().numpy()
    eta = None
    if out.eta:
        eta = out['eta'][0].cpu().detach().item()
    
    return beta, eta

def get_aggregated_leaderboard(model, data_collator, model_list, prompts: list[str], model_type="p2l", loss_type="bt"):
    device = next(model.parameters()).device
    all_coefs = []
    all_etas = []
    
    for prompt in prompts:
        data = [{"prompt": prompt, "labels": torch.tensor([0, 1])}]
        betas, eta = get_leaderboard(model, data_collator, model_list, data)
        betas = torch.tensor(betas)
        all_coefs.append(betas)
        all_etas.append(eta)
        

    all_coefs_tensor = torch.stack(all_coefs)  # shape: (num_prompts, num_models)
    all_etas = torch.tensor(all_etas)
    output = HeadOutputs(coefs=all_coefs_tensor, eta=all_etas)

    # Dummy labels for compatibility
    labels = torch.tensor([[0, 1, 0]] * len(prompts))  

    aggr_func = registered_helpers["p2l"]["aggregrate"]
    assert loss_type == "bag"
    leaderboard = aggr_func(
        head_output=output,
        labels=labels,
        model_list=model_list,
        loss_type=loss_type
    )

    return dict(sorted({
        model_list[i]: leaderboard[i].item() for i in range(len(model_list))
    }.items(), key=lambda kv: kv[1], reverse=True))

def display_leaderboard(model_path: str, inc_models=None):
    print("Loading model and configuration...")
    local_repo_path = snapshot_download(
        repo_id=model_path
    )
    
    model_list_path = os.path.join(local_repo_path, "model_list.json")
    
    with open(model_list_path) as fp:
        model_list = json.load(fp)
    
    model_config_path = hf_hub_download(model_path, filename="training_config.json")
    with open(model_config_path) as fp:
        model_config = yaml.safe_load(fp)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_collator = DataCollator(tokenizer, model_config["max_length"])

    P2LModel = get_p2l_model(
        model_config["model_type"],
        model_config["loss_type"],
        model_config["head_type"],
    )
    model = P2LModel.from_pretrained(
        model_path,
        CLS_id=tokenizer.cls_token_id,
        num_models=len(model_list),
        local_files_only=True,
    )
    
    model.eval()
    
    print("Model loaded successfully!")

    def _rank_single(prompt: str, progress=gr.Progress()):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt before submitting.")
        
        progress(0.3, desc="Processing prompt...")
        time.sleep(0.5)  # Add slight delay for better UX
        
        try:
            data = [{"prompt": prompt, "labels": torch.tensor([0, 1])}]
            beta, _ = get_leaderboard(model, data_collator, model_list, data)
            leaderboard = model_rankings = {
                model_list[i]: beta[i].item() for i in range(len(beta))
            }
            ranking = dict(sorted(leaderboard.items(), key=lambda kv: kv[1], reverse=True))

            progress(0.7, desc="Generating rankings...")
            
            df = pd.DataFrame(
                {
                    "Rank": range(1, len(ranking) + 1),
                    "Model": list(ranking.keys()),
                    "Score": [f"{v:.4f}" for v in ranking.values()],
                }
            )
            if inc_models is not None:
                df = df[df["Model"].isin(inc_models)]
                
            progress(1.0, desc="Done!")
            return df, "Successfully ranked models based on your prompt."
            
        except Exception as e:
            raise gr.Error(f"An error occurred: {str(e)}")

    def _rank_multiple(prompts: str, progress=gr.Progress()):
        if not prompts.strip():
            raise gr.Error("Please enter at least one prompt.")
        
        # Split prompts by newline and remove empty lines
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            raise gr.Error("Please enter at least one valid prompt.")
            
        progress(0.2, desc="Processing prompts...")
        

        leaderboard = get_aggregated_leaderboard(
            model=model,
            data_collator=data_collator,
            model_list=model_list,
            prompts=prompt_list,
            model_type=model_config["model_type"],
            loss_type=model_config["loss_type"]
        )

        progress(0.7, desc="Generating aggregated rankings...")
        
        df = pd.DataFrame(
            {
                "Rank": range(1, len(leaderboard) + 1),
                "Model": list(leaderboard.keys()),
                "Aggregated Score": [f"{v:.4f}" for v in leaderboard.values()],
            }
        )
        if inc_models is not None:
            df = df[df["Model"].isin(inc_models)]
            
        progress(1.0, desc="Done!")
        return (
            df,
            f"Successfully ranked models based on {len(prompt_list)} prompts.",
            gr.update(visible=True),
            f"Number of prompts processed: {len(prompt_list)}"
        )


    # Example prompts
    single_example_prompts = [
        ["Write a Python function to calculate the Fibonacci sequence."],
        ["Explain the concept of quantum entanglement to a high school student."],
        ["Create a creative story about a time-traveling archaeologist."],
    ]

    multi_example_prompts = [
        ["Write a function to sort a list\nImplement binary search\nCreate a linked list class"],
        ["Explain photosynthesis\nDescribe the water cycle\nHow does the immune system work"],
    ]

    custom_css = """
    .container { max-width: 900px; margin: auto; padding: 20px; }
    .output-panel { margin-top: 20px; }
    .tab-content { padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 10px; }
    .prompt-input { margin-bottom: 15px; }
    .info-text { font-size: 0.9em; color: #666; margin: 10px 0; }
    """

    with gr.Blocks(title="AI Model Ranking System", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎯 AI Model Ranking System
            
            Evaluate and compare AI models using either single prompts or multiple prompts for aggregated analysis.
            """
        )

        with gr.Tabs() as tabs:
            # Single Prompt Tab
            with gr.Tab("Single Prompt Evaluation", id=1):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown(
                        """
                        ### Single Prompt Evaluation
                        Enter a prompt to see how different models would perform on this specific task.
                        """
                    )
                    single_prompt = gr.Textbox(
                        label="Your Prompt",
                        placeholder="Enter a task or question...",
                        lines=4,
                        elem_classes="prompt-input"
                    )
                    with gr.Row():
                        single_submit = gr.Button("🔍 Rank Models", variant="primary", scale=2)
                        single_clear = gr.Button("🔄 Clear", variant="secondary", scale=1)
                    
                    gr.Examples(
                        examples=single_example_prompts,
                        inputs=single_prompt,
                        label="Try these examples"
                    )
                    
                    single_message = gr.Markdown("Enter a prompt above to see rankings")
                    single_table = gr.Dataframe(
                        headers=["Rank", "Model", "Score"],
                        label="Model Rankings"
                    )

            # Multiple Prompts Tab
            with gr.Tab("Multiple Prompts Evaluation", id=2):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown(
                        """
                        ### Multiple Prompts Evaluation
                        Enter multiple prompts (one per line) to get aggregated model rankings.
                        This method provides a more comprehensive evaluation across different tasks.
                        """
                    )
                    multi_prompt = gr.Textbox(
                        label="Your Prompts",
                        placeholder="Enter multiple prompts, one per line...",
                        lines=8,
                        elem_classes="prompt-input"
                    )
                    with gr.Row():
                        multi_submit = gr.Button("🔍 Get Aggregated Rankings", variant="primary", scale=2)
                        multi_clear = gr.Button("🔄 Clear", variant="secondary", scale=1)
                    
                    gr.Examples(
                        examples=multi_example_prompts,
                        inputs=multi_prompt,
                        label="Try these examples"
                    )
                    
                    multi_message = gr.Markdown("Enter prompts above to see aggregated rankings")
                    multi_table = gr.Dataframe(
                        headers=["Rank", "Model", "Aggregated Score"],
                        label="Aggregated Model Rankings"
                    )
                    prompt_count = gr.Markdown(visible=False)

        gr.Markdown(
            """
            ### 📝 Notes
            - Higher scores indicate better predicted performance
            - Rankings are relative and context-dependent
            - Multiple prompt evaluation provides more robust results
            """
        )

        # Set up component interactions
        single_submit.click(
            fn=_rank_single,
            inputs=single_prompt,
            outputs=[single_table, single_message],
        )
        single_clear.click(
            fn=lambda: (None, "Enter a prompt to see rankings."),
            outputs=[single_table, single_message],
        )

        multi_submit.click(
            fn=_rank_multiple,
            inputs=multi_prompt,
            outputs=[multi_table, multi_message, prompt_count, prompt_count],
        )
        multi_clear.click(
            fn=lambda: (None, "Enter prompts to see aggregated rankings.", gr.update(visible=False), ""),
            outputs=[multi_table, multi_message, prompt_count, prompt_count],
        )

    demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt2Leaderboard UI")
    parser.add_argument("--model-path", type=str, default="lmarena-ai/p2l-7b-grk-02222025", 
                      help="Hugging Face repo ID")
    parser.add_argument("--include-model-list", type=str, default=None)
    args = parser.parse_args()

    include_set = None
    if args.include_model_list:
        with open(args.include_model_list) as fp:
            include_set = set(json.load(fp))

    display_leaderboard(args.model_path, inc_models=include_set)
