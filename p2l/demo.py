import argparse
import gradio as gr
from model import get_tokenizer, get_p2l_model
from dataset import DataCollator
import json
import pandas as pd
from huggingface_hub import hf_hub_download
import yaml
import torch
import time
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


def get_leaderboard(model, data_collator, model_list, prompt):
    processed_data = data_collator(prompt)
    device = next(model.parameters()).device
    out = model(processed_data['input_ids'].to(device), processed_data['attention_mask'].to(device))
    beta = out.coefs[0].cpu().detach().numpy()
    
    model_rankings = {
        model_list[i]: beta[i].item() for i in range(len(beta))
    }
    print(prompt)
    print(model_rankings)
    return model_rankings

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
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    
    model.eval()
    
    print("Model loaded successfully!")

    def _rank(prompt: str, progress=gr.Progress()):
        if not prompt.strip():
            raise gr.Error("Please enter a prompt before submitting.")
        
        progress(0.3, desc="Processing prompt...")
        time.sleep(0.5)  # Add slight delay for better UX
        
        try:
            data = [{"prompt": prompt, "labels": torch.tensor([0, 1])}]
            leaderboard = get_leaderboard(model, data_collator, model_list, data)
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
            return df, f"Successfully ranked {len(df)} models based on your prompt."
            
        except Exception as e:
            raise gr.Error(f"An error occurred: {str(e)}")

    example_prompts = [
        ["Write a Python function to calculate the Fibonacci sequence."],
        ["Explain the concept of quantum entanglement to a high school student."],
        ["Create a creative story about a time-traveling archaeologist."],
    ]

    custom_css = """
    .container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }
    .output-panel {
        margin-top: 20px;
    }
    """

    with gr.Blocks(title="AI Model Ranking System", theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """
                # 🎯 AI Model Ranking System
                
                This interface helps you evaluate and rank different AI models based on your prompts. 
                Simply enter a prompt, and the system will analyze how different models would perform on it.
                
                ### How it works:
                1. Enter your prompt in the text box below
                2. Click "Rank Models" to see the results
                3. The models will be ranked by their predicted performance scores
                """
            )
            
            with gr.Column():
                prompt_box = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Enter a task or question you'd like the AI models to handle...",
                    lines=4,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Rank Models", variant="primary", scale=2)
                    clear_btn = gr.Button("🔄 Clear", variant="secondary", scale=1)
                
                gr.Examples(
                    examples=example_prompts,
                    inputs=prompt_box,
                    label="Try these example prompts"
                )
            
            with gr.Column(elem_classes="output-panel"):
                output_message = gr.Markdown("Results will appear here...")
                output_table = gr.Dataframe(
                    headers=["Rank", "Model", "Score"],
                    label="Model Rankings",
                    visible=True
                )

            gr.Markdown(
                """
                ### 📝 Notes
                - Higher scores indicate better predicted performance
                - Rankings are relative and context-dependent
                """
            )

        submit_btn.click(
            fn=_rank,
            inputs=prompt_box,
            outputs=[output_table, output_message],
        )
        clear_btn.click(
            fn=lambda: (None, "Enter a prompt to see rankings."),
            outputs=[output_table, output_message],
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
