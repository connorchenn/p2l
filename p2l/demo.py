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
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_prompts_from_topic(topic: str) -> list[str]:
    system_prompt = """You are tasked with generating 10 questions or prompt instructions that broadly represent a given topic. These questions/prompts will be used for benchmarking an LLM (Language Learning Model), so they should be effective in testing an LLM's understanding of the topic.

When creating these questions/prompts, consider the following guidelines:
- Vary the difficulty level from basic to advanced
- Include a mix of factual, analytical, and creative questions
- Cover different aspects and subtopics within the main topic
- Use a variety of question types (e.g., multiple-choice, open-ended, scenario-based)
- Avoid overly specific or obscure questions that may not effectively test general knowledge
- Ensure questions are clear, concise, and unambiguous"""

    user_prompt = f"Generate 10 questions/prompts for the topic: {topic}\n\nOutput as a numbered list without any additional text."

    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Parse the response to extract prompts
    content = response.choices[0].message.content
    prompts = []
    for line in content.split('\n'):
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, 51)):
            prompt = line.split('.', 1)[1].strip()
            prompts.append(prompt)
    
    return prompts

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
    
    # Thread-safe list for storing results
    results = [None] * len(prompts)
    lock = threading.Lock()

    def process_prompt(idx, prompt):
        data = [{"prompt": prompt, "labels": torch.tensor([0, 1])}]
        betas, eta = get_leaderboard(model, data_collator, model_list, data)
        betas = torch.tensor(betas)
        with lock:
            results[idx] = (betas, eta)

    # Use ThreadPoolExecutor to process prompts in parallel
    with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as executor:
        futures = []
        for idx, prompt in enumerate(prompts):
            futures.append(executor.submit(process_prompt, idx, prompt))
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred

    # Collect results in order
    for betas, eta in results:
        all_coefs.append(betas)
        all_etas.append(eta)

    all_coefs_tensor = torch.stack(all_coefs)  # shape: (num_prompts, num_models)
    all_etas = torch.tensor(all_etas)
    all_etas = all_etas.unsqueeze(1)
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

    leaderboard = leaderboard.coefs.cpu().detach().numpy()
    return dict(sorted({
        model_list[i]: leaderboard[i].item() for i in range(len(model_list))
    }.items(), key=lambda kv: kv[1], reverse=True))

def display_leaderboard(model_path: str, inc_models=None):
    print("Loading model and configuration...")
    local_repo_path = snapshot_download(
    repo_id=args.model_path,
    # allow_patterns=[
    #     "model.safetensors",
    #     "tokenizer.json",
    #     "tokenizer_config.json",
    #     "special_tokens_map.json",
    #     "added_tokens.json",
    #     "vocab.json",
    #     "merges.txt",
    #     "config.json",
    #     "training_args.bin",
    #     "training_config.json",
    #     "model_list.json",
    #     ".gitattributes"
    # ]
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

    custom_css = """
    .container { max-width: 900px; margin: auto; padding: 20px; }
    .output-panel { margin-top: 20px; }
    .tab-content { padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 10px; }
    .prompt-input { margin-bottom: 15px; }
    .info-text { font-size: 0.9em; color: #666; margin: 10px 0; }
    """

    with gr.Blocks(title="ModelMatch", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(
            """
            # 🎯 ModelMatch
            
            Evaluate and compare AI models using either individual prompts or topics.
            """
        )

        with gr.Tabs() as tabs:
            # Individual prompt evaluation tab
            with gr.Tab("Individual Prompt"):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown(
                        """
                        ### Individual Prompt Evaluation
                        Enter a specific prompt to evaluate models on a single task.
                        """
                    )
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter a specific task or question...",
                        lines=4,
                        elem_classes="prompt-input"
                    )
                    
                    with gr.Row():
                        prompt_submit_btn = gr.Button("🔍 Evaluate Models", variant="primary", scale=2)
                        prompt_clear_btn = gr.Button("🔄 Clear", variant="secondary", scale=1)
                    
                    # Example prompts
                    example_prompts = [
                        ["Write a Python function to calculate the Fibonacci sequence."],
                        ["Explain the concept of quantum entanglement to a high school student."],
                        ["Create a creative story about a time-traveling archaeologist."],
                    ]
                    
                    gr.Examples(
                        examples=example_prompts,
                        inputs=prompt_input,
                        label="Try these example prompts"
                    )
                    
                    prompt_status = gr.Markdown("")
                    with gr.Column(visible=False) as prompt_results:
                        prompt_rankings = gr.Dataframe(
                            headers=["Rank", "Model", "Score"],
                            label="Model Rankings"
                        )

            # Topic-based evaluation tab
            with gr.Tab("Topic Evaluation"):
                with gr.Column(elem_classes="tab-content"):
                    gr.Markdown(
                        """
                        ### Topic-Based Evaluation
                        Enter a topic and the system will evaluate models according to their performance on the topic.
                        """
                    )
                    topic_input = gr.Textbox(
                        label="Topic",
                        placeholder="Enter a topic (e.g., 'Python Programming', 'Climate Change', 'Machine Learning')...",
                        lines=1,
                        elem_classes="prompt-input"
                    )
                    
                    with gr.Row():
                        topic_submit_btn = gr.Button("🔍 Evaluate Models", variant="primary", scale=2)
                        topic_clear_btn = gr.Button("🔄 Clear", variant="secondary", scale=1)
                    
                    # Example topics
                    example_topics = [
                        ["Python Programming"],
                        ["Climate Change"],
                        ["Machine Learning"],
                        ["World History"],
                        ["Creative Writing"],
                    ]
                    
                    gr.Examples(
                        examples=example_topics,
                        inputs=topic_input,
                        label="Try these example topics"
                    )
                    
                    topic_status = gr.Markdown("")
                    with gr.Column(visible=False) as topic_results:
                        topic_progress = gr.Markdown("", elem_id="topic-progress")
                        topic_prompt_count = gr.Markdown("")
                        topic_rankings = gr.Dataframe(
                            headers=["Rank", "Model", "Aggregated Score"],
                            label="Model Rankings"
                        )

        def process_topic(topic: str, progress=gr.Progress(track_tqdm=True)):
            if not topic.strip():
                raise gr.Error("Please enter a topic before submitting.")
            
            progress(0.1, desc="Generating prompts from topic...")
            prompts = generate_prompts_from_topic(topic)
            
            if not prompts:
                raise gr.Error("Failed to generate prompts. Please try again.")
            
            progress(0.3, desc="Processing prompts...")
            
            try:
                # Update progress message
                progress_msg = "⏳ Evaluating prompts: 0/" + str(len(prompts))
                
                leaderboard = get_aggregated_leaderboard(
                    model=model,
                    data_collator=data_collator,
                    model_list=model_list,
                    prompts=prompts,
                    model_type=model_config["model_type"],
                    loss_type=model_config["loss_type"]
                )

                progress(0.8, desc="Generating final rankings...")
                
                rankings_df = pd.DataFrame(
                    {
                        "Rank": range(1, len(leaderboard) + 1),
                        "Model": list(leaderboard.keys()),
                        "Aggregated Score": [f"{v:.4f}" for v in leaderboard.values()],
                    }
                )
                
                if inc_models is not None:
                    rankings_df = rankings_df[rankings_df["Model"].isin(inc_models)]
                
                progress(1.0, desc="Done!")
                return (
                    "",
                    gr.update(visible=True),
                    "",  # Clear progress message when done
                    f"Evaluated models using {len(prompts)} diverse prompts",
                    rankings_df
                )
                
            except Exception as e:
                raise gr.Error(f"An error occurred: {str(e)}")

        def evaluate_single_prompt(prompt: str, progress=gr.Progress()):
            if not prompt.strip():
                raise gr.Error("Please enter a prompt before submitting.")
            
            progress(0.3, desc="Processing prompt...")
            
            try:
                data = [{"prompt": prompt, "labels": torch.tensor([0, 1])}]
                beta, _ = get_leaderboard(model, data_collator, model_list, data)
                leaderboard = {
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
                return (
                    "",
                    gr.update(visible=True),
                    df
                )
                
            except Exception as e:
                raise gr.Error(f"An error occurred: {str(e)}")

        # Set up component interactions
        topic_submit_btn.click(
            fn=process_topic,
            inputs=topic_input,
            outputs=[
                topic_status,
                topic_results,
                topic_progress,
                topic_prompt_count,
                topic_rankings
            ],
        )
        
        topic_clear_btn.click(
            fn=lambda: (
                "",
                gr.update(visible=False),
                "",
                "",
                None
            ),
            outputs=[
                topic_status,
                topic_results,
                topic_progress,
                topic_prompt_count,
                topic_rankings
            ],
        )

        prompt_submit_btn.click(
            fn=evaluate_single_prompt,
            inputs=prompt_input,
            outputs=[
                prompt_status,
                prompt_results,
                prompt_rankings
            ],
        )
        
        prompt_clear_btn.click(
            fn=lambda: (
                "",
                gr.update(visible=False),
                None
            ),
            outputs=[
                prompt_status,
                prompt_results,
                prompt_rankings
            ],
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
