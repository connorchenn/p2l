from p2l.eval_chrono import main as eval_chrono_main
from p2l.online_training.visualize_eval import visualize_eval
import argparse
import shutil
import os

def main(args):
    eval_chrono_main(args)
    visualize_eval(models=[args.output_dir], base=args.eval_folder, eval_plot_folder=args.eval_folder, accuracy=False)
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        shutil.rmtree('/root/.cache/huggingface/hub')
        print(f"Deleted validation results folder: {args.output_dir} and huggingface cache")

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
        "--online-align", type=str, help="json path for online aligned evals (eval after time t)"
    )
    parser.add_argument(
        "--train-time", type=str, help="file containing last times for each batch in train file"
    )
    parser.add_argument(
        "--val-time", type=str, help="file containing last times for each batch in val file"
    )
    parser.add_argument(
        "--eval-folder", type=str, default='eval_plots2', help='folder to save eval plots'
    )
    args = parser.parse_args()

    if args.time_align and (not args.train_time or not args.val_time):
        parser.error("--train-time and --val-time required when --time-align is set.")
    
    main(args)