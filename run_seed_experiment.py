import argparse
import subprocess
import yaml
from pathlib import Path
import os
import wandb
from datetime import datetime
import pandas as pd
import concurrent.futures
import glob
import re

def run_single_seed(args, seed, experiment_dir):
    """Run training and evaluation for a single seed."""
    print(f"Starting training for seed {seed}...")

    # Define output directory for configs and models
    config_output_dir = Path(args.config_eval_dir)
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Train with all regularization parameters
    train_cmd = [
        "python", "train_ascad_bert_improved.py",
        "--dataset_path", args.dataset_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--downsample", str(args.downsample) if args.downsample else "None",
        "--seed", str(seed),
        "--output_dir", str(config_output_dir),
        "--weight_decay", str(args.weight_decay),
        "--gradient_clip", str(args.gradient_clip),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--dropout_rate", str(args.dropout_rate)
    ]
    
    model_path = None
    try:
        train_result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print(f"Training completed for seed {seed}")
        print(f"Training stdout: {train_result.stdout[-500:]}")  # Last 500 chars
        if train_result.stderr:
            print(f"Training stderr: {train_result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed for seed {seed}: {e}")
        print(f"Training stderr: {e.stderr}")
        raise

    # Find the model file for this seed - look in current directory
    # Try multiple patterns to find the model file
    model_patterns = [
        f"bert_sca_*_seed{seed}_*.pth",  # Most general pattern
        f"best_bert_sca_*_seed{seed}.pth",  # Best model pattern
        f"bert_sca_*_seed{seed}_e{args.epochs}_*.pth"  # Pattern with epochs
    ]
    
    model_files = []
    for pattern in model_patterns:
        print(f"Looking for model file with pattern: {pattern} in {config_output_dir}")
        model_files = glob.glob(str(config_output_dir / pattern))
        if model_files:
            print(f"Found {len(model_files)} model files with pattern: {pattern}")
            break
    
    if not model_files:
        # List all .pth files in current directory for debugging
        all_pth_files = glob.glob(str(config_output_dir / "*.pth"))
        print(f"All .pth files in {config_output_dir}: {all_pth_files}")
        
        # List all files containing the seed number
        all_files_with_seed = [f for f in os.listdir(config_output_dir) if str(seed) in f]
        print(f"All files in {config_output_dir} containing seed {seed}: {all_files_with_seed}")
        
        raise RuntimeError(f"No model file found for seed {seed}! Looked for patterns: {model_patterns} in {config_output_dir}")
    
    model_path = model_files[0]  # Take the first match
    print(f"Using model file: {model_path}")

    # Evaluate and capture output
    print(f"Starting evaluation for seed {seed}...")
    eval_cmd = [
        "python", "test_ascad_bert_improved.py",
        "--dataset_path", args.dataset_path,
        "--model_path", model_path,
        "--num_traces", str(args.num_traces),
        "--batch_size", str(args.batch_size),
        "--key_byte", str(args.key_byte),
        "--profiling_mean_path", str(config_output_dir / "profiling_mean.npy"),
        "--profiling_std_path", str(config_output_dir / "profiling_std.npy")
    ]
    
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        eval_stdout = result.stdout
        print(f"Evaluation completed for seed {seed}")
        if result.stderr:
            print(f"Evaluation stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for seed {seed}: {e}")
        print(f"Evaluation stderr: {e.stderr}")
        # Don't raise here, still try to save results and clean up

    # Parse evaluation output for key metric (e.g., 'Best rank achieved' or 'Key found at trace')
    best_rank = None
    found_at_trace = None
    for line in eval_stdout.splitlines():
        if 'Best rank achieved:' in line:
            best_rank = int(re.findall(r'\d+', line)[0])
        if 'Key found at trace' in line or 'Key byte found at trace' in line:
            found_at_trace = int(re.findall(r'\d+', line)[0])

    # Save config and results as YAML in the experiment directory
    result_info = {
        "seed": seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "downsample": args.downsample,
        "key_byte": args.key_byte,
        "best_rank": best_rank,
        "found_at_trace": found_at_trace,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "early_stopping_patience": args.early_stopping_patience,
        "dropout_rate": args.dropout_rate
    }
    
    # Create the experiment directory if it doesn't exist
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save YAML file in the experiment directory
    yaml_path = experiment_dir / f"result_seed_{seed}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(result_info, f)
    print(f"Results saved to {yaml_path}")

    # Always try to delete ALL model files for this seed to save space
    deleted_count = 0
    failed_count = 0
    
    # Find all model files for this seed
    all_model_patterns = [
        f"bert_sca_*_seed{seed}_*.pth",  # All model patterns for this seed
        f"best_bert_sca_*_seed{seed}.pth"  # Best model pattern for this seed
    ]
    
    for pattern in all_model_patterns:
        model_files_to_delete = glob.glob(str(config_output_dir / pattern))
        for model_file in model_files_to_delete:
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                    print(f"✅ Model file {model_file} deleted successfully")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete model file {model_file}: {e}")
                    failed_count += 1
            else:
                print(f"⚠️ Model file {model_file} not found for deletion")
    
    if deleted_count > 0:
        print(f"✅ Successfully deleted {deleted_count} model files for seed {seed}")
    if failed_count > 0:
        print(f"⚠️ Failed to delete {failed_count} model files for seed {seed}")
    if deleted_count == 0 and failed_count == 0:
        print(f"ℹ️ No model files found to delete for seed {seed}")

    return result_info

def run_experiment(args):
    """Run the complete experiment: training and evaluation in parallel."""
    # Create the seed_search_experiment directory
    experiment_dir = Path(args.output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb for the overall experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="bert-sca-seed-experiment",
        name=f"seed_experiment_{timestamp}",
        config={
            "dataset_path": args.dataset_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "downsample": args.downsample,
            "num_traces": args.num_traces,
            "seeds": args.seeds,
            "model_type": "bert-base-uncased",
            "embedding": "hex",
            "trace_net": "fc",
            "fusion": "concat"
        }
    )

    print("\n=== Starting Parallel Training & Evaluation Phase ===")
    results_table = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_single_seed, args, seed, experiment_dir) for seed in args.seeds]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results_table.append(result)
            except Exception as exc:
                print(f"Seed generated an exception: {exc}")

    # Optionally, aggregate and log results as before
    # (You may want to parse evaluation outputs for best_val_acc, final_rank, etc.)
    # For now, just log the model paths and seeds
    wandb.log({
        "results_table": wandb.Table(dataframe=pd.DataFrame(results_table))
    })
    print("\n=== Experiment Complete ===")
    print(f"Results saved in: {experiment_dir}")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run complete random seed experiment")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ASCAD dataset")
    parser.add_argument("--output_dir", type=str, default="experiments/seed_search_experiment", help="Directory to save final results")
    parser.add_argument("--config_eval_dir", type=str, default="experiments/config_evaluations", help="Directory to save configs and intermediate models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--downsample", type=int, default=1000, help="Number of training samples to use")
    parser.add_argument("--num_traces", type=int, default=1000, help="Number of traces to use for evaluation")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 222, 333, 444, 555, 666, 777, 888, 999], help="Random seeds to try")
    parser.add_argument("--key_byte", type=int, default=2, help="Key byte index to evaluate")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel processes to use")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--early_stopping_patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for regularization")
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main() 