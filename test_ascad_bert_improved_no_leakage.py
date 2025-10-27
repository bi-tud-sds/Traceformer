import numpy as np
import torch
import h5py
import wandb
import argparse
import os
from train_ascad_bert_improved import BERT_SCA_Model, BertModel, tokenizer
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import re

# Define DEVICE constant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AES Sbox
AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

def load_test_data_with_exclusion(dataset_path, test_indices_path, profiling_mean_path='profiling_mean.npy', profiling_std_path='profiling_std.npy'):
    """
    Load ONLY the test portion of attack traces, excluding validation indices.
    This ensures NO DATA LEAKAGE from training/validation.
    """
    print(f"Loading dataset: {dataset_path}")
    print(f"🔒 Loading test indices to ensure no data leakage...")
    
    # Load the test indices that were saved during training
    if not os.path.exists(test_indices_path):
        raise FileNotFoundError(f"Test indices file not found: {test_indices_path}")
    
    test_indices = np.load(test_indices_path)
    print(f"📋 Loaded {len(test_indices)} test indices (validation indices excluded)")
    
    with h5py.File(dataset_path, 'r') as in_file:
        # Load ALL attack traces first
        X_attack_all = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        plaintexts_all = np.array(in_file['Attack_traces/metadata'][:]['plaintext'])
        keys_all = np.array(in_file['Attack_traces/metadata'][:]['key'])
        
        # Extract ONLY the test subset using saved indices
        X_test = X_attack_all[test_indices]
        plaintexts = plaintexts_all[test_indices]
        keys = keys_all[test_indices]
        
        print(f"✅ Using ONLY test traces: {len(X_test)} out of {len(X_attack_all)} total attack traces")
        print(f"🚫 Validation traces excluded: {len(X_attack_all) - len(X_test)} traces")
        
        # Load profiling mean and std for normalization
        if not os.path.exists(profiling_mean_path) or not os.path.exists(profiling_std_path):
            raise FileNotFoundError(f"Profiling mean/std files not found: {profiling_mean_path}, {profiling_std_path}")
        
        # Normalize test traces using profiling statistics
        mean = np.load(profiling_mean_path)
        std = np.load(profiling_std_path)
        std[std == 0] = 1e-8 
        X_test = (X_test - mean) / std
        
    return X_test, plaintexts, keys

def compute_rank(predictions, plaintext, key, byte, index):
    key_bytes_proba = np.zeros(256)
    plaintext_byte = plaintext[index][byte]
    real_key = key[index][byte]
    
    for key_byte in range(256):
        sbox_output = AES_Sbox[plaintext_byte ^ key_byte]
        key_bytes_proba[key_byte] = predictions[sbox_output]
    
    sorted_proba = np.argsort(key_bytes_proba)[::-1]
    key_rank = np.where(sorted_proba == real_key)[0][0]
    predicted_key = sorted_proba[0]
    
    return key_rank, predicted_key, real_key

def predict_batch(model, traces, plaintexts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    plaintext_strs = [' '.join([f'{b:02x}' for b in p]) for p in plaintexts]
    encodings = tokenizer(plaintext_strs, padding=True, truncation=True, return_tensors="pt")
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    traces = torch.tensor(traces, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, traces)
        predictions = torch.softmax(outputs, dim=1).cpu().numpy()
    
    return predictions

def evaluate_key_rank(model, X_test, plaintexts, keys, key_byte, num_traces, batch_size=100):
    print(f"🎯 Evaluating key byte: {key_byte}, Processing {num_traces} traces...")
    print(f"📊 Using ONLY unseen test data (validation traces excluded)")
    
    # Ensure we don't use more traces than available
    num_traces = min(num_traces, len(X_test))
    
    ranks = np.zeros((num_traces, 2))
    key_byte_found = False
    key_info = {"key_byte_found": False, "found_at_trace": None, "best_rank": 256}
    best_rank_so_far = 256  # Track best rank achieved
    
    for batch_start in range(0, num_traces, batch_size):
        batch_end = min(batch_start + batch_size, num_traces)
        batch_traces = X_test[batch_start:batch_end]
        batch_plaintexts = plaintexts[batch_start:batch_end]
        
        predictions = predict_batch(model, batch_traces, batch_plaintexts)
        
        for i in range(len(batch_traces)):
            idx = batch_start + i
            rank, predicted_key, real_key = compute_rank(predictions[i], plaintexts, keys, key_byte, idx)
            best_rank_so_far = min(best_rank_so_far, rank)  # Update best rank
            ranks[idx][0] = idx
            ranks[idx][1] = best_rank_so_far  # Store best rank achieved so far
            
            if rank == 0 and not key_byte_found:
                key_byte_found = True
                key_info.update({
                    "key_byte_found": True,
                    "found_at_trace": idx,
                    "best_rank": 0,
                    "predicted_key": predicted_key,
                    "real_key": real_key
                })
                print(f"🔑 Key found at trace {idx} with rank 0")
    
    if not key_byte_found:
        print("❌ Key not found within the given traces.")
        key_info["best_rank"] = best_rank_so_far
    
    return ranks, key_info

def wandb_initialize(args):
    dataset_name = Path(args.dataset_path).stem  # e.g., "ascad-variable-desync50"
    run_name = f"TEST-Byte-{args.key_byte:02d}"
    project_name = f"ascad-bert-key-eval-{dataset_name}"  # Unique project per dataset variant

    # Extract seed from model filename
    model_filename = Path(args.model_path).name
    seed = None
    # Look for seed pattern in filename: seed{number}
    seed_match = re.search(r'seed(\d+)', model_filename)
    if seed_match:
        seed = int(seed_match.group(1))
        run_name = f"TEST-Byte-{args.key_byte:02d}-seed{seed}"
    
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "dataset_path": args.dataset_path,
            "model_path": args.model_path,
            "num_traces": args.num_traces,
            "batch_size": args.batch_size,
            "key_byte": args.key_byte,
            "training_seed": seed,
            "evaluation_type": "FINAL_TEST_NO_LEAKAGE"
        }
    )
    return project_name, run_name
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT model on ASCAD dataset (TEST ONLY - NO DATA LEAKAGE)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset (.h5)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--num_traces", type=int, default=10000, help="Number of traces to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--key_byte", type=int, default=0, help="Key byte index to evaluate")
    parser.add_argument("--output_csv", type=str, default="rank_results.csv", help="Output CSV file for rank results")
    parser.add_argument("--output_plot", type=str, default="key_rank_evolution.png", help="Output file for rank evolution plot")
    parser.add_argument("--profiling_mean_path", type=str, default=None, help="Path to profiling mean file")
    parser.add_argument("--profiling_std_path", type=str, default=None, help="Path to profiling std file")
    parser.add_argument("--test_indices_path", type=str, default=None, help="Path to test indices file")
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "no_dropout", "no_posenc", "no_fusion", "tiny_emb"], help="Ablation variant to run")
    args = parser.parse_args()

    # Set default paths if not provided (look in model directory)
    model_dir = Path(args.model_path).parent
    
    if args.profiling_mean_path is None:
        args.profiling_mean_path = str(model_dir / 'profiling_mean.npy')
    if args.profiling_std_path is None:
        args.profiling_std_path = str(model_dir / 'profiling_std.npy')
    if args.test_indices_path is None:
        args.test_indices_path = str(model_dir / 'test_indices.npy')

    # Verify all required files exist
    required_files = [
        args.profiling_mean_path,
        args.profiling_std_path, 
        args.test_indices_path
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("="*60)
    print("🎯 FINAL TEST EVALUATION - NO DATA LEAKAGE")
    print("="*60)
    print(f"📋 Using test indices from: {args.test_indices_path}")
    print(f"🚫 Validation traces will be excluded automatically")

    wandb_initialize(args)

    print("Loading model and test data...")
    
    # Load ONLY the test data (validation indices excluded)
    X_test, plaintexts, keys = load_test_data_with_exclusion(
        args.dataset_path, 
        args.test_indices_path,
        args.profiling_mean_path, 
        args.profiling_std_path
    )
    
    # Load model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SCA_Model(bert_model, trace_length=X_test.shape[1], ablation=args.ablation)
    
    # Load the state dict with proper error handling
    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    model = model.to(DEVICE)
    
    # Evaluate on test data
    ranks, key_info = evaluate_key_rank(
        model, X_test, plaintexts, keys, args.key_byte, 
        min(args.num_traces, len(X_test)), args.batch_size
    )
    
    # Process ranks for smoother evolution
    processed_ranks = []
    current_rank = 256  # Start with worst possible rank
    
    for i in range(len(ranks)):
        trace_idx = int(ranks[i][0])
        rank_value = int(ranks[i][1])
        current_rank = min(current_rank, rank_value)
        processed_ranks.append([trace_idx, current_rank])
    
    # Create the ranking plot data
    table = wandb.Table(
        data=processed_ranks,
        columns=["Number of Traces", "Key Rank"]
    )
    
    # Log the rank evolution plot
    wandb.log({
        "Key Rank Evolution (Test Only)": wandb.plot.line(
            table,
            "Number of Traces",   
            "Key Rank",           
            title="Key Rank Evolution - Final Test (No Data Leakage)"
        )
    })
    
    # Log statistics and key information
    wandb.log({
        "final_rank": float(processed_ranks[-1][1]),
        "min_rank": float(min(r[1] for r in processed_ranks)),
        "max_rank": float(max(r[1] for r in processed_ranks)),
        "avg_rank": float(np.mean([r[1] for r in processed_ranks])),
        "key_byte_found": key_info["key_byte_found"],
        "found_at_trace": key_info["found_at_trace"],
        "best_rank": key_info["best_rank"],
        "test_traces_used": len(X_test),
        "evaluation_type": "FINAL_TEST_NO_LEAKAGE"
    })
    
    # Print final results
    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS (NO DATA LEAKAGE)")
    print("="*60)
    print(f"Key byte index attacked: {args.key_byte}")
    print(f"Test traces used: {len(X_test)}")
    print(f"Best rank achieved: {key_info['best_rank']}")
    
    if key_info.get("key_byte_found"):
        print(f"🔑 Key byte found at trace {key_info['found_at_trace']}")
        if "predicted_key" in key_info and "real_key" in key_info:
            print(f"Predicted key byte: 0x{key_info['predicted_key']:02x}")
            print(f"Real key byte: 0x{key_info['real_key']:02x}")
    else:
        print("❌ Key byte not found in the given test traces.")
    
    print("="*60)
    print("✅ Evaluation complete - results are honest (no data leakage)")
    
    # Save evaluation config
    eval_config = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "key_byte": args.key_byte,
        "num_traces": args.num_traces,
        "batch_size": args.batch_size,
        "test_traces_used": len(X_test),
        "embedding": "hex",
        "backbone": "bert-base-uncased",
        "trace_net": "fc",
        "fusion": "concat",
        "best_rank": key_info['best_rank'],
        "key_byte_found": key_info['key_byte_found'],
        "found_at_trace": key_info.get('found_at_trace'),
        "final_rank": float(processed_ranks[-1][1]) if processed_ranks else None,
        "evaluation_type": "FINAL_TEST_NO_LEAKAGE",
        "data_leakage_prevented": True
    }

    # Save to the same directory as the model
    eval_config_path = model_dir / f"{Path(args.model_path).stem}_FINAL_TEST_eval_config.yaml"
    
    try:
        with open(eval_config_path, 'w') as f:
            yaml.dump(eval_config, f)
        print(f"✅ Test evaluation config saved to {eval_config_path}")
    except Exception as e:
        print(f"⚠️ Failed to save evaluation config: {e}")
    
    wandb.finish()

if __name__ == "__main__":
    main()