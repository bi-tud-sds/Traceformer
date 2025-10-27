#!/usr/bin/env python3
import subprocess
import os
import time
from pathlib import Path
import csv
from datetime import datetime
import re
import argparse

def parse_metrics_from_output(output):
    metrics = {}
    # Parse lines for metrics
    best_rank_match = re.search(r"Best rank achieved: (\d+)", output)
    if best_rank_match:
        metrics["best_rank"] = int(best_rank_match.group(1))
    found_at_trace_match = re.search(r"Key byte found at trace (\d+)", output)
    if found_at_trace_match:
        metrics["found_at_trace"] = int(found_at_trace_match.group(1))
    predicted_key_match = re.search(r"Predicted key byte: 0x([0-9a-fA-F]+)", output)
    if predicted_key_match:
        metrics["predicted_key"] = predicted_key_match.group(1)
    real_key_match = re.search(r"Real key byte: 0x([0-9a-fA-F]+)", output)
    if real_key_match:
        metrics["real_key"] = real_key_match.group(1)
    return metrics

def get_seed_from_model_path(model_path):
    match = re.search(r"seed(\d+)", model_path)
    return int(match.group(1)) if match else None

CSV_HEADER = [
    "timestamp", "key_byte", "model_path", "dataset_path", "seed", "status", "duration_sec", "best_rank", "found_at_trace", "predicted_key", "real_key"
]

CSV_PATH = "key_byte_results/test_results.csv"

def run_test_for_byte(key_byte, model_path, dataset_path, ablation=None):
    cmd = [
        "python",
        "/root/alper/jupyter_transfer/downloaded_data/work/BERT/ASCAD_all_latest/test_ascad_bert_improved.py",
        "--dataset_path", dataset_path,
        "--model_path", model_path,
        "--key_byte", str(key_byte)
    ]
    if ablation:
        cmd += ["--ablation", ablation]
    print(f"\n{'='*80}")
    print(f"Testing key byte {key_byte}")
    print(f"{'='*80}\n")
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        status = "SUCCESS"
        output = result.stdout
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        output = e.stdout + "\n" + (e.stderr or "")
    end_time = time.time()
    duration = end_time - start_time
    metrics = parse_metrics_from_output(output)
    # Compose row
    row = {
        "timestamp": datetime.now().isoformat(),
        "key_byte": key_byte,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "seed": get_seed_from_model_path(model_path),
        "status": status,
        "duration_sec": f"{duration:.2f}",
        "best_rank": metrics.get("best_rank"),
        "found_at_trace": metrics.get("found_at_trace"),
        "predicted_key": metrics.get("predicted_key"),
        "real_key": metrics.get("real_key")
    }
    # Write to CSV
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return status == "SUCCESS"

def main():
    parser = argparse.ArgumentParser(description="Batch test all key bytes with optional ablation.")
    parser.add_argument("--ablation", type=str, default=None, choices=[None, "no_dropout", "no_posenc", "no_fusion", "tiny_emb"], help="Ablation variant to run (optional)")
    args = parser.parse_args()
    results_dir = Path("key_byte_results")
    results_dir.mkdir(exist_ok=True)
    model_path = "/root/alper/jupyter_transfer/downloaded_data/work/BERT/ASCAD_all_latest/experiments/cutting_edge/seed_42_bs256_lr8e-4/bert_sca_ascad-variable_seed42_e150_lr8e-04_s30000_20250708_001348.pth"
    dataset_path = "/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable.h5"
    for key_byte in range(16):
        success = run_test_for_byte(key_byte, model_path, dataset_path, ablation=args.ablation)
        time.sleep(2)

if __name__ == "__main__":
    main() 