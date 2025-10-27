#!/usr/bin/env python3
"""
Fixed script for 30k and 50k models with proper proportional validation
"""

import subprocess
import os
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class BERTSCAProportionalExperimentRunner:
    def __init__(self):
        self.seed = 123
        self.dataset_path = "/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable.h5"
        self.base_config = {
            "backbone": "bert-base-uncased",
            "batch_size": 64,
            "dropout_rate": 0.3,
            "early_stopping_patience": 5,
            "embedding": "hex",
            "epochs": 20,
            "fusion": "concat",
            "gradient_clip": 1,
            "learning_rate": 0.00005,
            "split_strategy": "attack_split_proportional",
            "trace_net": "fc",
            "use_early_stopping": True,
            "validation_ratio": 0.2,  # 20% of profiling size
            "weight_decay": 0.01
        }
        
        # Data sizes to experiment with - ONLY 30k and 50k for proper validation
        self.data_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"corrected_validation_experiment_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create simplified logs directory
        self.logs_dir = self.results_dir / "experiment_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_log = {
            "experiment_start": datetime.now().isoformat(),
            "experiment_type": "corrected_proportional_validation",
            "seed": self.seed,
            "config": self.base_config,
            "data_sizes": self.data_sizes,
            "validation_strategy": "proportional_to_profiling_size_CORRECTED",
            "results": {}
        }

    def calculate_validation_size(self, profiling_size):
        """Calculate validation size as 20% of profiling size - CORRECTED VERSION"""
        validation_traces = int(profiling_size * self.base_config["validation_ratio"])
        
        # FIXED: Use correct ASCAD attack set size
        max_available = 100000  # ✅ CORRECT: ASCAD has 100k attack traces, not 10k!
        
        # Ensure we don't exceed available attack traces, but be less conservative
        validation_traces = min(validation_traces, max_available - 1000)  # Leave 1k for safety
        
        # Calculate actual validation ratio
        actual_ratio = validation_traces / max_available
        
        return validation_traces, actual_ratio

    def run_training(self, data_size):
        """Train a model with specified data size and proportional validation"""
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING MODEL - DATA SIZE: {data_size:,} traces")
        
        # Calculate proportional validation size
        val_traces, val_ratio = self.calculate_validation_size(data_size)
        print(f"📊 Validation: {val_traces:,} traces ({val_ratio:.1%} of attack set)")
        print(f"⚖️ Profiling:Validation ratio = {data_size}:{val_traces} = {data_size/val_traces:.1f}:1")
        print(f"{'='*60}")
        
        output_dir = self.results_dir / f"model_{data_size//1000}k"
        output_dir.mkdir(exist_ok=True)
        
        # Build training command with proportional validation
        cmd = [
            "python", "train_ascad_bert_improved_no_leakage.py",
            "--dataset_path", self.dataset_path,
            "--downsample", str(data_size),
            "--epochs", str(self.base_config["epochs"]),
            "--batch_size", str(self.base_config["batch_size"]),
            "--learning_rate", str(self.base_config["learning_rate"]),
            "--val_size", str(val_ratio),  # Use calculated proportional ratio
            "--seed", str(self.seed),
            "--output_dir", str(output_dir),
            "--weight_decay", str(self.base_config["weight_decay"]),
            "--gradient_clip", str(self.base_config["gradient_clip"]),
            "--early_stopping_patience", str(self.base_config["early_stopping_patience"]),
            "--dropout_rate", str(self.base_config["dropout_rate"])
        ]
        
        start_time = time.time()
        print(f"🔧 Command: {' '.join(cmd)}")
        
        # Save training command to logs
        cmd_log = {
            "data_size": data_size,
            "validation_traces": val_traces,
            "validation_ratio": val_ratio,
            "command": " ".join(cmd),
            "timestamp": datetime.now().isoformat() 
        }
        
        with open(self.logs_dir / f"training_cmd_{data_size//1000}k.json", 'w') as f:
            json.dump(cmd_log, f, indent=2)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            training_time = time.time() - start_time
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            
            # Find the trained model file
            model_files = list(output_dir.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No model file found after training")
            
            model_path = model_files[0]
            print(f"📁 Model saved: {model_path}")
            
            # Save training output to logs (simplified)
            training_log = {
                "data_size": data_size,
                "validation_traces": val_traces,
                "validation_ratio": val_ratio,
                "training_time": training_time,
                "model_path": str(model_path),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.logs_dir / f"training_result_{data_size//1000}k.json", 'w') as f:
                json.dump(training_log, f, indent=2)
            
            return {
                "success": True,
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "training_time": training_time,
                "validation_traces": val_traces,
                "validation_ratio": val_ratio,
                "profiling_val_ratio": f"{data_size//1000}k:{val_traces//1000}k"
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            
            # Save error to logs
            error_log = {
                "data_size": data_size,
                "validation_traces": val_traces,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.logs_dir / f"training_error_{data_size//1000}k.json", 'w') as f:
                json.dump(error_log, f, indent=2)
            
            return {
                "success": False,
                "error": str(e),
                "validation_traces": val_traces,
                "validation_ratio": val_ratio
            }

    def run_testing(self, model_info, data_size):
        """Test a model on all 16 key bytes"""
        if not model_info["success"]:
            print(f"⚠️ Skipping testing for {data_size//1000}k - training failed")
            return {"success": False, "error": "Training failed"}
        
        print(f"\n{'='*60}")
        print(f"🎯 TESTING MODEL - DATA SIZE: {data_size:,} traces (ALL 16 KEY BYTES)")
        print(f"📊 Validation used: {model_info['validation_traces']} traces")
        print(f"{'='*60}")
        
        model_path = model_info["model_path"]
        output_dir = Path(model_info["output_dir"])
        
        all_key_results = {}
        total_testing_time = 0
        
        for key_byte in range(16):
            print(f"\n🔑 Testing Key Byte {key_byte:02d}/15...")
            
            cmd = [
                "python", "test_ascad_bert_improved_no_leakage.py",
                "--dataset_path", self.dataset_path,
                "--model_path", model_path,
                "--key_byte", str(key_byte),
                "--num_traces", "10000",
                "--batch_size", str(self.base_config["batch_size"]),
                "--profiling_mean_path", str(output_dir / "profiling_mean.npy"),
                "--profiling_std_path", str(output_dir / "profiling_std.npy"),
                "--test_indices_path", str(output_dir / "test_indices.npy")
            ]
            
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                test_time = time.time() - start_time
                total_testing_time += test_time
                
                # Parse results from output
                key_results = self.parse_test_output(result.stdout)
                key_results["test_time"] = test_time
                all_key_results[f"byte_{key_byte:02d}"] = key_results
                
                found_str = "Found" if key_results.get("key_found", False) else "Not Found"
                traces_str = f" @{key_results.get('found_at_trace', 'N/A')}" if key_results.get("key_found", False) else ""
                print(f"   ✅ Byte {key_byte:02d}: Rank {key_results.get('best_rank', 'N/A')} ({found_str}{traces_str})")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Byte {key_byte:02d}: Testing failed - {e}")
                all_key_results[f"byte_{key_byte:02d}"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Calculate summary statistics
        successful_bytes = [r for r in all_key_results.values() if r.get("success", True)]
        ranks = [r.get("best_rank", 256) for r in successful_bytes if "best_rank" in r]
        found_traces = [r.get("found_at_trace", None) for r in successful_bytes if r.get("key_found", False)]
        
        summary = {
            "total_testing_time": total_testing_time,
            "successful_bytes": len(successful_bytes),
            "failed_bytes": 16 - len(successful_bytes),
            "avg_rank": float(np.mean(ranks)) if ranks else None,
            "min_rank": int(np.min(ranks)) if ranks else None,
            "max_rank": int(np.max(ranks)) if ranks else None,
            "keys_found": len([r for r in successful_bytes if r.get("key_found", False)]),
            "avg_traces_to_find": float(np.mean(found_traces)) if found_traces else None,
            "min_traces_to_find": int(np.min(found_traces)) if found_traces else None,
            "max_traces_to_find": int(np.max(found_traces)) if found_traces else None,
            "validation_traces_used": model_info["validation_traces"],
            "profiling_val_ratio": model_info["profiling_val_ratio"]
        }
        
        print(f"\n📊 SUMMARY FOR {data_size//1000}k MODEL:")
        print(f"   Validation used: {model_info['validation_traces']} traces")
        print(f"   Keys found: {summary['keys_found']}/16")
        if summary['avg_rank'] is not None:
            print(f"   Avg rank: {summary['avg_rank']:.2f}")
        if summary['avg_traces_to_find'] is not None:
            print(f"   Avg traces to find: {summary['avg_traces_to_find']:.1f}")
        
        # Save testing results to logs
        testing_log = {
            "data_size": data_size,
            "summary": summary,
            "key_byte_details": all_key_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.logs_dir / f"testing_result_{data_size//1000}k.json", 'w') as f:
            json.dump(testing_log, f, indent=2)
        
        return {
            "success": True,
            "key_byte_results": all_key_results,
            "summary": summary
        }

    def parse_test_output(self, stdout):
        """Parse test script output to extract key metrics"""
        lines = stdout.split('\n')
        results = {"success": True}
        
        for line in lines:
            if "Key byte found at trace" in line or "🔑 Key found at trace" in line:
                try:
                    # Handle both possible formats
                    if "found at trace" in line:
                        found_at = int(line.split("trace")[1].split()[0])
                    else:
                        found_at = int(line.split("@")[1].split()[0])
                    results["key_found"] = True
                    results["found_at_trace"] = found_at
                except:
                    pass
            elif "Best rank achieved:" in line:
                try:
                    rank = int(line.split("Best rank achieved:")[1].strip())
                    results["best_rank"] = rank
                except:
                    pass
            elif "Key byte not found" in line or "❌ Key byte not found" in line:
                results["key_found"] = False
        
        return results

    def save_experiment_results(self):
        """Save comprehensive experiment results"""
        # Save main experiment log (JSON serializable)
        log_copy = json.loads(json.dumps(self.experiment_log, default=str))
        
        with open(self.results_dir / "experiment_log.json", 'w') as f:
            json.dump(log_copy, f, indent=2)
        
        # Create summary CSV for easy analysis
        try:
            import pandas as pd
            
            summary_data = []
            for data_size_key, results in self.experiment_log["results"].items():
                if results["testing"]["success"]:
                    summary = results["testing"]["summary"]
                    training = results["training"]
                    
                    summary_data.append({
                        "data_size": results["data_size"],
                        "validation_traces": training.get("validation_traces", "N/A"),
                        "profiling_val_ratio": training.get("profiling_val_ratio", "N/A"),
                        "training_time_sec": training["training_time"],
                        "testing_time_sec": summary["total_testing_time"],
                        "keys_found": summary["keys_found"],
                        "success_rate": f"{summary['keys_found']}/16",
                        "avg_rank": summary["avg_rank"],
                        "min_rank": summary["min_rank"],
                        "max_rank": summary["max_rank"],
                        "avg_traces_to_find": summary["avg_traces_to_find"],
                        "min_traces_to_find": summary["min_traces_to_find"],
                        "max_traces_to_find": summary["max_traces_to_find"]
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(self.results_dir / "corrected_validation_summary.csv", index=False)
                print(f"📊 Summary saved to: {self.results_dir / 'corrected_validation_summary.csv'}")
                
        except ImportError:
            print("⚠️ pandas not available, skipping CSV export")

    def run_full_experiment(self):
        """Run the complete experiment pipeline with CORRECTED proportional validation"""
        print(f"\n🚀 STARTING CORRECTED BERT-SCA PROPORTIONAL VALIDATION EXPERIMENT")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📊 Logs directory: {self.logs_dir}")
        print(f"🎯 Seed: {self.seed}")
        print(f"📊 Data sizes: {self.data_sizes}")
        print(f"⚖️ Validation strategy: 20% of profiling size (CORRECTED - 100k attack traces)")
        
        # Print validation sizes upfront
        print(f"\n📋 Planned validation sizes (CORRECTED):")
        for size in self.data_sizes:
            val_traces, val_ratio = self.calculate_validation_size(size)
            print(f"   {size//1000}k profiling → {val_traces:,} validation ({size//1000}k:{val_traces//1000}k ratio)")
        
        experiment_start = time.time()
        
        for data_size in self.data_sizes:
            size_key = f"{data_size//1000}k"
            
            # Step 1: Training
            print(f"\n{'#'*80}")
            print(f"STEP 1: TRAINING {size_key} MODEL (CORRECTED PROPORTIONAL VALIDATION)")
            print(f"{'#'*80}")
            
            training_results = self.run_training(data_size)
            
            # Step 2: Testing
            print(f"\n{'#'*80}")
            print(f"STEP 2: TESTING {size_key} MODEL")
            print(f"{'#'*80}")
            
            testing_results = self.run_testing(training_results, data_size)
            
            # Store results
            self.experiment_log["results"][size_key] = {
                "data_size": data_size,
                "training": training_results,
                "testing": testing_results
            }
            
            # Save intermediate results
            self.save_experiment_results()
        
        total_time = time.time() - experiment_start
        self.experiment_log["experiment_end"] = datetime.now().isoformat()
        self.experiment_log["total_experiment_time"] = total_time
        
        # Final save
        self.save_experiment_results()
        
        print(f"\n{'='*80}")
        print(f"🎉 CORRECTED PROPORTIONAL VALIDATION EXPERIMENT COMPLETED!")
        print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"📁 Results saved in: {self.results_dir}")
        print(f"📊 Logs saved in: {self.logs_dir}")
        print(f"⚖️ All models used CORRECTED proportional validation")
        print(f"   30k profiling → 6k validation (5:1 ratio)")
        print(f"   50k profiling → 10k validation (5:1 ratio)")
        print(f"{'='*80}")

if __name__ == "__main__":
    runner = BERTSCAProportionalExperimentRunner()
    runner.run_full_experiment()