#!/usr/bin/env python3
"""
Script to find the best random seeds based on the lowest number of traces needed.
Filters seeds where found_at_trace is between 0-30 and generates a summary text file.
"""

import os
import yaml
import glob
from pathlib import Path

def find_best_seeds(experiment_dir, output_file, min_traces=0, max_traces=30):
    """
    Find seeds where found_at_trace is between min_traces and max_traces.
    
    Args:
        experiment_dir (str): Path to the seed search experiment directory
        output_file (str): Path to the output text file
        min_traces (int): Minimum number of traces (inclusive)
        max_traces (int): Maximum number of traces (inclusive)
    """
    
    # Get all result files
    result_files = glob.glob(os.path.join(experiment_dir, "result_seed_*.yaml"))
    
    if not result_files:
        print(f"No result files found in {experiment_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Parse all results
    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Extract seed and found_at_trace
            seed = data.get('seed')
            found_at_trace = data.get('found_at_trace')
            
            if seed is not None and found_at_trace is not None:
                results.append({
                    'seed': seed,
                    'found_at_trace': found_at_trace,
                    'file': os.path.basename(file_path)
                })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Filter results based on trace range
    filtered_results = [
        result for result in results 
        if min_traces <= result['found_at_trace'] <= max_traces
    ]
    
    # Sort by found_at_trace (ascending)
    filtered_results.sort(key=lambda x: x['found_at_trace'])
    
    # Write results to text file
    with open(output_file, 'w') as f:
        f.write(f"Best Seeds Analysis (found_at_trace between {min_traces}-{max_traces})\n")
        f.write("=" * 60 + "\n\n")
        
        if filtered_results:
            f.write(f"Found {len(filtered_results)} seeds with traces between {min_traces}-{max_traces}:\n\n")
            
            # Write detailed results
            f.write("Seed\tTraces\tFile\n")
            f.write("-" * 30 + "\n")
            
            for result in filtered_results:
                f.write(f"{result['seed']}\t{result['found_at_trace']}\t{result['file']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SUMMARY:\n")
            f.write(f"Total seeds analyzed: {len(results)}\n")
            f.write(f"Seeds in range {min_traces}-{max_traces}: {len(filtered_results)}\n")
            
            if filtered_results:
                best_result = filtered_results[0]
                f.write(f"Best seed: {best_result['seed']} (found at trace {best_result['found_at_trace']})\n")
                
                # Group by trace count
                trace_counts = {}
                for result in filtered_results:
                    trace_count = result['found_at_trace']
                    if trace_count not in trace_counts:
                        trace_counts[trace_count] = []
                    trace_counts[trace_count].append(result['seed'])
                
                f.write("\nSeeds grouped by trace count:\n")
                for trace_count in sorted(trace_counts.keys()):
                    seeds = trace_counts[trace_count]
                    f.write(f"  {trace_count} traces: {seeds}\n")
        else:
            f.write(f"No seeds found with traces between {min_traces}-{max_traces}\n")
            f.write(f"Total seeds analyzed: {len(results)}\n")
            
            # Show the range of trace values found
            if results:
                trace_values = [r['found_at_trace'] for r in results]
                f.write(f"Trace range in data: {min(trace_values)} - {max(trace_values)}\n")
    
    print(f"Analysis complete! Results written to {output_file}")
    print(f"Found {len(filtered_results)} seeds with traces between {min_traces}-{max_traces}")
    
    if filtered_results:
        print(f"Best seed: {filtered_results[0]['seed']} (found at trace {filtered_results[0]['found_at_trace']})")

if __name__ == "__main__":
    # Configuration
    experiment_dir = "experiments/seed_search_experiment"
    output_file = "best_seeds_0_30.txt"
    min_traces = 0
    max_traces = 30
    
    # Run the analysis
    find_best_seeds(experiment_dir, output_file, min_traces, max_traces) 