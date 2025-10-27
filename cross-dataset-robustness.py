#!/usr/bin/env python3
"""
Cross-Dataset Robustness Evaluation Script
Evaluates trained BERT model across multiple ASCAD dataset variants
to assess robustness against desynchronization countermeasures
"""

import os
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Tuple
import re

class CrossDatasetEvaluator:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'robustness_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.experiment_start = datetime.now()
        
        # Dataset configurations
        self.datasets = {
            'no_desync': {
                'path': '/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable.h5',
                'description': 'No Desynchronization',
                'jitter_level': 0,
                'expected_performance': 'Best (baseline)'
            },
            'desync_50': {
                'path': '/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable-desync50.h5',
                'description': '50-Sample Jitter',
                'jitter_level': 50,
                'expected_performance': 'Moderate degradation'
            },
            'desync_100': {
                'path': '/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable-desync100.h5',
                'description': '100-Sample Jitter', 
                'jitter_level': 100,
                'expected_performance': 'Significant degradation'
            }
        }
        
    def verify_model_and_datasets(self):
        """Verify that model and all datasets exist"""
        # Check model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Check model directory has required files
        model_dir = self.model_path.parent
        required_files = ['profiling_mean.npy', 'profiling_std.npy', 'test_indices.npy']
        
        for filename in required_files:
            file_path = model_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required model file not found: {file_path}")
        
        # Check all datasets exist
        missing_datasets = []
        for dataset_name, dataset_info in self.datasets.items():
            if not os.path.exists(dataset_info['path']):
                missing_datasets.append(f"{dataset_name}: {dataset_info['path']}")
        
        if missing_datasets:
            raise FileNotFoundError(f"Missing datasets:\n" + "\n".join(missing_datasets))
        
        self.logger.info("✅ All required files and datasets verified")
    
    def evaluate_on_dataset(self, dataset_name: str, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model on a specific dataset"""
        self.logger.info(f"🎯 Evaluating on {dataset_name}: {dataset_info['description']}")
        start_time = time.time()
        
        dataset_results = {}
        model_dir = self.model_path.parent
        
        # Create dataset-specific output directory
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        
        # Evaluate all 16 key bytes
        total_traces_list = []
        successful_bytes = 0
        
        for byte_idx in range(16):
            self.logger.info(f"  Evaluating byte {byte_idx}/15...")
            
            try:
                # Run evaluation script for this byte
                cmd = [
                    'python', 'test_ascad_bert_improved_no_leakage.py',
                    '--dataset_path', dataset_info['path'],
                    '--model_path', str(self.model_path),
                    '--key_byte', str(byte_idx),
                    '--num_traces', '10000',
                    '--batch_size', '64',
                    '--profiling_mean_path', str(model_dir / 'profiling_mean.npy'),
                    '--profiling_std_path', str(model_dir / 'profiling_std.npy'),
                    '--test_indices_path', str(model_dir / 'test_indices.npy')
                ]
                
                # Add ablation flag if model was trained with ablation
                ablation_flag = self.detect_ablation_from_model_path()
                if ablation_flag:
                    cmd.extend(['--ablation', ablation_flag])
                
                # Run evaluation
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.logger.error(f"Evaluation failed for {dataset_name} byte {byte_idx}: {result.stderr}")
                    dataset_results[f'byte_{byte_idx:02d}'] = {
                        'byte_index': byte_idx,
                        'key_found': False,
                        'found_at_trace': float('inf'),
                        'best_rank': 255,
                        'error': result.stderr
                    }
                    continue
                
                # Parse results
                traces_to_rank0, best_rank, key_found = self.parse_evaluation_output(result.stdout)
                
                dataset_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': key_found,
                    'found_at_trace': traces_to_rank0 if key_found else float('inf'),
                    'best_rank': best_rank,
                    'dataset_variant': dataset_name,
                    'jitter_level': dataset_info['jitter_level']
                }
                
                if key_found:
                    total_traces_list.append(traces_to_rank0)
                    successful_bytes += 1
                    self.logger.info(f"    Byte {byte_idx}: ✅ Key found at trace {traces_to_rank0}")
                else:
                    self.logger.info(f"    Byte {byte_idx}: ❌ Key not found (best rank: {best_rank})")
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"Evaluation timeout for {dataset_name} byte {byte_idx}")
                dataset_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': False,
                    'found_at_trace': float('inf'),
                    'best_rank': 255,
                    'error': 'Timeout'
                }
            except Exception as e:
                self.logger.error(f"Evaluation error for {dataset_name} byte {byte_idx}: {str(e)}")
                dataset_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': False,
                    'found_at_trace': float('inf'),
                    'best_rank': 255,
                    'error': str(e)
                }
        
        # Calculate summary statistics
        evaluation_time = time.time() - start_time
        
        summary = {
            'dataset_name': dataset_name,
            'dataset_description': dataset_info['description'],
            'jitter_level': dataset_info['jitter_level'],
            'successful_bytes': successful_bytes,
            'failed_bytes': 16 - successful_bytes,
            'success_rate': successful_bytes / 16 * 100,
            'avg_traces_to_find': float(np.mean(total_traces_list)) if total_traces_list else float('inf'),
            'min_traces_to_find': float(min(total_traces_list)) if total_traces_list else float('inf'),
            'max_traces_to_find': float(max(total_traces_list)) if total_traces_list else float('inf'),
            'std_traces': float(np.std(total_traces_list)) if len(total_traces_list) > 1 else 0.0,
            'median_traces': float(np.median(total_traces_list)) if total_traces_list else float('inf'),
            'evaluation_time': evaluation_time
        }
        
        dataset_results['summary'] = summary
        
        # Save dataset-specific results
        results_file = dataset_output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(dataset_results, f, indent=2, default=str)
        
        self.logger.info(f"✅ {dataset_name}: {successful_bytes}/16 bytes successful")
        if total_traces_list:
            self.logger.info(f"    Average traces to rank 0: {np.mean(total_traces_list):.1f}")
        
        return dataset_results
    
    def detect_ablation_from_model_path(self) -> str:
        """Detect ablation type from model path"""
        model_path_str = str(self.model_path).lower()
        if 'no_dropout' in model_path_str:
            return 'no_dropout'
        elif 'no_posenc' in model_path_str or 'no_positional' in model_path_str:
            return 'no_posenc'
        elif 'no_fusion' in model_path_str:
            return 'no_fusion'
        elif 'tiny_emb' in model_path_str or 'tiny_embedding' in model_path_str:
            return 'tiny_emb'
        else:
            return None
    
    def parse_evaluation_output(self, stdout_text: str) -> Tuple[int, int, bool]:
        """Parse evaluation results from stdout"""
        traces_to_rank0 = float('inf')
        best_rank = 255
        key_found = False
        
        try:
            # Look for key found message
            if "Key byte found at trace" in stdout_text or "Key found at trace" in stdout_text:
                key_found = True
                # Extract trace number
                match = re.search(r'found at trace (\d+)', stdout_text)
                if match:
                    traces_to_rank0 = int(match.group(1))
                best_rank = 0
            
            # Look for best rank achieved
            best_rank_match = re.search(r'Best rank achieved: (\d+)', stdout_text)
            if best_rank_match:
                best_rank = int(best_rank_match.group(1))
                
        except Exception as e:
            self.logger.warning(f"Failed to parse evaluation output: {str(e)}")
            
        return traces_to_rank0, best_rank, key_found
    
    def generate_robustness_analysis(self):
        """Generate comprehensive robustness analysis and visualizations"""
        self.logger.info("📊 Generating robustness analysis...")
        
        # Create comprehensive comparison table
        comparison_table = self.create_robustness_table()
        
        # Generate visualizations
        self.generate_robustness_plots()
        
        # Create detailed analysis report
        self.generate_robustness_report(comparison_table)
        
        self.logger.info("✅ Robustness analysis complete")
    
    def create_robustness_table(self) -> pd.DataFrame:
        """Create comprehensive robustness comparison table"""
        table_data = []
        
        for dataset_name, dataset_results in self.results.items():
            summary = dataset_results.get('summary', {})
            
            # Calculate specific metrics for byte-02 (profiled target)
            byte_02_result = dataset_results.get('byte_02', {})
            byte_02_traces = byte_02_result.get('found_at_trace', float('inf'))
            if byte_02_traces == float('inf'):
                byte_02_traces = "Not Found"
            
            table_data.append({
                'Dataset': summary.get('dataset_description', dataset_name),
                'Jitter Level': f"{summary.get('jitter_level', 0)} samples",
                'Success Rate': f"{summary.get('success_rate', 0):.1f}% ({summary.get('successful_bytes', 0)}/16)",
                'Avg. Traces': f"{summary.get('avg_traces_to_find', float('inf')):.1f}" if summary.get('avg_traces_to_find', float('inf')) != float('inf') else "∞",
                'Min Traces': f"{summary.get('min_traces_to_find', float('inf')):.0f}" if summary.get('min_traces_to_find', float('inf')) != float('inf') else "∞",
                'Max Traces': f"{summary.get('max_traces_to_find', float('inf')):.0f}" if summary.get('max_traces_to_find', float('inf')) != float('inf') else "∞",
                'Byte-02 (Profiled)': f"{byte_02_traces}" if isinstance(byte_02_traces, str) else f"{byte_02_traces:.0f}",
                'Evaluation Time': f"{summary.get('evaluation_time', 0):.1f}s"
            })
        
        df = pd.DataFrame(table_data)
        
        # Save table
        df.to_csv(self.output_dir / 'robustness_comparison_table.csv', index=False)
        
        # Save LaTeX table
        latex_table = df.to_latex(index=False, escape=False)
        with open(self.output_dir / 'robustness_table.tex', 'w') as f:
            f.write(latex_table)
        
        return df
    
    def generate_robustness_plots(self):
        """Generate comprehensive robustness visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Success Rate vs Jitter Level
        self._plot_success_rate_degradation()
        
        # 2. Average Traces vs Jitter Level
        self._plot_trace_requirement_increase()
        
        # 3. Per-Byte Performance Heatmap
        self._plot_byte_performance_heatmap()
        
        # 4. Robustness Distribution Analysis
        self._plot_robustness_distributions()
        
        # 5. Byte-02 (Profiled Target) Special Analysis
        self._plot_profiled_target_robustness()
    
    def _plot_success_rate_degradation(self):
        """Plot success rate degradation with increasing jitter"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        jitter_levels = []
        success_rates = []
        dataset_names = []
        
        for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
            if dataset_name in self.results:
                summary = self.results[dataset_name]['summary']
                jitter_levels.append(summary['jitter_level'])
                success_rates.append(summary['success_rate'])
                dataset_names.append(summary['dataset_description'])
        
        # Plot line with markers
        ax.plot(jitter_levels, success_rates, 'o-', linewidth=3, markersize=10, color='#2E86AB')
        
        # Add value labels
        for i, (x, y, name) in enumerate(zip(jitter_levels, success_rates, dataset_names)):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
            ax.annotate(name, (x, y-5), textcoords="offset points", 
                       xytext=(0,-25), ha='center', style='italic')
        
        ax.set_xlabel('Desynchronization Level (samples)', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('BERT Model Robustness: Success Rate vs Temporal Jitter', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trace_requirement_increase(self):
        """Plot average trace requirement increase with jitter"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        jitter_levels = []
        avg_traces = []
        byte_02_traces = []
        dataset_names = []
        
        for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
            if dataset_name in self.results:
                summary = self.results[dataset_name]['summary']
                jitter_levels.append(summary['jitter_level'])
                
                avg_val = summary['avg_traces_to_find']
                avg_traces.append(avg_val if avg_val != float('inf') else None)
                
                byte_02_result = self.results[dataset_name].get('byte_02', {})
                byte_02_val = byte_02_result.get('found_at_trace', float('inf'))
                byte_02_traces.append(byte_02_val if byte_02_val != float('inf') else None)
                
                dataset_names.append(summary['dataset_description'])
        
        # Plot 1: Average traces across all bytes
        valid_avg = [(x, y) for x, y in zip(jitter_levels, avg_traces) if y is not None]
        if valid_avg:
            x_vals, y_vals = zip(*valid_avg)
            ax1.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=10, color='#A23B72')
            
            for x, y in zip(x_vals, y_vals):
                ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
        
        ax1.set_xlabel('Desynchronization Level (samples)', fontsize=12)
        ax1.set_ylabel('Average Traces to Rank 0', fontsize=12)
        ax1.set_title('Average Attack Complexity vs Jitter', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Byte-02 (profiled target) specific
        valid_byte02 = [(x, y) for x, y in zip(jitter_levels, byte_02_traces) if y is not None]
        if valid_byte02:
            x_vals, y_vals = zip(*valid_byte02)
            ax2.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=10, color='#F18F01')
            
            for x, y in zip(x_vals, y_vals):
                ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
        
        ax2.set_xlabel('Desynchronization Level (samples)', fontsize=12)
        ax2.set_ylabel('Traces to Rank 0 (Byte-02)', fontsize=12)
        ax2.set_title('Profiled Target Performance vs Jitter', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_trace_requirements.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_byte_performance_heatmap(self):
        """Plot per-byte performance across datasets"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data matrix
        datasets = ['no_desync', 'desync_50', 'desync_100']
        byte_data = []
        dataset_labels = []
        
        for dataset_name in datasets:
            if dataset_name in self.results:
                row_data = []
                for byte_idx in range(16):
                    byte_result = self.results[dataset_name].get(f'byte_{byte_idx:02d}', {})
                    traces = byte_result.get('found_at_trace', float('inf'))
                    if traces == float('inf'):
                        traces = 10000  # Cap for visualization
                    row_data.append(traces)
                
                byte_data.append(row_data)
                dataset_labels.append(self.results[dataset_name]['summary']['dataset_description'])
        
        if byte_data:
            # Use log scale for better visualization
            log_data = np.log10(np.array(byte_data) + 1)
            
            heatmap = ax.imshow(log_data, cmap='RdYlBu_r', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(16))
            ax.set_xticklabels([f'Byte {i:02d}' for i in range(16)])
            ax.set_yticks(range(len(dataset_labels)))
            ax.set_yticklabels(dataset_labels)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label('Log10(Traces + 1)', rotation=270, labelpad=20)
            
            # Highlight byte-02 (profiled target)
            rect = plt.Rectangle((1.5, -0.5), 1, len(dataset_labels), 
                               fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect)
            ax.text(2, len(dataset_labels), 'Profiled\nTarget', ha='center', va='bottom', 
                   color='red', fontweight='bold')
            
            # Add text annotations for actual values
            for i in range(len(dataset_labels)):
                for j in range(16):
                    value = byte_data[i][j]
                    if value < 10000:  # Only show if found
                        text = ax.text(j, i, f'{int(value)}', ha="center", va="center", 
                                     color="black", fontsize=8)
            
            ax.set_title('Per-Byte Attack Performance Across Desynchronization Levels', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('AES Key Byte Index', fontsize=12)
            ax.set_ylabel('Dataset Variant', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_byte_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_distributions(self):
        """Plot distribution of performance across bytes for each dataset"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_for_violin = []
        labels = []
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, dataset_name in enumerate(['no_desync', 'desync_50', 'desync_100']):
            if dataset_name in self.results:
                traces_list = []
                for byte_idx in range(16):
                    byte_result = self.results[dataset_name].get(f'byte_{byte_idx:02d}', {})
                    if byte_result.get('key_found', False):
                        traces_list.append(byte_result['found_at_trace'])
                
                if traces_list:
                    data_for_violin.append(traces_list)
                    labels.append(self.results[dataset_name]['summary']['dataset_description'])
        
        if data_for_violin:
            violin_parts = ax.violinplot(data_for_violin, positions=range(len(labels)), 
                                       showmeans=True, showmedians=True)
            
            # Customize violin plot colors
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Traces Required for Rank-0 Recovery', fontsize=12)
            ax.set_title('Distribution of Attack Performance Across Desynchronization Levels', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_profiled_target_robustness(self):
        """Special analysis plot for byte-02 (profiled target)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        jitter_levels = []
        byte_02_traces = []
        degradation_factors = []
        
        baseline_traces = None
        
        for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
            if dataset_name in self.results:
                byte_02_result = self.results[dataset_name].get('byte_02', {})
                traces = byte_02_result.get('found_at_trace', float('inf'))
                
                if traces != float('inf'):
                    summary = self.results[dataset_name]['summary']
                    jitter_levels.append(summary['jitter_level'])
                    byte_02_traces.append(traces)
                    
                    if baseline_traces is None:
                        baseline_traces = traces
                        degradation_factors.append(1.0)
                    else:
                        degradation_factors.append(traces / baseline_traces)
        
        if len(byte_02_traces) > 1:
            # Create dual-axis plot
            ax2 = ax.twinx()
            
            # Plot absolute traces
            line1 = ax.plot(jitter_levels, byte_02_traces, 'o-', linewidth=3, markersize=10, 
                           color='#2E86AB', label='Traces Required')
            
            # Plot degradation factor
            line2 = ax2.plot(jitter_levels, degradation_factors, 's--', linewidth=3, markersize=8, 
                            color='#D32F2F', label='Degradation Factor')
            
            # Add value labels
            for x, y in zip(jitter_levels, byte_02_traces):
                ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold', color='#2E86AB')
            
            for x, y in zip(jitter_levels, degradation_factors):
                ax2.annotate(f'{y:.1f}×', (x, y), textcoords="offset points", 
                            xytext=(0,-20), ha='center', fontweight='bold', color='#D32F2F')
            
            ax.set_xlabel('Desynchronization Level (samples)', fontsize=12)
            ax.set_ylabel('Traces to Rank 0 (Byte-02)', fontsize=12, color='#2E86AB')
            ax2.set_ylabel('Performance Degradation Factor', fontsize=12, color='#D32F2F')
            
            # Color the y-axis labels
            ax.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='#D32F2F')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.set_title('Profiled Target (Byte-02) Robustness Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'robustness_profiled_target.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_robustness_report(self, comparison_table: pd.DataFrame):
        """Generate comprehensive robustness analysis report"""
        report_file = self.output_dir / 'robustness_analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Cross-Dataset Robustness Analysis Report\n\n")
            f.write(f"**Evaluation Date:** {self.experiment_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model Evaluated:** {self.model_path.name}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Calculate overall robustness metrics
            baseline_success = None
            for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
                if dataset_name in self.results:
                    summary = self.results[dataset_name]['summary']
                    success_rate = summary['success_rate']
                    
                    if baseline_success is None:
                        baseline_success = success_rate
                        f.write(f"- **Baseline Performance (No Jitter):** {success_rate:.1f}% success rate\n")
                    else:
                        degradation = baseline_success - success_rate
                        f.write(f"- **{summary['dataset_description']}:** {success_rate:.1f}% success rate ({degradation:+.1f}% change)\n")
            
            f.write("\n### Key Findings\n\n")
            
            # Analyze byte-02 (profiled target) specifically
            if 'no_desync' in self.results and 'byte_02' in self.results['no_desync']:
                baseline_byte02 = self.results['no_desync']['byte_02'].get('found_at_trace', float('inf'))
                f.write(f"#### Profiled Target (Byte-02) Analysis:\n\n")
                
                for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
                    if dataset_name in self.results:
                        byte_02_result = self.results[dataset_name].get('byte_02', {})
                        traces = byte_02_result.get('found_at_trace', float('inf'))
                        jitter = self.results[dataset_name]['summary']['jitter_level']
                        
                        if traces != float('inf'):
                            if baseline_byte02 != float('inf') and baseline_byte02 > 0:
                                factor = traces / baseline_byte02
                                f.write(f"- **{jitter}-sample jitter:** {traces:.0f} traces ({factor:.1f}× baseline)\n")
                            else:
                                f.write(f"- **{jitter}-sample jitter:** {traces:.0f} traces\n")
                        else:
                            f.write(f"- **{jitter}-sample jitter:** Key not found\n")
            
            f.write("\n### Robustness Characteristics\n\n")
            
            # General robustness analysis
            f.write("#### Overall Model Behavior:\n\n")
            for dataset_name in ['no_desync', 'desync_50', 'desync_100']:
                if dataset_name in self.results:
                    summary = self.results[dataset_name]['summary']
                    f.write(f"- **{summary['dataset_description']}:**\n")
                    f.write(f"  - Success Rate: {summary['success_rate']:.1f}% ({summary['successful_bytes']}/16 bytes)\n")
                    
                    if summary['avg_traces_to_find'] != float('inf'):
                        f.write(f"  - Average Traces: {summary['avg_traces_to_find']:.1f}\n")
                        f.write(f"  - Range: {summary['min_traces_to_find']:.0f} - {summary['max_traces_to_find']:.0f} traces\n")
                    else:
                        f.write(f"  - Average Traces: Not applicable (insufficient successes)\n")
                    f.write("\n")
            
            f.write("## Detailed Results\n\n")
            f.write(comparison_table.to_markdown(index=False))
            
            f.write("\n\n## Technical Analysis\n\n")
            
            f.write("### Attention Mechanism Robustness\n\n")
            f.write("The BERT-based transformer model demonstrates varying levels of robustness against ")
            f.write("temporal desynchronization countermeasures. The attention mechanism's ability to ")
            f.write("focus on relevant temporal features shows both strengths and limitations when ")
            f.write("faced with increasing levels of trace misalignment.\n\n")
            
            f.write("### Implications for Side-Channel Security\n\n")
            f.write("These results provide insights into the effectiveness of temporal jitter as a ")
            f.write("countermeasure against transformer-based side-channel attacks. The degradation ")
            f.write("patterns observed can inform both attack optimization and defense strategies.\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **For Attackers:** Focus on the profiled target byte where specialized performance is maintained\n")
            f.write("2. **For Defenders:** Higher jitter levels provide measurable protection but may not completely prevent attacks\n")
            f.write("3. **For Researchers:** Consider hybrid approaches combining transformers with preprocessing for jitter compensation\n")
        
        self.logger.info(f"📊 Comprehensive report generated: {report_file}")
    
    def run_complete_evaluation(self):
        """Run the complete cross-dataset robustness evaluation"""
        self.logger.info("🚀 Starting Cross-Dataset Robustness Evaluation")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"🎯 Model: {self.model_path}")
        
        # Verify all required files and datasets
        self.verify_model_and_datasets()
        
        # Evaluate on each dataset
        total_datasets = len(self.datasets)
        completed_datasets = 0
        
        for dataset_name, dataset_info in self.datasets.items():
            self.logger.info(f"📊 Evaluating dataset {completed_datasets + 1}/{total_datasets}: {dataset_name}")
            
            try:
                result = self.evaluate_on_dataset(dataset_name, dataset_info)
                self.results[dataset_name] = result
                
                completed_datasets += 1
                self.logger.info(f"✅ Completed {completed_datasets}/{total_datasets} datasets")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {dataset_name}: {str(e)}")
                completed_datasets += 1
                continue
        
        # Generate comprehensive analysis
        if self.results:
            self.logger.info("📊 Generating robustness analysis...")
            self.generate_robustness_analysis()
        
        # Final summary
        total_time = (datetime.now() - self.experiment_start).total_seconds()
        successful_datasets = len(self.results)
        
        self.logger.info("🎉 Cross-Dataset Evaluation Complete!")
        self.logger.info(f"📊 Results: {successful_datasets}/{total_datasets} datasets evaluated")
        self.logger.info(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
        
        return {
            'results': self.results,
            'total_time': total_time,
            'successful_datasets': successful_datasets,
            'total_datasets': total_datasets
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run cross-dataset robustness evaluation')
    parser.add_argument('--model_path', required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--output_dir', default='robustness_evaluation_results', 
                       help='Output directory for all results')
    parser.add_argument('--datasets_base_dir', default='/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key',
                       help='Base directory containing ASCAD dataset variants')
    parser.add_argument('--dry_run', action='store_true', 
                       help='Print configuration without running evaluation')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN - Robustness Evaluation Preview:")
        print(f"📁 Model: {args.model_path}")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📁 Datasets base directory: {args.datasets_base_dir}")
        
        # Show datasets that will be evaluated
        datasets = {
            'no_desync': f"{args.datasets_base_dir}/ascad-variable.h5",
            'desync_50': f"{args.datasets_base_dir}/ascad-variable-desync50.h5", 
            'desync_100': f"{args.datasets_base_dir}/ascad-variable-desync100.h5"
        }
        
        print(f"\n📊 Datasets to evaluate ({len(datasets)}):")
        for name, path in datasets.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {name}: {path} {exists}")
        
        print(f"\n🎯 Evaluation scope:")
        print(f"  - All 16 AES key bytes per dataset")
        print(f"  - Total evaluations: {len(datasets) * 16} = {len(datasets) * 16}")
        print(f"  - Estimated time: ~{len(datasets) * 15} minutes")
        
        return
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Run the complete robustness evaluation
    evaluator = CrossDatasetEvaluator(args.model_path, args.output_dir)
    
    try:
        final_results = evaluator.run_complete_evaluation()
        
        print("\n🎉 ROBUSTNESS EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"📊 Datasets Evaluated: {final_results['successful_datasets']}/{final_results['total_datasets']}")
        print(f"⏱️ Total Time: {final_results['total_time']/60:.1f} minutes")
        print(f"📁 Results: {args.output_dir}")
        print("\n📈 Generated Files:")
        print("  - robustness_comparison_table.csv (comparison data)")
        print("  - robustness_table.tex (LaTeX table)")
        print("  - robustness_analysis_report.md (comprehensive report)")
        print("  - robustness_*.png (5 visualization plots)")
        print("  - [dataset_name]/evaluation_results.json (detailed results per dataset)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Robustness evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Robustness evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()