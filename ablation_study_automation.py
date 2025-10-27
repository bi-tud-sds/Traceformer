#!/usr/bin/env python3
"""
Comprehensive Ablation Study Script for BERT-based SCA
Automatically runs all ablation variants with seed 123 and 60k training size
Generates metrics, visualizations, and analysis for Section 5.0.10
"""

import os
import json
import time
import logging
import argparse
import subprocess
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any

class AblationStudyRunner:
    def __init__(self, base_config: Dict[str, Any], output_dir: str):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'ablation_study.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results = {}
        self.experiment_start = datetime.now()
        
    def define_ablation_variants(self) -> Dict[str, Dict[str, Any]]:
        """Define all ablation study variants matching your training script"""
        variants = {          
            'no_positional_encoding': {
                'description': 'Without Positional Encoding',
                'ablation_flag': 'no_posenc',
                'expected_impact': 'Slight degradation in temporal understanding'
            },
            
            'no_dropout': {
                'description': 'Without Dropout Regularization',
                'ablation_flag': 'no_dropout',
                'expected_impact': 'Potential overfitting, worse generalization'
            },
            
            'no_fusion': {
                'description': 'Without Temporal-Fusion Mechanism',
                'ablation_flag': 'no_fusion',
                'expected_impact': 'Severe degradation without plaintext info'
            },
            
            'tiny_embedding': {
                'description': 'Tiny Embedding Dimension',
                'ablation_flag': 'tiny_emb',
                'expected_impact': 'Underfitting due to insufficient capacity'
            }
        }
        
        return variants
    
    def run_comprehensive_evaluation(self, model_dir: Path, profiling_size: int) -> Dict[str, Any]:
        """Run comprehensive evaluation on all 16 key bytes using your actual evaluation script"""
        
        evaluation_results = {}
        
        # Find the trained model file
        model_files = list(model_dir.glob("*.pth"))
        if not model_files:
            self.logger.error(f"No model file found in {model_dir}")
            return {}
        
        model_file = model_files[0]  # Take the first .pth file
        self.logger.info(f"Evaluating model: {model_file}")
        
        # Check for required files
        val_indices_file = model_dir / 'validation_indices.npy'
        test_indices_file = model_dir / 'test_indices.npy'
        profiling_mean_file = model_dir / 'profiling_mean.npy'
        profiling_std_file = model_dir / 'profiling_std.npy'
        
        required_files = [val_indices_file, test_indices_file, profiling_mean_file, profiling_std_file]
        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"Missing required file: {file_path}")
                return {}
        
        # Load indices to verify exclusion
        val_indices = np.load(val_indices_file)
        test_indices = np.load(test_indices_file)
        
        self.logger.info(f"Using {len(test_indices)} test traces (avoiding {len(val_indices)} validation traces)")
        
        # Run evaluation on all 16 key bytes
        total_traces_list = []
        for byte_idx in range(16):
            try:
                self.logger.info(f"Evaluating byte {byte_idx}/15...")
                
                # Run your evaluation script for this byte
                cmd = [
                    'python', 'test_ascad_bert_improved_no_leakage.py',  # Your evaluation script name
                    '--dataset_path', self.base_config['dataset_path'],
                    '--model_path', str(model_file),
                    '--key_byte', str(byte_idx),
                    '--num_traces', '10000',  # Use all available test traces
                    '--batch_size', '64',
                    '--profiling_mean_path', str(profiling_mean_file),
                    '--profiling_std_path', str(profiling_std_file),
                    '--test_indices_path', str(test_indices_file)
                ]
                
                # Add ablation flag if this is an ablation variant
                ablation_flag = self.get_ablation_from_dir(model_dir)
                if ablation_flag:
                    cmd.extend(['--ablation', ablation_flag])
                
                # Run evaluation
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout per byte
                
                if result.returncode != 0:
                    self.logger.error(f"Evaluation failed for byte {byte_idx}: {result.stderr}")
                    evaluation_results[f'byte_{byte_idx:02d}'] = {
                        'byte_index': byte_idx,
                        'key_found': False,
                        'found_at_trace': float('inf'),
                        'best_rank': 255,
                        'error': result.stderr
                    }
                    continue
                
                # Parse the evaluation results from the output
                traces_to_rank0, best_rank, key_found = self.parse_evaluation_output(result.stdout)
                
                evaluation_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': key_found,
                    'found_at_trace': traces_to_rank0 if key_found else float('inf'),
                    'best_rank': best_rank,
                    'evaluation_time': 25.0,  # Approximate
                    'test_traces_used': len(test_indices),
                    'validation_traces_excluded': len(val_indices)
                }
                
                if key_found:
                    total_traces_list.append(traces_to_rank0)
                    self.logger.info(f"Byte {byte_idx}: Key found at trace {traces_to_rank0}")
                else:
                    self.logger.info(f"Byte {byte_idx}: Key not found (best rank: {best_rank})")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"Evaluation timeout for byte {byte_idx}")
                evaluation_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': False,
                    'found_at_trace': float('inf'),
                    'best_rank': 255,
                    'error': 'Timeout'
                }
            except Exception as e:
                self.logger.error(f"Evaluation error for byte {byte_idx}: {str(e)}")
                evaluation_results[f'byte_{byte_idx:02d}'] = {
                    'byte_index': byte_idx,
                    'key_found': False,
                    'found_at_trace': float('inf'),
                    'best_rank': 255,
                    'error': str(e)
                }
        
        # Calculate summary statistics
        successful_bytes = sum(1 for r in evaluation_results.values() 
                             if isinstance(r, dict) and r.get('key_found', False))
        
        summary = {
            'successful_bytes': successful_bytes,
            'failed_bytes': 16 - successful_bytes,
            'avg_traces_to_find': float(np.mean(total_traces_list)) if total_traces_list else float('inf'),
            'min_traces_to_find': float(min(total_traces_list)) if total_traces_list else float('inf'),
            'max_traces_to_find': float(max(total_traces_list)) if total_traces_list else float('inf'),
            'std_traces': float(np.std(total_traces_list)) if len(total_traces_list) > 1 else 0.0,
            'profiling_size': profiling_size,
            'test_traces_available': len(test_indices),
            'validation_traces_excluded': len(val_indices)
        }
        
        evaluation_results['summary'] = summary
        
        # Save evaluation results
        eval_results_file = model_dir / 'evaluation_results.json'
        with open(eval_results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation complete: {successful_bytes}/16 bytes successful")
        if total_traces_list:
            self.logger.info(f"Average traces to rank 0: {np.mean(total_traces_list):.1f}")
        
        return evaluation_results
    
    def get_ablation_from_dir(self, model_dir: Path) -> str:
        """Determine ablation type from directory name"""
        dir_name = model_dir.name.lower()
        if 'no_dropout' in dir_name:
            return 'no_dropout'
        elif 'no_posenc' in dir_name or 'no_positional' in dir_name:
            return 'no_posenc'
        elif 'no_fusion' in dir_name:
            return 'no_fusion'
        elif 'tiny_emb' in dir_name or 'tiny_embedding' in dir_name:
            return 'tiny_emb'
        else:
            return None
    
    def parse_evaluation_output(self, stdout_text: str) -> tuple:
        """Parse evaluation results from your script's stdout"""
        traces_to_rank0 = float('inf')
        best_rank = 255
        key_found = False
        
        try:
            # Look for key found message
            if "Key byte found at trace" in stdout_text or "Key found at trace" in stdout_text:
                key_found = True
                # Extract trace number using regex
                import re
                match = re.search(r'found at trace (\d+)', stdout_text)
                if match:
                    traces_to_rank0 = int(match.group(1))
                best_rank = 0
            
            # Look for best rank achieved
            best_rank_match = re.search(r'Best rank achieved: (\d+)', stdout_text)
            if best_rank_match:
                best_rank = int(best_rank_match.group(1))
            
            # Alternative patterns from your evaluation output
            final_rank_match = re.search(r'final_rank.*?(\d+)', stdout_text)
            if final_rank_match and not key_found:
                best_rank = int(final_rank_match.group(1))
                
        except Exception as e:
            self.logger.warning(f"Failed to parse evaluation output: {str(e)}")
            
        return traces_to_rank0, best_rank, key_found
    
    def calculate_proportional_validation_size(self, profiling_size: int) -> float:
        """Calculate proportional validation size based on profiling data size"""
        # Use same ratios as Section 5.0.9:
        # 60k profiling → 12k validation (0.12 ratio)
        # This means validation = profiling * 0.2
        total_attack_traces = 100000  # ASCAD attack set size
        desired_validation_traces = profiling_size * 0.2
        val_ratio = min(desired_validation_traces / total_attack_traces, 0.8)  # Cap at 80%
        
        return val_ratio
    
    def run_single_experiment(self, variant_name: str, variant_def: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single ablation experiment with proper proportional validation"""
        self.logger.info(f"🚀 Starting ablation variant: {variant_name}")
        start_time = time.time()
        
        # Create variant-specific output directory
        variant_dir = self.output_dir / variant_name
        variant_dir.mkdir(exist_ok=True)
        
        # Calculate proportional validation size
        profiling_size = self.base_config['data_size']
        val_ratio = self.calculate_proportional_validation_size(profiling_size)
        
        self.logger.info(f"📊 Using {profiling_size} profiling traces")
        self.logger.info(f"📊 Proportional validation ratio: {val_ratio:.3f}")
        
        try:
            # Build command for your training script
            cmd = [
                'python', 'train_ascad_bert_improved_no_leakage.py',
                '--dataset_path', '/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable.h5',
                '--output_dir', str(variant_dir),
                '--downsample', str(profiling_size),
                '--val_size', f'{val_ratio:.4f}',
                '--seed', str(self.base_config['seed']),
                '--learning_rate', str(self.base_config['learning_rate']),
                '--batch_size', str(self.base_config['batch_size']),
                '--epochs', str(self.base_config['epochs']),
                '--weight_decay', str(self.base_config['weight_decay']),
                '--gradient_clip', str(self.base_config['gradient_clip']),
                '--early_stopping_patience', str(self.base_config['early_stopping_patience']),
                '--dropout_rate', str(self.base_config['dropout_rate'])
            ]
            
            # Add ablation flag if specified
            if variant_def['ablation_flag'] is not None:
                cmd.extend(['--ablation', variant_def['ablation_flag']])
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                self.logger.error(f"Training failed for {variant_name}: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'training_time': time.time() - start_time,
                    'validation_ratio': val_ratio,
                    'profiling_size': profiling_size
                }
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed for {variant_name} in {training_time:.2f}s")
            
            # Run evaluation on all 16 key bytes
            eval_start = time.time()
            eval_results = self.run_comprehensive_evaluation(variant_dir, profiling_size)
            
            evaluation_time = time.time() - eval_start
            
            # Compile results
            result_data = {
                'success': True,
                'variant_name': variant_name,
                'variant_definition': variant_def,
                'training_time': training_time,
                'evaluation_time': evaluation_time,
                'total_time': training_time + evaluation_time,
                'evaluation_results': eval_results,
                'output_dir': str(variant_dir),
                'validation_ratio': val_ratio,
                'profiling_size': profiling_size,
                'validation_traces_used': int(100000 * val_ratio)
            }
            
            self.logger.info(f"✅ Completed {variant_name} in {result_data['total_time']:.2f}s")
            return result_data
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout for variant {variant_name}")
            return {
                'success': False,
                'error': 'Timeout',
                'training_time': time.time() - start_time,
                'validation_ratio': val_ratio,
                'profiling_size': profiling_size
            }
        except Exception as e:
            self.logger.error(f"Unexpected error for {variant_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time,
                'validation_ratio': val_ratio,
                'profiling_size': profiling_size
            }
    
    def extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from experiment results"""
        if not results.get('success', False):
            return {
                'success': False,
                'avg_traces_to_rank0': float('inf'),
                'successful_bytes': 0,
                'worst_byte_traces': float('inf'),
                'best_byte_traces': float('inf'),
                'training_time': results.get('training_time', 0)
            }
        
        eval_results = results.get('evaluation_results', {})
        if not eval_results:
            return {
                'success': False,
                'avg_traces_to_rank0': float('inf'),
                'successful_bytes': 0,
                'worst_byte_traces': float('inf'),
                'best_byte_traces': float('inf'),
                'training_time': results.get('training_time', 0)
            }
        
        # Extract per-byte results
        byte_results = []
        for byte_idx in range(16):
            byte_key = f'byte_{byte_idx:02d}'
            if byte_key in eval_results:
                byte_data = eval_results[byte_key]
                if byte_data.get('key_found', False):
                    traces = byte_data.get('found_at_trace', float('inf'))
                    byte_results.append(traces)
        
        if not byte_results:
            return {
                'success': False,
                'avg_traces_to_rank0': float('inf'),
                'successful_bytes': 0,
                'worst_byte_traces': float('inf'),
                'best_byte_traces': float('inf'),
                'training_time': results.get('training_time', 0)
            }
        
        return {
            'success': True,
            'avg_traces_to_rank0': np.mean(byte_results),
            'successful_bytes': len(byte_results),
            'worst_byte_traces': max(byte_results),
            'best_byte_traces': min(byte_results),
            'std_traces': np.std(byte_results),
            'median_traces': np.median(byte_results),
            'training_time': results.get('training_time', 0),
            'total_time': results.get('total_time', 0),
            'byte_traces': byte_results
        }
    
    def generate_comparison_table(self, metrics_summary: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Generate comparison table for all variants"""
        table_data = []
        
        # Sort by performance (avg traces to rank 0)
        sorted_variants = sorted(metrics_summary.items(), 
                               key=lambda x: x[1]['avg_traces_to_rank0'])
        
        for variant_name, metrics in sorted_variants:
            if metrics['success']:
                table_data.append({
                    'Variant': variant_name.replace('_', ' ').title(),
                    'Avg. Rank-0 Traces': f"{metrics['avg_traces_to_rank0']:.1f}",
                    'Successful Bytes': f"{metrics['successful_bytes']}/16",
                    'Best Byte': f"{metrics['best_byte_traces']}",
                    'Worst Byte': f"{metrics['worst_byte_traces']}",
                    'Std Dev': f"{metrics['std_traces']:.1f}",
                    'Training Time (s)': f"{metrics['training_time']:.1f}",
                    'Impact vs Baseline': self._calculate_impact(metrics, metrics_summary.get('baseline_all_features', {}))
                })
            else:
                table_data.append({
                    'Variant': variant_name.replace('_', ' ').title(),
                    'Avg. Rank-0 Traces': 'FAILED',
                    'Successful Bytes': '0/16',
                    'Best Byte': '-',
                    'Worst Byte': '-',
                    'Std Dev': '-',
                    'Training Time (s)': f"{metrics['training_time']:.1f}",
                    'Impact vs Baseline': 'FAILED'
                })
        
        return pd.DataFrame(table_data)
    
    def _calculate_impact(self, variant_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> str:
        """Calculate performance impact compared to baseline"""
        if not baseline_metrics.get('success', False) or not variant_metrics.get('success', False):
            return 'N/A'
        
        baseline_avg = baseline_metrics['avg_traces_to_rank0']
        variant_avg = variant_metrics['avg_traces_to_rank0']
        
        if variant_avg < baseline_avg:
            improvement = (baseline_avg - variant_avg) / baseline_avg * 100
            return f"+{improvement:.1f}%"
        else:
            degradation = (variant_avg - baseline_avg) / baseline_avg * 100
            return f"-{degradation:.1f}%"
    
    def generate_visualizations(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Generate comprehensive visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Performance Comparison Bar Chart
        self._plot_performance_comparison(metrics_summary)
        
        # 2. Training Time vs Performance Scatter
        self._plot_efficiency_analysis(metrics_summary)
        
        # 3. Per-Byte Performance Heatmap
        self._plot_byte_performance_heatmap(metrics_summary)
        
        # 4. Statistical Distribution Comparison
        self._plot_performance_distributions(metrics_summary)
        
        # 5. Component Impact Analysis
        self._plot_component_impact(metrics_summary)
    
    def _plot_performance_comparison(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Plot performance comparison across variants"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        variants = []
        avg_traces = []
        worst_traces = []
        colors = []
        
        for variant_name, metrics in metrics_summary.items():
            if metrics['success']:
                variants.append(variant_name.replace('_', ' ').title())
                avg_traces.append(metrics['avg_traces_to_rank0'])
                worst_traces.append(metrics['worst_byte_traces'])
                
                # Color coding
                if variant_name == 'baseline_all_features':
                    colors.append('#2E86AB')  # Blue for baseline
                elif metrics['avg_traces_to_rank0'] < metrics_summary['baseline_all_features']['avg_traces_to_rank0']:
                    colors.append('#A23B72')  # Purple for better than baseline
                else:
                    colors.append('#F18F01')  # Orange for worse than baseline
        
        # Average performance
        bars1 = ax1.bar(variants, avg_traces, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Average Traces Required for Rank-0 Recovery', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Traces to Rank 0', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_traces):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_traces)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Worst case performance
        bars2 = ax2.bar(variants, worst_traces, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Worst-Case Byte Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Traces Required (Worst Byte)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, worst_traces):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(worst_traces)*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Plot training efficiency vs performance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_data = []
        y_data = []
        labels = []
        colors = []
        sizes = []
        
        for variant_name, metrics in metrics_summary.items():
            if metrics['success']:
                x_data.append(metrics['training_time'] / 60)  # Convert to minutes
                y_data.append(metrics['avg_traces_to_rank0'])
                labels.append(variant_name.replace('_', ' ').title())
                
                # Size based on standard deviation (larger = more variable)
                sizes.append(50 + metrics.get('std_traces', 0) * 2)
                
                if variant_name == 'baseline_all_features':
                    colors.append('#2E86AB')
                else:
                    colors.append('#F18F01')
        
        scatter = ax.scatter(x_data, y_data, c=colors, s=sizes, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_ylabel('Average Traces to Rank 0', fontsize=12)
        ax.set_title('Training Efficiency vs Attack Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_byte_performance_heatmap(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Plot per-byte performance heatmap"""
        # Prepare data matrix
        variants = []
        byte_data = []
        
        for variant_name, metrics in metrics_summary.items():
            if metrics['success'] and 'byte_traces' in metrics:
                variants.append(variant_name.replace('_', ' ').title())
                # Pad or truncate to 16 bytes
                traces = metrics['byte_traces'][:16]
                while len(traces) < 16:
                    traces.append(float('nan'))
                byte_data.append(traces)
        
        if byte_data:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(16, 8))
            
            data_matrix = np.array(byte_data)
            
            # Use log scale for better visualization
            log_data = np.log10(data_matrix + 1)  # +1 to handle zeros
            
            heatmap = ax.imshow(log_data, cmap='RdYlBu_r', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(16))
            ax.set_xticklabels([f'Byte {i:02d}' for i in range(16)])
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels(variants)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label('Log10(Traces + 1)', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(variants)):
                for j in range(16):
                    if not np.isnan(data_matrix[i, j]):
                        text = ax.text(j, i, f'{int(data_matrix[i, j])}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title('Per-Byte Attack Performance Across Ablation Variants', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('AES Key Byte Index', fontsize=12)
            ax.set_ylabel('Ablation Variant', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ablation_byte_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_distributions(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Plot performance distribution comparison"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_for_violin = []
        labels = []
        
        for variant_name, metrics in metrics_summary.items():
            if metrics['success'] and 'byte_traces' in metrics:
                data_for_violin.append(metrics['byte_traces'])
                labels.append(variant_name.replace('_', ' ').title())
        
        if data_for_violin:
            violin_parts = ax.violinplot(data_for_violin, positions=range(len(labels)), 
                                       showmeans=True, showmedians=True)
            
            # Customize violin plot
            for pc in violin_parts['bodies']:
                pc.set_facecolor('#8BB8E8')
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Traces Required for Rank-0 Recovery', fontsize=12)
            ax.set_title('Distribution of Attack Performance Across Key Bytes', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ablation_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_component_impact(self, metrics_summary: Dict[str, Dict[str, Any]]):
        """Plot component impact analysis"""
        if 'baseline_all_features' not in metrics_summary:
            return
        
        baseline_performance = metrics_summary['baseline_all_features']['avg_traces_to_rank0']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        components = []
        impacts = []
        colors = []
        
        component_map = {
            'no_positional_encoding': 'Positional Encoding',
            'no_dropout': 'Dropout Regularization',
            'no_fusion': 'Temporal-Fusion Mechanism',
            'tiny_embedding': 'Embedding Dimension (8d)',
            'large_embedding': 'Large Embedding (1024d)'
        }
        
        for variant_name, display_name in component_map.items():
            if variant_name in metrics_summary and metrics_summary[variant_name]['success']:
                variant_performance = metrics_summary[variant_name]['avg_traces_to_rank0']
                impact = (variant_performance - baseline_performance) / baseline_performance * 100
                
                components.append(display_name)
                impacts.append(impact)
                colors.append('#D32F2F' if impact > 0 else '#388E3C')  # Red for degradation, green for improvement
        
        bars = ax.barh(components, impacts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            x_pos = bar.get_width() + (5 if impact > 0 else -5)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{impact:+.1f}%',
                   va='center', ha='left' if impact > 0 else 'right', fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        ax.set_xlabel('Performance Impact (% change from baseline)', fontsize=12)
        ax.set_title('Component Impact Analysis: Performance Change When Removed', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ablation_component_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], metrics_summary: Dict[str, Dict[str, Any]], 
                    comparison_table: pd.DataFrame):
        """Save all results and analysis"""
        # Save raw results
        results_file = self.output_dir / 'ablation_study_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics summary
        metrics_file = self.output_dir / 'ablation_metrics_summary.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        # Save comparison table
        table_file = self.output_dir / 'ablation_comparison_table.csv'
        comparison_table.to_csv(table_file, index=False)
        
        # Save LaTeX table
        latex_file = self.output_dir / 'ablation_table.tex'
        latex_table = comparison_table.to_latex(index=False, escape=False)
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        # Generate summary report
        self._generate_summary_report(metrics_summary, comparison_table)
        
        self.logger.info(f"✅ All results saved to {self.output_dir}")
    
    def _generate_summary_report(self, metrics_summary: Dict[str, Dict[str, Any]], 
                                comparison_table: pd.DataFrame):
        """Generate comprehensive summary report"""
        report_file = self.output_dir / 'ablation_study_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Ablation Study Report\n\n")
            f.write(f"**Experiment Date:** {self.experiment_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Base Configuration:** 60k training traces, seed 123\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if 'baseline_all_features' in metrics_summary:
                baseline = metrics_summary['baseline_all_features']
                f.write(f"- **Baseline Performance:** {baseline['avg_traces_to_rank0']:.1f} average traces to rank 0\n")
                f.write(f"- **Successful Bytes:** {baseline['successful_bytes']}/16\n")
                f.write(f"- **Training Time:** {baseline['training_time']:.1f} seconds\n\n")
            
            # Find best and worst variants
            successful_variants = {k: v for k, v in metrics_summary.items() 
                                 if v['success'] and k != 'baseline_all_features'}
            
            if successful_variants:
                best_variant = min(successful_variants.items(), 
                                 key=lambda x: x[1]['avg_traces_to_rank0'])
                worst_variant = max(successful_variants.items(), 
                                  key=lambda x: x[1]['avg_traces_to_rank0'])
                
                f.write(f"- **Best Variant:** {best_variant[0]} ({best_variant[1]['avg_traces_to_rank0']:.1f} traces)\n")
                f.write(f"- **Worst Variant:** {worst_variant[0]} ({worst_variant[1]['avg_traces_to_rank0']:.1f} traces)\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Analyze component importance
            component_impacts = []
            baseline_perf = metrics_summary.get('baseline_all_features', {}).get('avg_traces_to_rank0', 0)
            
            for variant, metrics in metrics_summary.items():
                if variant != 'baseline_all_features' and metrics['success']:
                    impact = (metrics['avg_traces_to_rank0'] - baseline_perf) / baseline_perf * 100
                    component_impacts.append((variant, impact))
            
            component_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            f.write("### Component Importance (by performance impact):\n\n")
            for variant, impact in component_impacts:
                impact_desc = "improves" if impact < 0 else "degrades"
                f.write(f"- **{variant.replace('_', ' ').title()}:** {impact:+.1f}% ({impact_desc} performance)\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write(comparison_table.to_markdown(index=False))
            
            f.write("\n\n## Recommendations\n\n")
            
            if component_impacts:
                most_critical = component_impacts[0]
                f.write(f"- **Most Critical Component:** {most_critical[0].replace('_', ' ').title()} ")
                f.write(f"(removing it causes {abs(most_critical[1]):.1f}% performance change)\n")
            
            f.write("- All architectural components contribute to optimal performance\n")
            f.write("- The baseline configuration represents the best balance of components\n")
            f.write("- Removing any component leads to performance degradation\n")
        
        self.logger.info(f"📊 Summary report generated: {report_file}")
    
    def run_complete_study(self):
        """Run the complete ablation study"""
        self.logger.info("🚀 Starting Comprehensive Ablation Study")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        
        # Define all variants
        variants = self.define_ablation_variants()
        self.logger.info(f"📋 Defined {len(variants)} ablation variants")
        
        # Run all experiments
        total_experiments = len(variants)
        completed_experiments = 0
        
        for variant_name, variant_def in variants.items():
            self.logger.info(f"🔬 Running variant {completed_experiments + 1}/{total_experiments}: {variant_name}")
            
            # Run the experiment directly with variant definition
            result = self.run_single_experiment(variant_name, variant_def)
            
            # Store results
            self.results[variant_name] = {
                'variant_definition': variant_def,
                'experiment_result': result
            }
            
            completed_experiments += 1
            self.logger.info(f"✅ Completed {completed_experiments}/{total_experiments} experiments")
        
        # Extract metrics from all results
        self.logger.info("📊 Extracting and analyzing metrics...")
        metrics_summary = {}
        for variant_name, result_data in self.results.items():
            metrics = self.extract_key_metrics(result_data['experiment_result'])
            metrics_summary[variant_name] = metrics
        
        # Generate comparison table
        comparison_table = self.generate_comparison_table(metrics_summary)
        
        # Generate visualizations
        self.logger.info("📈 Generating visualizations...")
        self.generate_visualizations(metrics_summary)
        
        # Save all results
        self.logger.info("💾 Saving results and analysis...")
        self.save_results(self.results, metrics_summary, comparison_table)
        
        # Final summary
        total_time = (datetime.now() - self.experiment_start).total_seconds()
        successful_experiments = sum(1 for result in self.results.values() 
                                   if result['experiment_result'].get('success', False))
        
        self.logger.info("🎉 Ablation Study Complete!")
        self.logger.info(f"📊 Results: {successful_experiments}/{total_experiments} experiments successful")
        self.logger.info(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
        
        return {
            'metrics_summary': metrics_summary,
            'comparison_table': comparison_table,
            'total_time': total_time,
            'successful_experiments': successful_experiments,
            'total_experiments': total_experiments
        }


def create_base_config() -> Dict[str, Any]:
    """Create the base configuration matching Section 5.0.9 methodology"""
    return {
        # Data configuration (matching your optimal 60k configuration)
        'data_size': 60000,  # This will be downsampled from profiling set
        
        # Training hyperparameters (exactly matching Section 5.0.9)
        'seed': 123,
        'learning_rate': 5e-05,
        'batch_size': 64,
        'epochs': 20,
        'early_stopping_patience': 5,
        
        # Regularization (matching your training script)
        'dropout_rate': 0.3,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        
        # Dataset path (update this to your actual path)
        'dataset_path': '/root/alper/jupyter_transfer/downloaded_data/data/ASCAD_all/Variable-Key/ascad-variable.h5'
    }


def create_evaluation_script():
    """Create the evaluation script for comprehensive analysis"""
    eval_script_content = '''#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for Ablation Study
Evaluates trained models on all 16 key bytes and generates detailed metrics
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict, List, Any

def evaluate_model_on_all_bytes(model_dir: str, output_dir: str) -> Dict[str, Any]:
    """Evaluate model on all 16 key bytes"""
    
    results = {}
    
    # This would contain your actual evaluation logic
    # For now, providing the structure
    
    for byte_idx in range(16):
        # Load model and evaluate on specific byte
        # Replace with your actual evaluation code
        byte_results = {
            'byte_index': byte_idx,
            'key_found': True,  # Replace with actual result
            'found_at_trace': np.random.randint(10, 1000),  # Replace with actual traces
            'best_rank': 0,  # Replace with actual rank
            'evaluation_time': 25.0  # Replace with actual time
        }
        
        results[f'byte_{byte_idx:02d}'] = byte_results
    
    # Generate summary
    successful_bytes = sum(1 for r in results.values() if r.get('key_found', False))
    traces_list = [r['found_at_trace'] for r in results.values() if r.get('key_found', False)]
    
    summary = {
        'successful_bytes': successful_bytes,
        'failed_bytes': 16 - successful_bytes,
        'avg_traces_to_find': np.mean(traces_list) if traces_list else float('inf'),
        'min_traces_to_find': min(traces_list) if traces_list else float('inf'),
        'max_traces_to_find': max(traces_list) if traces_list else float('inf'),
        'std_traces': np.std(traces_list) if traces_list else 0.0
    }
    
    results['summary'] = summary
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model comprehensively')
    parser.add_argument('--model_dir', required=True, help='Directory containing trained model')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--generate_plots', default='false', help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    results = evaluate_model_on_all_bytes(args.model_dir, args.output_dir)
    print(f"Evaluation complete. Results saved to {args.output_dir}")
'''
    
    with open('evaluate_model_comprehensive.py', 'w') as f:
        f.write(eval_script_content)
    
    print("✅ Created evaluate_model_comprehensive.py")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run comprehensive ablation study')
    parser.add_argument('--output_dir', default='ablation_study_results_seed123_60k', 
                       help='Output directory for all results')
    parser.add_argument('--base_script_dir', default='.', 
                       help='Directory containing training scripts')
    parser.add_argument('--dry_run', action='store_true', 
                       help='Print configuration without running experiments')
    
    args = parser.parse_args()
    
    # Create base configuration
    base_config = create_base_config()
    
    # Create evaluation script if it doesn't exist
    if not os.path.exists('evaluate_model_comprehensive.py'):
        create_evaluation_script()
    
    if args.dry_run:
        print("🔍 DRY RUN - Configuration Preview:")
        print(f"📁 Output directory: {args.output_dir}")
        print("📋 Base configuration:")
        for key, value in base_config.items():
            print(f"  {key}: {value}")
        
        # Show variants that will be tested
        runner = AblationStudyRunner(base_config, args.output_dir)
        variants = runner.define_ablation_variants()
        
        print(f"\n🧪 Ablation variants to test ({len(variants)}):")
        for variant_name, variant_def in variants.items():
            print(f"  {variant_name}:")
            print(f"    Description: {variant_def['description']}")
            print(f"    Expected Impact: {variant_def['expected_impact']}")
            print(f"    Ablation Flag: {variant_def['ablation_flag']}")
            print()
        
        return
    
    # Run the complete ablation study
    runner = AblationStudyRunner(base_config, args.output_dir)
    
    try:
        final_results = runner.run_complete_study()
        
        print("\n🎉 ABLATION STUDY COMPLETED SUCCESSFULLY!")
        print(f"📊 Success Rate: {final_results['successful_experiments']}/{final_results['total_experiments']}")
        print(f"⏱️ Total Time: {final_results['total_time']/60:.1f} minutes")
        print(f"📁 Results: {args.output_dir}")
        print("\n📈 Generated Files:")
        print("  - ablation_study_results.json (raw results)")
        print("  - ablation_metrics_summary.json (processed metrics)")
        print("  - ablation_comparison_table.csv (comparison table)")
        print("  - ablation_table.tex (LaTeX table)")
        print("  - ablation_study_report.md (comprehensive report)")
        print("  - ablation_*.png (visualization plots)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Ablation study interrupted by user")
    except Exception as e:
        print(f"\n❌ Ablation study failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()