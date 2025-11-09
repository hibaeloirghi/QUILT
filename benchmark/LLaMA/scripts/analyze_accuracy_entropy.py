#!/usr/bin/env python3
"""
Accuracy and Entropy Analysis Script
Compares results between no-sampling and samples5 conditions
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def load_entropy_data(directory: str) -> Dict[str, Dict]:
    """Load all entropy JSON files from a directory"""
    data = {}
    entropy_files = glob.glob(os.path.join(directory, "*_entropy.json"))
    
    for file_path in sorted(entropy_files):
        qid = os.path.basename(file_path).replace("_entropy.json", "")
        try:
            with open(file_path, 'r') as f:
                data[qid] = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return data

def compute_answer_confidence(logprobs_list: List[List]) -> Tuple[float, float]:
    """
    Compute average probability and confidence metrics from answer logprobs
    Returns: (avg_prob, min_prob, max_prob)
    """
    if not logprobs_list or len(logprobs_list) == 0:
        return None, None, None
    
    sequence_probs = []
    for logprobs in logprobs_list:
        if not logprobs:
            continue
        
        # Extract probabilities
        if isinstance(logprobs[0], dict):
            probs = [item.get("prob", np.exp(item.get("logprob", 0))) for item in logprobs]
        else:
            probs = [np.exp(p) if isinstance(p, (int, float)) else 0 for p in logprobs]
        
        if probs:
            # Sequence probability is product of token probabilities
            seq_prob = np.prod(probs)
            sequence_probs.append(seq_prob)
    
    if not sequence_probs:
        return None, None, None
    
    return np.mean(sequence_probs), np.min(sequence_probs), np.max(sequence_probs)

def analyze_condition(data: Dict[str, Dict], condition_name: str) -> Dict:
    """Analyze a single condition (no-sampling or samples5)"""
    results = {
        'condition': condition_name,
        'total': len(data),
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0,
        'entropy_metrics': {
            'predictive_entropy': [],
            'semantic_entropy': [],
            'sta_predictive': [],
            'sta_semantic': [],
        },
        'confidence_metrics': {
            'avg_prob': [],
            'min_prob': [],
            'max_prob': [],
        },
        'correct_entropy': {
            'predictive_entropy': [],
            'semantic_entropy': [],
            'sta_predictive': [],
            'sta_semantic': [],
        },
        'incorrect_entropy': {
            'predictive_entropy': [],
            'semantic_entropy': [],
            'sta_predictive': [],
            'sta_semantic': [],
        },
        'correct_confidence': {
            'avg_prob': [],
            'min_prob': [],
            'max_prob': [],
        },
        'incorrect_confidence': {
            'avg_prob': [],
            'min_prob': [],
            'max_prob': [],
        },
        'step_entropies': [],
        'action_entropies': [],
        'tool_entropies': [],
    }
    
    for qid, entry in data.items():
        # Accuracy
        is_correct = entry.get('correct', False)
        if is_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1
        
        # Entropy metrics
        pred_ent = entry.get('predictive_entropy')
        sem_ent = entry.get('semantic_entropy')
        sta_pred = entry.get('sta_predictive')
        sta_sem = entry.get('sta_semantic')
        
        if pred_ent is not None:
            results['entropy_metrics']['predictive_entropy'].append(pred_ent)
            if is_correct:
                results['correct_entropy']['predictive_entropy'].append(pred_ent)
            else:
                results['incorrect_entropy']['predictive_entropy'].append(pred_ent)
        
        if sem_ent is not None:
            results['entropy_metrics']['semantic_entropy'].append(sem_ent)
            if is_correct:
                results['correct_entropy']['semantic_entropy'].append(sem_ent)
            else:
                results['incorrect_entropy']['semantic_entropy'].append(sem_ent)
        
        if sta_pred is not None:
            results['entropy_metrics']['sta_predictive'].append(sta_pred)
            if is_correct:
                results['correct_entropy']['sta_predictive'].append(sta_pred)
            else:
                results['incorrect_entropy']['sta_predictive'].append(sta_pred)
        
        if sta_sem is not None:
            results['entropy_metrics']['sta_semantic'].append(sta_sem)
            if is_correct:
                results['correct_entropy']['sta_semantic'].append(sta_sem)
            else:
                results['incorrect_entropy']['sta_semantic'].append(sta_sem)
        
        # Confidence metrics from answer logprobs
        answer_logprobs = entry.get('answer_logprobs', [])
        if answer_logprobs:
            avg_prob, min_prob, max_prob = compute_answer_confidence(answer_logprobs)
            if avg_prob is not None:
                results['confidence_metrics']['avg_prob'].append(avg_prob)
                if is_correct:
                    results['correct_confidence']['avg_prob'].append(avg_prob)
                else:
                    results['incorrect_confidence']['avg_prob'].append(avg_prob)
            
            if min_prob is not None:
                results['confidence_metrics']['min_prob'].append(min_prob)
                if is_correct:
                    results['correct_confidence']['min_prob'].append(min_prob)
                else:
                    results['incorrect_confidence']['min_prob'].append(min_prob)
            
            if max_prob is not None:
                results['confidence_metrics']['max_prob'].append(max_prob)
                if is_correct:
                    results['correct_confidence']['max_prob'].append(max_prob)
                else:
                    results['incorrect_confidence']['max_prob'].append(max_prob)
        
        # Step-wise entropies
        step_entropies = entry.get('step_entropies', [])
        for step in step_entropies:
            if step.get('predictive_entropy') is not None:
                results['step_entropies'].append(step['predictive_entropy'])
        
        # Action entropies
        action_samples = entry.get('action_samples', [])
        for action in action_samples:
            selected_entropy = action.get('selected_entropy')
            if selected_entropy is not None and selected_entropy != float('inf'):
                results['action_entropies'].append(selected_entropy)
        
        # Tool entropies
        tool_entropies = entry.get('tool_entropies', [])
        results['tool_entropies'].extend(tool_entropies)
    
    # Compute accuracy
    if results['total'] > 0:
        results['accuracy'] = results['correct'] / results['total']
    
    return results

def compute_statistics(values: List[float]) -> Dict:
    """Compute statistics for a list of values"""
    if not values or len(values) == 0:
        return {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None, 'count': 0}
    
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'count': len(values)
    }

def print_comparison(no_sampling: Dict, samples5: Dict):
    """Print detailed comparison between conditions"""
    print("=" * 80)
    print("ACCURACY ANALYSIS: NO-SAMPLING vs SAMPLES5")
    print("=" * 80)
    print()
    
    # Accuracy comparison
    print("ACCURACY METRICS")
    print("-" * 80)
    print(f"{'Metric':<30} {'No-Sampling':<25} {'Samples5':<25}")
    print("-" * 80)
    print(f"{'Total Questions':<30} {no_sampling['total']:<25} {samples5['total']:<25}")
    print(f"{'Correct':<30} {no_sampling['correct']:<25} {samples5['correct']:<25}")
    print(f"{'Incorrect':<30} {no_sampling['incorrect']:<25} {samples5['incorrect']:<25}")
    print(f"{'Accuracy':<30} {no_sampling['accuracy']:.4f} ({no_sampling['accuracy']*100:.2f}%){'':<15} {samples5['accuracy']:.4f} ({samples5['accuracy']*100:.2f}%)")
    
    accuracy_diff = samples5['accuracy'] - no_sampling['accuracy']
    print(f"{'Accuracy Difference':<30} {'':<25} {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
    print()
    
    # Entropy metrics comparison
    print("ENTROPY METRICS (Overall)")
    print("-" * 80)
    print(f"{'Metric':<30} {'No-Sampling':<25} {'Samples5':<25}")
    print("-" * 80)
    
    for metric in ['predictive_entropy', 'semantic_entropy', 'sta_predictive', 'sta_semantic']:
        no_vals = no_sampling['entropy_metrics'][metric]
        sam_vals = samples5['entropy_metrics'][metric]
        
        if no_vals and sam_vals:
            no_stats = compute_statistics(no_vals)
            sam_stats = compute_statistics(sam_vals)
            print(f"{metric:<30} Mean: {no_stats['mean']:.4f} (n={no_stats['count']}){'':<5} Mean: {sam_stats['mean']:.4f} (n={sam_stats['count']})")
            print(f"{'':<30} Med: {no_stats['median']:.4f}, Std: {no_stats['std']:.4f}{'':<5} Med: {sam_stats['median']:.4f}, Std: {sam_stats['std']:.4f}")
        else:
            print(f"{metric:<30} {'N/A':<25} {'N/A':<25}")
        print()
    
    # Entropy by correctness
    print("ENTROPY METRICS BY CORRECTNESS")
    print("-" * 80)
    print(f"{'Metric':<30} {'No-Sampling (Correct)':<25} {'No-Sampling (Incorrect)':<25} {'Samples5 (Correct)':<25} {'Samples5 (Incorrect)':<25}")
    print("-" * 80)
    
    for metric in ['predictive_entropy', 'semantic_entropy', 'sta_predictive', 'sta_semantic']:
        no_correct = no_sampling['correct_entropy'][metric]
        no_incorrect = no_sampling['incorrect_entropy'][metric]
        sam_correct = samples5['correct_entropy'][metric]
        sam_incorrect = samples5['incorrect_entropy'][metric]
        
        if no_correct or no_incorrect or sam_correct or sam_incorrect:
            no_c_stats = compute_statistics(no_correct) if no_correct else {'mean': None, 'count': 0}
            no_i_stats = compute_statistics(no_incorrect) if no_incorrect else {'mean': None, 'count': 0}
            sam_c_stats = compute_statistics(sam_correct) if sam_correct else {'mean': None, 'count': 0}
            sam_i_stats = compute_statistics(sam_incorrect) if sam_incorrect else {'mean': None, 'count': 0}
            
            print(f"{metric:<30} ", end="")
            print(f"{no_c_stats['mean']:.4f} (n={no_c_stats['count']}){'':<10} " if no_c_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{no_i_stats['mean']:.4f} (n={no_i_stats['count']}){'':<10} " if no_i_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{sam_c_stats['mean']:.4f} (n={sam_c_stats['count']}){'':<10} " if sam_c_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{sam_i_stats['mean']:.4f} (n={sam_i_stats['count']})" if sam_i_stats['mean'] is not None else "N/A")
        print()
    
    # Confidence metrics
    print("CONFIDENCE METRICS (Answer Probability)")
    print("-" * 80)
    print(f"{'Metric':<30} {'No-Sampling':<25} {'Samples5':<25}")
    print("-" * 80)
    
    for metric in ['avg_prob', 'min_prob', 'max_prob']:
        no_vals = no_sampling['confidence_metrics'][metric]
        sam_vals = samples5['confidence_metrics'][metric]
        
        if no_vals and sam_vals:
            no_stats = compute_statistics(no_vals)
            sam_stats = compute_statistics(sam_vals)
            print(f"{metric:<30} Mean: {no_stats['mean']:.6f} (n={no_stats['count']}){'':<5} Mean: {sam_stats['mean']:.6f} (n={sam_stats['count']})")
        else:
            print(f"{metric:<30} {'N/A':<25} {'N/A':<25}")
        print()
    
    # Confidence by correctness
    print("CONFIDENCE METRICS BY CORRECTNESS")
    print("-" * 80)
    print(f"{'Metric':<30} {'No-Sampling (Correct)':<25} {'No-Sampling (Incorrect)':<25} {'Samples5 (Correct)':<25} {'Samples5 (Incorrect)':<25}")
    print("-" * 80)
    
    for metric in ['avg_prob', 'min_prob', 'max_prob']:
        no_correct = no_sampling['correct_confidence'][metric]
        no_incorrect = no_sampling['incorrect_confidence'][metric]
        sam_correct = samples5['correct_confidence'][metric]
        sam_incorrect = samples5['incorrect_confidence'][metric]
        
        if no_correct or no_incorrect or sam_correct or sam_incorrect:
            no_c_stats = compute_statistics(no_correct) if no_correct else {'mean': None, 'count': 0}
            no_i_stats = compute_statistics(no_incorrect) if no_incorrect else {'mean': None, 'count': 0}
            sam_c_stats = compute_statistics(sam_correct) if sam_correct else {'mean': None, 'count': 0}
            sam_i_stats = compute_statistics(sam_incorrect) if sam_incorrect else {'mean': None, 'count': 0}
            
            print(f"{metric:<30} ", end="")
            print(f"{no_c_stats['mean']:.6f} (n={no_c_stats['count']}){'':<10} " if no_c_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{no_i_stats['mean']:.6f} (n={no_i_stats['count']}){'':<10} " if no_i_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{sam_c_stats['mean']:.6f} (n={sam_c_stats['count']}){'':<10} " if sam_c_stats['mean'] is not None else f"{'N/A':<25} ", end="")
            print(f"{sam_i_stats['mean']:.6f} (n={sam_i_stats['count']})" if sam_i_stats['mean'] is not None else "N/A")
        print()
    
    # Step and action entropies
    print("STEP-WISE AND ACTION ENTROPIES")
    print("-" * 80)
    
    if no_sampling['step_entropies'] or samples5['step_entropies']:
        no_step_stats = compute_statistics(no_sampling['step_entropies'])
        sam_step_stats = compute_statistics(samples5['step_entropies'])
        print(f"{'Step Entropy (avg per step)':<30} ", end="")
        print(f"{no_step_stats['mean']:.4f} (n={no_step_stats['count']}){'':<10} " if no_step_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_step_stats['mean']:.4f} (n={sam_step_stats['count']})" if sam_step_stats['mean'] is not None else "N/A")
        print()
    
    if no_sampling['action_entropies'] or samples5['action_entropies']:
        no_action_stats = compute_statistics(no_sampling['action_entropies'])
        sam_action_stats = compute_statistics(samples5['action_entropies'])
        print(f"{'Action Entropy (selected)':<30} ", end="")
        print(f"{no_action_stats['mean']:.4f} (n={no_action_stats['count']}){'':<10} " if no_action_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_action_stats['mean']:.4f} (n={sam_action_stats['count']})" if sam_action_stats['mean'] is not None else "N/A")
        print()
    
    if no_sampling['tool_entropies'] or samples5['tool_entropies']:
        no_tool_stats = compute_statistics(no_sampling['tool_entropies'])
        sam_tool_stats = compute_statistics(samples5['tool_entropies'])
        print(f"{'Tool Entropy (total)':<30} ", end="")
        print(f"{no_tool_stats['mean']:.4f} (n={no_tool_stats['count']}){'':<10} " if no_tool_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_tool_stats['mean']:.4f} (n={sam_tool_stats['count']})" if sam_tool_stats['mean'] is not None else "N/A")
        print()
    
    print("=" * 80)

def create_plots(no_sampling: Dict, samples5: Dict, output_dir: str):
    """Create visualization plots for the analysis"""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = ['No-Sampling', 'Samples5']
    accuracies = [no_sampling['accuracy'], samples5['accuracy']]
    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(conditions, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_title('Accuracy Comparison: No-Sampling vs Samples5', fontsize=14, fontweight='bold')
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}\n({no_sampling["correct"] if i==0 else samples5["correct"]}/{no_sampling["total"] if i==0 else samples5["total"]})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Entropy Metrics Comparison (Box Plots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    metrics = ['predictive_entropy', 'semantic_entropy', 'sta_predictive', 'sta_semantic']
    metric_labels = ['Predictive Entropy\nH(Y|Z,x)', 'Semantic Entropy\nH_c(Y|Z,x)', 
                     'STA-Predictive\nH(Y|Z,x) + H(Z|a)', 'STA-Semantic\nH_c(Y|Z,x) + H(Z|a)']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        no_vals = no_sampling['entropy_metrics'][metric]
        sam_vals = samples5['entropy_metrics'][metric]
        
        if no_vals and sam_vals:
            data = [no_vals, sam_vals]
            bp = ax.boxplot(data, labels=['No-Sampling', 'Samples5'], patch_artist=True,
                           showmeans=True, meanline=True)
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][1].set_facecolor('#2ecc71')
            for patch in bp['boxes']:
                patch.set_alpha(0.7)
            ax.set_ylabel('Entropy', fontsize=11, fontweight='bold')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=12, fontweight='bold')
    
    plt.suptitle('Entropy Metrics Distribution Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_entropy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Entropy by Correctness
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        no_correct = no_sampling['correct_entropy'][metric]
        no_incorrect = no_sampling['incorrect_entropy'][metric]
        sam_correct = samples5['correct_entropy'][metric]
        sam_incorrect = samples5['incorrect_entropy'][metric]
        
        if (no_correct or no_incorrect or sam_correct or sam_incorrect):
            data = [no_correct, no_incorrect, sam_correct, sam_incorrect]
            labels = ['No-Samp\nCorrect', 'No-Samp\nIncorrect', 'Samples5\nCorrect', 'Samples5\nIncorrect']
            bp = ax.boxplot([d for d in data if d], labels=[l for l, d in zip(labels, data) if d],
                           patch_artist=True, showmeans=True, meanline=True)
            colors_box = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors_box[i % len(colors_box)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Entropy', fontsize=11, fontweight='bold')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=12, fontweight='bold')
    
    plt.suptitle('Entropy Metrics by Correctness', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_entropy_by_correctness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence Metrics (Answer Probability)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    conf_metrics = ['avg_prob', 'min_prob', 'max_prob']
    conf_labels = ['Average Probability', 'Minimum Probability', 'Maximum Probability']
    
    for idx, (metric, label) in enumerate(zip(conf_metrics, conf_labels)):
        ax = axes[idx]
        no_vals = no_sampling['confidence_metrics'][metric]
        sam_vals = samples5['confidence_metrics'][metric]
        
        if no_vals and sam_vals:
            data = [no_vals, sam_vals]
            bp = ax.boxplot(data, labels=['No-Sampling', 'Samples5'], patch_artist=True,
                           showmeans=True, meanline=True)
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][1].set_facecolor('#2ecc71')
            for patch in bp['boxes']:
                patch.set_alpha(0.7)
            ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=12, fontweight='bold')
    
    plt.suptitle('Answer Confidence Metrics (Log Scale)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_confidence_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter: Predictive Entropy vs Correctness
    fig, ax = plt.subplots(figsize=(10, 6))
    no_pred = no_sampling['entropy_metrics']['predictive_entropy']
    sam_pred = samples5['entropy_metrics']['predictive_entropy']
    no_correct = [1 if entry.get('correct') else 0 for entry in 
                  [{'correct': i < no_sampling['correct']} for i in range(len(no_pred))]]
    sam_correct = [1 if entry.get('correct') else 0 for entry in 
                   [{'correct': i < samples5['correct']} for i in range(len(sam_pred))]]
    
    # Reconstruct correctness from the data structure
    no_correct_list = []
    no_incorrect_list = []
    sam_correct_list = []
    sam_incorrect_list = []
    
    # Get correct/incorrect indices
    no_correct_ent = no_sampling['correct_entropy']['predictive_entropy']
    no_incorrect_ent = no_sampling['incorrect_entropy']['predictive_entropy']
    sam_correct_ent = samples5['correct_entropy']['predictive_entropy']
    sam_incorrect_ent = samples5['incorrect_entropy']['predictive_entropy']
    
    ax.scatter(no_correct_ent, [1]*len(no_correct_ent), c='#2ecc71', marker='o', 
              s=100, alpha=0.6, label='No-Sampling: Correct', edgecolors='black', linewidths=1)
    ax.scatter(no_incorrect_ent, [0]*len(no_incorrect_ent), c='#e74c3c', marker='o', 
              s=100, alpha=0.6, label='No-Sampling: Incorrect', edgecolors='black', linewidths=1)
    ax.scatter(sam_correct_ent, [1.1]*len(sam_correct_ent), c='#27ae60', marker='s', 
              s=100, alpha=0.6, label='Samples5: Correct', edgecolors='black', linewidths=1)
    ax.scatter(sam_incorrect_ent, [0.1]*len(sam_incorrect_ent), c='#c0392b', marker='s', 
              s=100, alpha=0.6, label='Samples5: Incorrect', edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Predictive Entropy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correctness', fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Incorrect', 'Correct'])
    ax.set_title('Predictive Entropy vs Correctness', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_entropy_vs_correctness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Step-wise Entropy Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    no_step = no_sampling['step_entropies']
    sam_step = samples5['step_entropies']
    
    # Filter out -0.0 values
    no_step = [s for s in no_step if s > 0 or s != 0.0]
    sam_step = [s for s in sam_step if s > 0 or s != 0.0]
    
    if no_step and sam_step:
        ax.hist(no_step, bins=30, alpha=0.6, label='No-Sampling', color='#3498db', edgecolor='black')
        ax.hist(sam_step, bins=30, alpha=0.6, label='Samples5', color='#2ecc71', edgecolor='black')
        ax.set_xlabel('Step Entropy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Step-wise Entropies', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_step_entropy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Action Entropy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    no_action = [a for a in no_sampling['action_entropies'] if a > 0 or a != 0.0]
    sam_action = [a for a in samples5['action_entropies'] if a > 0 or a != 0.0]
    
    if no_action and sam_action:
        data = [no_action, sam_action]
        bp = ax.boxplot(data, labels=['No-Sampling', 'Samples5'], patch_artist=True,
                       showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#2ecc71')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        ax.set_ylabel('Action Entropy (Selected)', fontsize=12, fontweight='bold')
        ax.set_title('Selected Action Entropy Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_action_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze accuracy and entropy metrics')
    parser.add_argument('--no_sampling_dir', type=str, required=True,
                       help='Directory with no-sampling results')
    parser.add_argument('--samples5_dir', type=str, required=True,
                       help='Directory with samples5 results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detailed results (JSON)')
    parser.add_argument('--plot_dir', type=str, default=None,
                       help='Directory to save plots (if not specified, plots saved to output directory)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    no_sampling_data = load_entropy_data(args.no_sampling_dir)
    samples5_data = load_entropy_data(args.samples5_dir)
    
    print(f"Loaded {len(no_sampling_data)} examples from no-sampling")
    print(f"Loaded {len(samples5_data)} examples from samples5")
    print()
    
    # Analyze
    print("Analyzing...")
    no_sampling_results = analyze_condition(no_sampling_data, "no-sampling")
    samples5_results = analyze_condition(samples5_data, "samples5")
    
    # Print comparison
    print_comparison(no_sampling_results, samples5_results)
    
    # Save detailed results if requested
    if args.output:
        output_data = {
            'no_sampling': no_sampling_results,
            'samples5': samples5_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")
    
    # Create plots
    plot_dir = args.plot_dir
    if plot_dir is None and args.output:
        plot_dir = os.path.dirname(args.output) or '.'
        plot_dir = os.path.join(plot_dir, 'plots')
    elif plot_dir is None:
        plot_dir = 'plots'
    
    print("\nGenerating plots...")
    create_plots(no_sampling_results, samples5_results, plot_dir)

if __name__ == '__main__':
    main()

