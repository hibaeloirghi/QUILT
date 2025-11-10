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
        'action_entropies': [],  # Selected action entropies
        'all_action_entropies': [],  # All action sample entropies (selected + discarded)
        'discarded_action_entropies': [],  # Only discarded action entropies
        'tool_entropies': [],
        'steps_per_question': [],  # Number of steps per question
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
        
        # Action entropies - extract all samples (selected + discarded)
        action_samples = entry.get('action_samples', [])
        
        # Count number of steps for this question (max step number from action_samples)
        if action_samples:
            max_step = max([action.get('step', 0) for action in action_samples])
            results['steps_per_question'].append(max_step)
        else:
            # If no action_samples, try to infer from step_entropies (already loaded above)
            if step_entropies:
                step_nums = [step.get('step', 0) for step in step_entropies if step.get('step') is not None]
                if step_nums:
                    max_step = max(step_nums)
                    results['steps_per_question'].append(max_step)
        
        for action in action_samples:
            selected_entropy = action.get('selected_entropy')
            if selected_entropy is not None and selected_entropy != float('inf'):
                results['action_entropies'].append(selected_entropy)
            
            # Extract all action sample entropies (selected + discarded)
            samples = action.get('samples', [])
            for sample in samples:
                sample_entropy = sample.get('entropy')
                if sample_entropy is not None and sample_entropy != float('inf'):
                    results['all_action_entropies'].append(sample_entropy)
                    # Track discarded samples separately
                    if not sample.get('selected', False):
                        results['discarded_action_entropies'].append(sample_entropy)
        
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
    
    # All action entropies (selected + discarded) - this is what sampling affects
    if no_sampling['all_action_entropies'] or samples5['all_action_entropies']:
        no_all_action_stats = compute_statistics(no_sampling['all_action_entropies'])
        sam_all_action_stats = compute_statistics(samples5['all_action_entropies'])
        print(f"{'Action Entropy (all samples)':<30} ", end="")
        print(f"{no_all_action_stats['mean']:.4f} (n={no_all_action_stats['count']}){'':<10} " if no_all_action_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_all_action_stats['mean']:.4f} (n={sam_all_action_stats['count']})" if sam_all_action_stats['mean'] is not None else "N/A")
        print()
    
    if no_sampling['discarded_action_entropies'] or samples5['discarded_action_entropies']:
        no_discarded_stats = compute_statistics(no_sampling['discarded_action_entropies'])
        sam_discarded_stats = compute_statistics(samples5['discarded_action_entropies'])
        print(f"{'Action Entropy (discarded)':<30} ", end="")
        print(f"{no_discarded_stats['mean']:.4f} (n={no_discarded_stats['count']}){'':<10} " if no_discarded_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_discarded_stats['mean']:.4f} (n={sam_discarded_stats['count']})" if sam_discarded_stats['mean'] is not None else "N/A")
        print()
    
    if no_sampling['tool_entropies'] or samples5['tool_entropies']:
        no_tool_stats = compute_statistics(no_sampling['tool_entropies'])
        sam_tool_stats = compute_statistics(samples5['tool_entropies'])
        print(f"{'Tool Entropy (total)':<30} ", end="")
        print(f"{no_tool_stats['mean']:.4f} (n={no_tool_stats['count']}){'':<10} " if no_tool_stats['mean'] is not None else f"{'N/A':<25} ", end="")
        print(f"{sam_tool_stats['mean']:.4f} (n={sam_tool_stats['count']})" if sam_tool_stats['mean'] is not None else "N/A")
        print()
    
    print("=" * 80)

def print_comparison_threeway(no_sampling: Dict, samples2: Dict = None, samples5: Dict = None):
    """Print detailed three-way comparison between conditions"""
    print("=" * 80)
    title = "ACCURACY ANALYSIS: NO-SAMPLING"
    if samples2:
        title += " vs SAMPLES2"
    if samples5:
        title += " vs SAMPLES5"
    print(title)
    print("=" * 80)
    print()
    
    # Accuracy comparison
    print("ACCURACY METRICS")
    print("-" * 100)
    header = f"{'Metric':<30} {'No-Sampling':<20}"
    if samples2:
        header += f" {'Samples2':<20}"
    if samples5:
        header += f" {'Samples5':<20}"
    print(header)
    print("-" * 100)
    
    def format_val(val, width=20):
        if val is None:
            return f"{'N/A':<{width}}"
        return f"{val:<{width}}"
    
    print(f"{'Total Questions':<30} {format_val(no_sampling['total'])}", end="")
    if samples2:
        print(f" {format_val(samples2['total'])}", end="")
    if samples5:
        print(f" {format_val(samples5['total'])}", end="")
    print()
    
    print(f"{'Correct':<30} {format_val(no_sampling['correct'])}", end="")
    if samples2:
        print(f" {format_val(samples2['correct'])}", end="")
    if samples5:
        print(f" {format_val(samples5['correct'])}", end="")
    print()
    
    print(f"{'Incorrect':<30} {format_val(no_sampling['incorrect'])}", end="")
    if samples2:
        print(f" {format_val(samples2['incorrect'])}", end="")
    if samples5:
        print(f" {format_val(samples5['incorrect'])}", end="")
    print()
    
    acc_no = f"{no_sampling['accuracy']:.4f} ({no_sampling['accuracy']*100:.2f}%)"
    acc_s2 = f"{samples2['accuracy']:.4f} ({samples2['accuracy']*100:.2f}%)" if samples2 else None
    acc_s5 = f"{samples5['accuracy']:.4f} ({samples5['accuracy']*100:.2f}%)" if samples5 else None
    print(f"{'Accuracy':<30} {format_val(acc_no)}", end="")
    if samples2:
        print(f" {format_val(acc_s2)}", end="")
    if samples5:
        print(f" {format_val(acc_s5)}", end="")
    print()
    print()
    
    # Action entropies (key metric)
    print("ACTION ENTROPIES (KEY METRIC FOR SAMPLING)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    
    metrics_to_show = [
        ('Action Entropy (selected)', 'action_entropies'),
        ('Action Entropy (all samples)', 'all_action_entropies'),
        ('Action Entropy (discarded)', 'discarded_action_entropies'),
    ]
    
    for label, key in metrics_to_show:
        no_vals = no_sampling.get(key, [])
        s2_vals = samples2.get(key, []) if samples2 else []
        s5_vals = samples5.get(key, []) if samples5 else []
        
        no_stats = compute_statistics(no_vals) if no_vals else {'mean': None, 'count': 0}
        s2_stats = compute_statistics(s2_vals) if s2_vals else {'mean': None, 'count': 0}
        s5_stats = compute_statistics(s5_vals) if s5_vals else {'mean': None, 'count': 0}
        
        no_str = f"{no_stats['mean']:.4f} (n={no_stats['count']})" if no_stats['mean'] is not None else "N/A"
        s2_str = f"{s2_stats['mean']:.4f} (n={s2_stats['count']})" if s2_stats['mean'] is not None else "N/A"
        s5_str = f"{s5_stats['mean']:.4f} (n={s5_stats['count']})" if s5_stats['mean'] is not None else "N/A"
        
        print(f"{label:<30} {format_val(no_str)}", end="")
        if samples2:
            print(f" {format_val(s2_str)}", end="")
        if samples5:
            print(f" {format_val(s5_str)}", end="")
        print()
    print()
    
    # Average steps per question
    print("AVERAGE STEPS PER QUESTION")
    print("-" * 100)
    print(header)
    print("-" * 100)
    
    no_steps = no_sampling.get('steps_per_question', [])
    s2_steps = samples2.get('steps_per_question', []) if samples2 else []
    s5_steps = samples5.get('steps_per_question', []) if samples5 else []
    
    no_steps_stats = compute_statistics(no_steps) if no_steps else {'mean': None, 'std': None, 'count': 0}
    s2_steps_stats = compute_statistics(s2_steps) if s2_steps else {'mean': None, 'std': None, 'count': 0}
    s5_steps_stats = compute_statistics(s5_steps) if s5_steps else {'mean': None, 'std': None, 'count': 0}
    
    no_steps_str = f"{no_steps_stats['mean']:.2f} ± {no_steps_stats['std']:.2f} (n={no_steps_stats['count']})" if no_steps_stats['mean'] is not None else "N/A"
    s2_steps_str = f"{s2_steps_stats['mean']:.2f} ± {s2_steps_stats['std']:.2f} (n={s2_steps_stats['count']})" if s2_steps_stats['mean'] is not None else "N/A"
    s5_steps_str = f"{s5_steps_stats['mean']:.2f} ± {s5_steps_stats['std']:.2f} (n={s5_steps_stats['count']})" if s5_steps_stats['mean'] is not None else "N/A"
    
    print(f"{'Avg Steps per Question':<30} {format_val(no_steps_str)}", end="")
    if samples2:
        print(f" {format_val(s2_steps_str)}", end="")
    if samples5:
        print(f" {format_val(s5_steps_str)}", end="")
    print()
    print()
    
    print("=" * 80)

def extract_sample_type(directory_path: str) -> str:
    """Extract sample type from directory name (e.g., 'samples2' from '...-samples2/flight-easy')"""
    import re
    # Look for pattern like '-samples2' or '-samples5' in the path
    match = re.search(r'-samples(\d+)', directory_path)
    if match:
        return f"samples{match.group(1)}"
    # Fallback: check if it contains 'samples5' or similar
    if 'samples5' in directory_path.lower():
        return 'samples5'
    elif 'samples' in directory_path.lower():
        # Try to extract any samplesN pattern
        match = re.search(r'samples(\d+)', directory_path, re.IGNORECASE)
        if match:
            return f"samples{match.group(1)}"
    return 'samples5'  # Default fallback

def create_plots(no_sampling: Dict, samples5: Dict, output_dir: str, samples_dir: str = None):
    """Create visualization plots for the analysis
    
    Args:
        no_sampling: Results from no-sampling condition
        samples5: Results from sampling condition
        output_dir: Directory to save plots
        samples_dir: Directory path for sampling condition (to extract sample type)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract sample type from directory name
    sample_type = 'samples5'  # Default
    if samples_dir:
        sample_type = extract_sample_type(samples_dir)
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = ['No-Sampling', sample_type.capitalize()]
    accuracies = [no_sampling['accuracy'], samples5['accuracy']]
    colors = ['#3498db', '#2ecc71']
    bars = ax.bar(conditions, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_title(f'Accuracy Comparison: No-Sampling vs {sample_type.capitalize()}', fontsize=14, fontweight='bold')
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
            bp = ax.boxplot(data, labels=['No-Sampling', sample_type.capitalize()], patch_artist=True,
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
            labels = ['No-Samp\nCorrect', 'No-Samp\nIncorrect', f'{sample_type.capitalize()}\nCorrect', f'{sample_type.capitalize()}\nIncorrect']
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
            bp = ax.boxplot(data, labels=['No-Sampling', sample_type.capitalize()], patch_artist=True,
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
              s=100, alpha=0.6, label=f'{sample_type.capitalize()}: Correct', edgecolors='black', linewidths=1)
    ax.scatter(sam_incorrect_ent, [0.1]*len(sam_incorrect_ent), c='#c0392b', marker='s', 
              s=100, alpha=0.6, label=f'{sample_type.capitalize()}: Incorrect', edgecolors='black', linewidths=1)
    
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
        ax.hist(sam_step, bins=30, alpha=0.6, label=sample_type.capitalize(), color='#2ecc71', edgecolor='black')
        ax.set_xlabel('Step Entropy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Step-wise Entropies', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_step_entropy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Action Entropy Comparison (Selected)
    fig, ax = plt.subplots(figsize=(10, 6))
    no_action = [a for a in no_sampling['action_entropies'] if a > 0 or a != 0.0]
    sam_action = [a for a in samples5['action_entropies'] if a > 0 or a != 0.0]
    
    if no_action and sam_action:
        data = [no_action, sam_action]
        bp = ax.boxplot(data, labels=['No-Sampling', sample_type.capitalize()], patch_artist=True,
                       showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#2ecc71')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        ax.set_ylabel('Action Entropy (Selected)', fontsize=12, fontweight='bold')
        ax.set_title('Selected Action Entropy Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_action_entropy_selected.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. All Action Entropy Comparison (Selected + Discarded) - KEY METRIC FOR SAMPLING
    fig, ax = plt.subplots(figsize=(10, 6))
    no_all_action = [a for a in no_sampling['all_action_entropies'] if a > 0 or a != 0.0]
    sam_all_action = [a for a in samples5['all_action_entropies'] if a > 0 or a != 0.0]
    
    if no_all_action and sam_all_action:
        data = [no_all_action, sam_all_action]
        bp = ax.boxplot(data, labels=['No-Sampling', sample_type.capitalize()], patch_artist=True,
                       showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#2ecc71')
        for patch in bp['boxes']:
            patch.set_alpha(0.7)
        ax.set_ylabel('Action Entropy (All Samples)', fontsize=12, fontweight='bold')
        ax.set_title('All Action Sample Entropy Comparison\n(Selected + Discarded)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_action_entropy_all_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Action Entropy Distribution (Selected vs Discarded) - Normalized for fair comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # No-Sampling
    ax = axes[0]
    no_selected = [a for a in no_sampling['action_entropies'] if a > 0 or a != 0.0]
    no_discarded = [a for a in no_sampling['discarded_action_entropies'] if a > 0 or a != 0.0]
    no_total = len(no_selected) + len(no_discarded)
    if no_selected or no_discarded:
        ax.hist(no_selected, bins=30, alpha=0.6, label=f'Selected (n={len(no_selected)})', 
                color='#2ecc71', edgecolor='black', density=True)
        ax.hist(no_discarded, bins=30, alpha=0.6, label=f'Discarded (n={len(no_discarded)})', 
                color='#e74c3c', edgecolor='black', density=True)
        ax.set_xlabel('Action Entropy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'No-Sampling: Action Entropy Distribution\n(Total: {no_total} actions)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Sampling condition
    ax = axes[1]
    sam_selected = [a for a in samples5['action_entropies'] if a > 0 or a != 0.0]
    sam_discarded = [a for a in samples5['discarded_action_entropies'] if a > 0 or a != 0.0]
    sam_total = len(sam_selected) + len(sam_discarded)
    if sam_selected or sam_discarded:
        ax.hist(sam_selected, bins=30, alpha=0.6, label=f'Selected (n={len(sam_selected)})', 
                color='#2ecc71', edgecolor='black', density=True)
        ax.hist(sam_discarded, bins=30, alpha=0.6, label=f'Discarded (n={len(sam_discarded)})', 
                color='#e74c3c', edgecolor='black', density=True)
        ax.set_xlabel('Action Entropy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{sample_type.capitalize()}: Action Entropy Distribution\n(Total: {sam_total} actions)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Set equal y-axis limits for fair comparison (using density)
    all_data = no_selected + no_discarded + sam_selected + sam_discarded
    if all_data:
        # Compute density histograms to find max density
        max_density = 0
        for data_list in [no_selected, no_discarded, sam_selected, sam_discarded]:
            if len(data_list) > 0:
                counts, _ = np.histogram(data_list, bins=30, density=True)
                if len(counts) > 0:
                    max_density = max(max_density, np.max(counts))
        
        if max_density > 0:
            y_max = max_density * 1.1
            axes[0].set_ylim([0, y_max])
            axes[1].set_ylim([0, y_max])
    
    plt.suptitle('Action Entropy: Selected vs Discarded Samples (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_action_entropy_selected_vs_discarded.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9b. Action Count Comparison (new plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Selected', 'Discarded', 'Total']
    no_counts = [len(no_selected), len(no_discarded), no_total]
    sam_counts = [len(sam_selected), len(sam_discarded), sam_total]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_counts, width, label='No-Sampling', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, sam_counts, width, label=sample_type.capitalize(), color='#2ecc71', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Action Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Actions', fontsize=12, fontweight='bold')
    ax.set_title('Action Count Comparison: No-Sampling vs ' + sample_type.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9b_action_count_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")

def create_plots_threeway(no_sampling: Dict, samples2: Dict = None, samples5: Dict = None, 
                         output_dir: str = 'plots', samples2_dir: str = None, samples5_dir: str = None):
    """Create visualization plots for three-way analysis"""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract sample types
    sample2_type = 'samples2'
    sample5_type = 'samples5'
    if samples2_dir:
        sample2_type = extract_sample_type(samples2_dir)
    if samples5_dir:
        sample5_type = extract_sample_type(samples5_dir)
    
    # 1. Accuracy Comparison (Three-way)
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = ['No-Sampling']
    accuracies = [no_sampling['accuracy']]
    colors = ['#3498db']
    
    if samples2:
        conditions.append(sample2_type.capitalize())
        accuracies.append(samples2['accuracy'])
        colors.append('#2ecc71')
    
    if samples5:
        conditions.append(sample5_type.capitalize())
        accuracies.append(samples5['accuracy'])
        colors.append('#e74c3c')
    
    bars = ax.bar(conditions, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_title('Accuracy Comparison: Three-Way', fontsize=14, fontweight='bold')
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        total = no_sampling['total'] if i == 0 else (samples2['total'] if samples2 and i == 1 else samples5['total'])
        correct = no_sampling['correct'] if i == 0 else (samples2['correct'] if samples2 and i == 1 else samples5['correct'])
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_accuracy_comparison_threeway.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Action Entropy Comparison (All Samples) - KEY METRIC
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    no_all = [a for a in no_sampling.get('all_action_entropies', []) if a > 0 or a != 0.0]
    if no_all:
        data_to_plot.append(no_all)
        labels_to_plot.append('No-Sampling')
        colors_to_plot.append('#3498db')
    
    if samples2:
        s2_all = [a for a in samples2.get('all_action_entropies', []) if a > 0 or a != 0.0]
        if s2_all:
            data_to_plot.append(s2_all)
            labels_to_plot.append(sample2_type.capitalize())
            colors_to_plot.append('#2ecc71')
    
    if samples5:
        s5_all = [a for a in samples5.get('all_action_entropies', []) if a > 0 or a != 0.0]
        if s5_all:
            data_to_plot.append(s5_all)
            labels_to_plot.append(sample5_type.capitalize())
            colors_to_plot.append('#e74c3c')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                       showmeans=True, meanline=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors_to_plot[i])
            patch.set_alpha(0.7)
        ax.set_ylabel('Action Entropy (All Samples)', fontsize=12, fontweight='bold')
        ax.set_title('All Action Sample Entropy Comparison (Three-Way)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_action_entropy_all_samples_threeway.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Action Count Comparison (Three-way)
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = ['Selected', 'Discarded', 'Total']
    x = np.arange(len(categories))
    width = 0.25
    
    no_selected = [a for a in no_sampling.get('action_entropies', []) if a > 0 or a != 0.0]
    no_discarded = [a for a in no_sampling.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
    no_counts = [len(no_selected), len(no_discarded), len(no_selected) + len(no_discarded)]
    
    bars1 = ax.bar(x - width, no_counts, width, label='No-Sampling', color='#3498db', alpha=0.7, edgecolor='black')
    
    offset = 0
    if samples2:
        s2_selected = [a for a in samples2.get('action_entropies', []) if a > 0 or a != 0.0]
        s2_discarded = [a for a in samples2.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
        s2_counts = [len(s2_selected), len(s2_discarded), len(s2_selected) + len(s2_discarded)]
        bars2 = ax.bar(x, s2_counts, width, label=sample2_type.capitalize(), color='#2ecc71', alpha=0.7, edgecolor='black')
        offset = width
    
    if samples5:
        s5_selected = [a for a in samples5.get('action_entropies', []) if a > 0 or a != 0.0]
        s5_discarded = [a for a in samples5.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
        s5_counts = [len(s5_selected), len(s5_discarded), len(s5_selected) + len(s5_discarded)]
        bars3 = ax.bar(x + offset, s5_counts, width, label=sample5_type.capitalize(), color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1] + ([bars2] if samples2 else []) + ([bars3] if samples5 else []):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Action Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Actions', fontsize=12, fontweight='bold')
    ax.set_title('Action Count Comparison (Three-Way)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_action_count_comparison_threeway.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Action Entropy Distribution (Selected vs Discarded) - Three-way Normalized
    num_conditions = 1 + (1 if samples2 else 0) + (1 if samples5 else 0)
    fig, axes = plt.subplots(1, num_conditions, figsize=(6*num_conditions, 6))
    if num_conditions == 1:
        axes = [axes]  # Make it iterable
    
    all_data_lists = []  # Collect all data for y-axis calculation
    
    # No-Sampling
    ax = axes[0]
    no_selected = [a for a in no_sampling.get('action_entropies', []) if a > 0 or a != 0.0]
    no_discarded = [a for a in no_sampling.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
    no_total = len(no_selected) + len(no_discarded)
    all_data_lists.extend([no_selected, no_discarded])
    
    if no_selected or no_discarded:
        ax.hist(no_selected, bins=30, alpha=0.6, label=f'Selected (n={len(no_selected)})', 
                color='#2ecc71', edgecolor='black', density=True)
        ax.hist(no_discarded, bins=30, alpha=0.6, label=f'Discarded (n={len(no_discarded)})', 
                color='#e74c3c', edgecolor='black', density=True)
        ax.set_xlabel('Action Entropy', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'No-Sampling: Action Entropy Distribution\n(Total: {no_total} actions)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Samples2
    ax_idx = 1
    if samples2:
        ax = axes[ax_idx]
        s2_selected = [a for a in samples2.get('action_entropies', []) if a > 0 or a != 0.0]
        s2_discarded = [a for a in samples2.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
        s2_total = len(s2_selected) + len(s2_discarded)
        all_data_lists.extend([s2_selected, s2_discarded])
        
        if s2_selected or s2_discarded:
            ax.hist(s2_selected, bins=30, alpha=0.6, label=f'Selected (n={len(s2_selected)})', 
                    color='#2ecc71', edgecolor='black', density=True)
            ax.hist(s2_discarded, bins=30, alpha=0.6, label=f'Discarded (n={len(s2_discarded)})', 
                    color='#e74c3c', edgecolor='black', density=True)
            ax.set_xlabel('Action Entropy', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{sample2_type.capitalize()}: Action Entropy Distribution\n(Total: {s2_total} actions)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Samples5
    if samples5:
        ax = axes[ax_idx]
        s5_selected = [a for a in samples5.get('action_entropies', []) if a > 0 or a != 0.0]
        s5_discarded = [a for a in samples5.get('discarded_action_entropies', []) if a > 0 or a != 0.0]
        s5_total = len(s5_selected) + len(s5_discarded)
        all_data_lists.extend([s5_selected, s5_discarded])
        
        if s5_selected or s5_discarded:
            ax.hist(s5_selected, bins=30, alpha=0.6, label=f'Selected (n={len(s5_selected)})', 
                    color='#2ecc71', edgecolor='black', density=True)
            ax.hist(s5_discarded, bins=30, alpha=0.6, label=f'Discarded (n={len(s5_discarded)})', 
                    color='#e74c3c', edgecolor='black', density=True)
            ax.set_xlabel('Action Entropy', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f'{sample5_type.capitalize()}: Action Entropy Distribution\n(Total: {s5_total} actions)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # Set equal y-axis limits for fair comparison (using density)
    if all_data_lists and any(len(d) > 0 for d in all_data_lists):
        max_density = 0
        for data_list in all_data_lists:
            if len(data_list) > 0:
                counts, _ = np.histogram(data_list, bins=30, density=True)
                if len(counts) > 0:
                    max_density = max(max_density, np.max(counts))
        
        if max_density > 0:
            y_max = max_density * 1.1
            for ax in axes:
                ax.set_ylim([0, y_max])
    
    plt.suptitle('Action Entropy: Selected vs Discarded Samples (Normalized, Three-Way)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_action_entropy_selected_vs_discarded_threeway.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Average Steps Per Question Comparison (Three-way)
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = []
    avg_steps = []
    std_steps = []
    colors = []
    step_counts = []  # Collect step counts for label calculation
    
    # No-Sampling
    no_steps = no_sampling.get('steps_per_question', [])
    if no_steps:
        conditions.append('No-Sampling')
        avg_steps.append(np.mean(no_steps))
        std_steps.append(np.std(no_steps))
        colors.append('#3498db')
        step_counts.append(no_steps)
    
    # Samples2
    s2_steps = []
    if samples2:
        s2_steps = samples2.get('steps_per_question', [])
        if s2_steps:
            conditions.append(sample2_type.capitalize())
            avg_steps.append(np.mean(s2_steps))
            std_steps.append(np.std(s2_steps))
            colors.append('#2ecc71')
            step_counts.append(s2_steps)
    
    # Samples5
    s5_steps = []
    if samples5:
        s5_steps = samples5.get('steps_per_question', [])
        if s5_steps:
            conditions.append(sample5_type.capitalize())
            avg_steps.append(np.mean(s5_steps))
            std_steps.append(np.std(s5_steps))
            colors.append('#e74c3c')
            step_counts.append(s5_steps)
    
    if conditions:
        
        bars = ax.bar(conditions, avg_steps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5, yerr=std_steps, capsize=5)
        ax.set_ylabel('Average Steps per Question', fontsize=12, fontweight='bold')
        ax.set_title('Average Number of Steps per Question (Three-Way)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, avg, std, step_list) in enumerate(zip(bars, avg_steps, std_steps, step_counts)):
            height = bar.get_height()
            total_questions = len(step_list)
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{avg:.2f} ± {std:.2f}\n(n={total_questions})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_avg_steps_per_question_threeway.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze accuracy and entropy metrics')
    parser.add_argument('--no_sampling_dir', type=str, required=True,
                       help='Directory with no-sampling results')
    parser.add_argument('--samples5_dir', type=str, default=None,
                       help='Directory with samples5 results (optional)')
    parser.add_argument('--samples2_dir', type=str, default=None,
                       help='Directory with samples2 results (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detailed results (JSON)')
    parser.add_argument('--plot_dir', type=str, default=None,
                       help='Directory to save plots (if not specified, plots saved to output directory)')
    
    args = parser.parse_args()
    
    # Validate that at least one sampling directory is provided
    if not args.samples5_dir and not args.samples2_dir:
        parser.error("At least one of --samples5_dir or --samples2_dir must be provided")
    
    # Load data
    print("Loading data...")
    conditions = []
    
    # Load no-sampling (always required)
    no_sampling_data = load_entropy_data(args.no_sampling_dir)
    conditions.append(('no-sampling', args.no_sampling_dir, no_sampling_data))
    print(f"Loaded {len(no_sampling_data)} examples from no-sampling")
    
    # Load samples5 if provided
    samples5_data = None
    samples5_dir = None
    if args.samples5_dir:
        samples5_data = load_entropy_data(args.samples5_dir)
        conditions.append(('samples5', args.samples5_dir, samples5_data))
        print(f"Loaded {len(samples5_data)} examples from samples5")
    
    # Load samples2 if provided
    samples2_data = None
    samples2_dir = None
    if args.samples2_dir:
        samples2_data = load_entropy_data(args.samples2_dir)
        conditions.append(('samples2', args.samples2_dir, samples2_data))
        print(f"Loaded {len(samples2_data)} examples from samples2")
    
    print()
    
    # Analyze all conditions
    print("Analyzing...")
    results = {}
    for name, dir_path, data in conditions:
        results[name] = analyze_condition(data, name)
    
    # Print comparison
    if len(results) == 2:
        # Two-way comparison
        keys = list(results.keys())
        print_comparison(results[keys[0]], results[keys[1]])
    elif len(results) == 3:
        # Three-way comparison
        print_comparison_threeway(results['no-sampling'], 
                                 results.get('samples2'), 
                                 results.get('samples5'))
    
    # Save detailed results if requested
    if args.output:
        output_data = results.copy()
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
    if len(results) == 2:
        # Two-way plots
        keys = list(results.keys())
        samples_dir = args.samples5_dir if 'samples5' in keys else args.samples2_dir
        create_plots(results[keys[0]], results[keys[1]], plot_dir, samples_dir=samples_dir)
    elif len(results) == 3:
        # Three-way plots
        create_plots_threeway(results['no-sampling'], 
                             results.get('samples2'), 
                             results.get('samples5'),
                             plot_dir,
                             samples2_dir=args.samples2_dir,
                             samples5_dir=args.samples5_dir)

if __name__ == '__main__':
    main()

