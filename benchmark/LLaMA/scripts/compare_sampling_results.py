#!/usr/bin/env python3
"""
Compare accuracy between runs with and without action sampling.

Usage:
    python compare_sampling_results.py <dir_with_sampling> <dir_without_sampling>
    
Example:
    python compare_sampling_results.py \
        benchmark/LLaMA/logs/llama-3.1-8b-2025-11-08\ 20:04:59-samples5/flight-easy \
        benchmark/LLaMA/logs/llama-3.1-8b-2025-11-08\ 20:05:30-no-sampling/flight-easy
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def load_results(directory):
    """Load all entropy JSON files from a directory and extract accuracy info."""
    results = {}
    correct_count = 0
    total_count = 0
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return None
    
    # Find all _entropy.json files
    json_files = list(dir_path.glob("*_entropy.json"))
    
    if len(json_files) == 0:
        print(f"Warning: No entropy JSON files found in {directory}")
        return None
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                qid = data.get('qid', json_file.stem.replace('_entropy', ''))
                correct = data.get('correct', False)
                question = data.get('question', '')
                answer = data.get('answer', '')
                ground_truth = data.get('ground_truth', '')
                
                results[qid] = {
                    'correct': correct,
                    'question': question,
                    'answer': answer,
                    'ground_truth': ground_truth
                }
                
                total_count += 1
                if correct:
                    correct_count += 1
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return {
        'results': results,
        'accuracy': correct_count / total_count if total_count > 0 else 0.0,
        'correct': correct_count,
        'total': total_count
    }

def compare_results(dir_with_sampling, dir_without_sampling):
    """Compare results from two runs."""
    print("="*80)
    print("COMPARING ACTION SAMPLING vs NO SAMPLING")
    print("="*80)
    print()
    
    # Load results
    print(f"Loading results from WITH sampling: {dir_with_sampling}")
    with_sampling = load_results(dir_with_sampling)
    if with_sampling is None:
        return
    
    print(f"Loading results from WITHOUT sampling: {dir_without_sampling}")
    without_sampling = load_results(dir_without_sampling)
    if without_sampling is None:
        return
    
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Overall accuracy
    print(f"WITH sampling (5 samples):")
    print(f"  Accuracy: {with_sampling['accuracy']:.2%} ({with_sampling['correct']}/{with_sampling['total']})")
    print()
    print(f"WITHOUT sampling (1 sample):")
    print(f"  Accuracy: {without_sampling['accuracy']:.2%} ({without_sampling['correct']}/{without_sampling['total']})")
    print()
    
    # Difference
    diff = with_sampling['accuracy'] - without_sampling['accuracy']
    print(f"Difference: {diff:+.2%} ({'+' if diff > 0 else ''}{diff*100:.2f} percentage points)")
    if diff > 0:
        print("  → Sampling improves accuracy!")
    elif diff < 0:
        print("  → No sampling performs better")
    else:
        print("  → No difference")
    print()
    
    # Per-question comparison
    print("="*80)
    print("PER-QUESTION COMPARISON")
    print("="*80)
    print()
    
    # Find common questions
    common_qids = set(with_sampling['results'].keys()) & set(without_sampling['results'].keys())
    
    if len(common_qids) == 0:
        print("Warning: No common questions found between the two runs")
        return
    
    # Categorize questions
    both_correct = []
    both_incorrect = []
    sampling_helps = []  # Correct with sampling, incorrect without
    sampling_hurts = []   # Incorrect with sampling, correct without
    
    for qid in common_qids:
        with_correct = with_sampling['results'][qid]['correct']
        without_correct = without_sampling['results'][qid]['correct']
        
        if with_correct and without_correct:
            both_correct.append(qid)
        elif not with_correct and not without_correct:
            both_incorrect.append(qid)
        elif with_correct and not without_correct:
            sampling_helps.append(qid)
        else:  # not with_correct and without_correct
            sampling_hurts.append(qid)
    
    print(f"Total common questions: {len(common_qids)}")
    print(f"  Both correct: {len(both_correct)}")
    print(f"  Both incorrect: {len(both_incorrect)}")
    print(f"  Sampling helps (correct with, incorrect without): {len(sampling_helps)}")
    print(f"  Sampling hurts (incorrect with, correct without): {len(sampling_hurts)}")
    print()
    
    # Show examples where sampling helps
    if sampling_helps:
        print("="*80)
        print(f"QUESTIONS WHERE SAMPLING HELPS ({len(sampling_helps)} questions)")
        print("="*80)
        for qid in sampling_helps[:5]:  # Show first 5
            with_data = with_sampling['results'][qid]
            without_data = without_sampling['results'][qid]
            print(f"\nQID: {qid}")
            print(f"Question: {with_data['question']}")
            print(f"Ground truth: {with_data['ground_truth']}")
            print(f"  WITH sampling: {with_data['answer']} ✓")
            print(f"  WITHOUT sampling: {without_data['answer']} ✗")
        if len(sampling_helps) > 5:
            print(f"\n... and {len(sampling_helps) - 5} more")
        print()
    
    # Show examples where sampling hurts
    if sampling_hurts:
        print("="*80)
        print(f"QUESTIONS WHERE SAMPLING HURTS ({len(sampling_hurts)} questions)")
        print("="*80)
        for qid in sampling_hurts[:5]:  # Show first 5
            with_data = with_sampling['results'][qid]
            without_data = without_sampling['results'][qid]
            print(f"\nQID: {qid}")
            print(f"Question: {with_data['question']}")
            print(f"Ground truth: {with_data['ground_truth']}")
            print(f"  WITH sampling: {with_data['answer']} ✗")
            print(f"  WITHOUT sampling: {without_data['answer']} ✓")
        if len(sampling_hurts) > 5:
            print(f"\n... and {len(sampling_hurts) - 5} more")
        print()
    
    print("="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    dir_with_sampling = sys.argv[1]
    dir_without_sampling = sys.argv[2]
    
    compare_results(dir_with_sampling, dir_without_sampling)

