#!/usr/bin/env python3
"""
Analyze computational cost from Slurm job accounting data and entropy files.
Compares runtime, CPU usage, and memory across different sampling strategies.
"""

import subprocess
import json
import os
import glob
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting will be skipped.")


def parse_time_to_seconds(time_str: str) -> float:
    """Convert Slurm time format (HH:MM:SS, HH:MM:SS.fff, HH:MM.fff, or DD-HH:MM:SS) to seconds.
    Handles fractional values in any time component."""
    if not time_str or time_str == 'Unknown':
        return 0.0
    
    # Handle format like "1-12:34:56" (days-hours:minutes:seconds)
    if '-' in time_str:
        days, time_part = time_str.split('-')
        days = int(days)
    else:
        days = 0
        time_part = time_str
    
    # Parse HH:MM:SS or HH:MM:SS.fff (with fractional seconds)
    # Also handle cases where minutes or hours might have fractional parts
    parts = time_part.split(':')
    if len(parts) == 3:
        hours = float(parts[0])  # Could be fractional
        minutes = float(parts[1])  # Could be fractional (e.g., "45.388")
        # Handle seconds which may have fractional part (e.g., "45.388")
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = float(parts[0])  # Could be fractional
        minutes = float(parts[1])  # Could be fractional (e.g., "45.388")
        seconds = 0.0
    else:
        return 0.0
    
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def parse_memory_to_mb(mem_str: str) -> float:
    """Convert Slurm memory format (e.g., '1024M', '2G') to MB"""
    if not mem_str or mem_str == 'Unknown':
        return 0.0
    
    # Remove whitespace
    mem_str = mem_str.strip()
    
    # Extract number and unit
    match = re.match(r'([\d.]+)([KMGT]?)', mem_str.upper())
    if not match:
        return 0.0
    
    value = float(match.group(1))
    unit = match.group(2)
    
    multipliers = {'': 1, 'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024*1024}
    return value * multipliers.get(unit, 1)


def extract_job_ids_from_out_files(workspace_root: str = None) -> Dict[str, List[str]]:
    """
    Extract job IDs from .out files and match them to conditions based on output directory.
    
    Returns:
        Dict mapping condition names to lists of job IDs
    """
    if workspace_root is None:
        workspace_root = os.getcwd()
    
    results = defaultdict(list)
    
    # Find all llama_toolqa_*.out files
    out_files = glob.glob(os.path.join(workspace_root, "llama_toolqa_*.out"))
    
    for out_file in out_files:
        try:
            with open(out_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < 15:
                    continue
                
                # Extract job ID from first line
                job_id_match = re.search(r'Job ID:\s*(\d+)', lines[0])
                if not job_id_match:
                    continue
                job_id = job_id_match.group(1)
                
                # Extract output directory from the file
                output_dir = None
                for line in lines[:20]:  # Check first 20 lines
                    if 'Output directory:' in line or 'output_dir' in line.lower():
                        # Extract path after "Output directory:" or similar
                        match = re.search(r'Output directory:\s*(.+)', line)
                        if match:
                            output_dir = match.group(1).strip()
                            # Remove "(will skip existing entropy files)" if present
                            output_dir = re.sub(r'\s*\(.*\)\s*$', '', output_dir)
                            break
                
                if not output_dir:
                    continue
                
                # Match to condition based on output directory path
                if 'no-sampling' in output_dir.lower() or 'no_sampling' in output_dir.lower():
                    results['no-sampling'].append(job_id)
                elif 'samples2' in output_dir.lower():
                    results['samples2'].append(job_id)
                elif 'samples5' in output_dir.lower():
                    results['samples5'].append(job_id)
        except Exception as e:
            continue
    
    return results


def query_slurm_jobs_by_id(job_ids: List[str]) -> List[Dict]:
    """
    Query Slurm for specific job IDs.
    
    Args:
        job_ids: List of job IDs to query
    
    Returns:
        List of job records
    """
    if not job_ids:
        return []
    
    results = []
    
    # Build sacct command for specific job IDs - include ReqTRES to get GPU allocation
    cmd = ['sacct', '-j', ','.join(job_ids), 
           '--format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,Start,End,ReqMem,ReqTRES', 
           '--parsable2', '--noheader']
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error querying Slurm: {e.stderr}")
        return results
    
    # Parse output
    for line in output.strip().split('\n'):
        if not line:
            continue
        
        fields = line.split('|')
        if len(fields) < 10:
            continue
        
        job_id = fields[0]
        job_name = fields[1]
        state = fields[2]
        elapsed = fields[3]
        total_cpu = fields[4]
        max_rss = fields[5]
        alloc_cpus = fields[6]
        start = fields[7]
        end = fields[8]
        req_mem = fields[9] if len(fields) > 9 else 'Unknown'
        req_tres = fields[10] if len(fields) > 10 else ''
        
        # Skip batch/array job headers
        if '.batch' in job_id or '.extern' in job_id or ('.' in job_id and job_id.split('.')[-1].isdigit()):
            continue
        
        # Only process llama-toolqa jobs
        if 'llama' not in job_name.lower():
            continue
        
        # Parse GPU allocation from ReqTRES (format: "gres/gpu:rtxa5000=1" or "gres/gpu=1")
        num_gpus = 0
        if req_tres:
            import re
            # Look for gres/gpu=X or gres/gpu:type=X pattern
            gpu_match = re.search(r'gres/gpu(?::\w+)?=(\d+)', req_tres)
            if gpu_match:
                num_gpus = int(gpu_match.group(1))
        
        # Include all states
        elapsed_sec = parse_time_to_seconds(elapsed)
        cpu_sec = parse_time_to_seconds(total_cpu)
        max_rss_mb = parse_memory_to_mb(max_rss)
        req_mem_mb = parse_memory_to_mb(req_mem)
        alloc_cpus_int = int(alloc_cpus) if alloc_cpus.isdigit() else 0
        
        # Calculate GPU-hours: elapsed_time * num_gpus (for GPU jobs)
        gpu_hours = (elapsed_sec * num_gpus) / 3600.0 if num_gpus > 0 else 0.0
        
        # CPU efficiency: For GPU jobs, this is less meaningful, but we'll calculate it
        # Note: CPU time can exceed wall-clock time due to multi-threading, so efficiency can be >100%
        cpu_efficiency = (cpu_sec / (elapsed_sec * alloc_cpus_int)) * 100 if elapsed_sec > 0 and alloc_cpus_int > 0 else 0
        
        results.append({
            'job_id': job_id,
            'job_name': job_name,
            'state': state,
            'elapsed_seconds': elapsed_sec,
            'cpu_seconds': cpu_sec,
            'gpu_hours': gpu_hours,
            'num_gpus': num_gpus,
            'max_rss_mb': max_rss_mb,
            'req_mem_mb': req_mem_mb,
            'alloc_cpus': alloc_cpus_int,
            'start': start,
            'end': end,
            'cpu_efficiency': cpu_efficiency,
            'is_gpu_job': num_gpus > 0
        })
    
    return results


def query_slurm_jobs(condition_dirs: Dict[str, str], start_date: str = None, end_date: str = None) -> Dict[str, List[Dict]]:
    """
    Query Slurm accounting data for jobs matching directory timestamps.
    
    Args:
        condition_dirs: Dict mapping condition names to directory paths
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        Dict mapping condition names to lists of job records
    """
    results = defaultdict(list)
    
    # Extract timestamps from directories
    condition_timestamps = {}
    for condition, dir_path in condition_dirs.items():
        if dir_path:
            timestamp = extract_timestamp_from_dir(dir_path)
            if timestamp:
                condition_timestamps[condition] = timestamp
    
    if not condition_timestamps:
        print("Warning: Could not extract timestamps from directory names")
        return results
    
    # Build sacct command - include ReqTRES to get GPU allocation
    cmd = ['sacct', '--format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,Start,End,ReqMem,ReqTRES', 
           '--parsable2', '--noheader']
    
    if start_date:
        cmd.extend(['--starttime', start_date])
    if end_date:
        cmd.extend(['--endtime', end_date])
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error querying Slurm: {e.stderr}")
        return results
    
    # Parse output
    for line in output.strip().split('\n'):
        if not line:
            continue
        
        fields = line.split('|')
        if len(fields) < 10:
            continue
        
        job_id = fields[0]
        job_name = fields[1]
        state = fields[2]
        elapsed = fields[3]
        total_cpu = fields[4]
        max_rss = fields[5]
        alloc_cpus = fields[6]
        start = fields[7]
        end = fields[8]
        req_mem = fields[9] if len(fields) > 9 else 'Unknown'
        req_tres = fields[10] if len(fields) > 10 else ''
        
        # Skip batch/array job headers (they have .batch or .0 suffix)
        if '.batch' in job_id or ('.' in job_id and job_id.split('.')[-1].isdigit()):
            continue
        
        # Only process llama-toolqa jobs
        if 'llama' not in job_name.lower():
            continue
        
        # Parse GPU allocation from ReqTRES
        num_gpus = 0
        if req_tres:
            import re
            gpu_match = re.search(r'gres/gpu(?::\w+)?=(\d+)', req_tres)
            if gpu_match:
                num_gpus = int(gpu_match.group(1))
        
        # Match job to condition based on start time
        for condition, target_timestamp in condition_timestamps.items():
            # Check if job start time is within 1 hour of target timestamp
            if start and target_timestamp:
                try:
                    from datetime import datetime
                    job_start = datetime.fromisoformat(start.replace('+00:00', '').split('.')[0])
                    target_start = datetime.fromisoformat(target_timestamp)
                    time_diff = abs((job_start - target_start).total_seconds())
                    
                    # Match if within 24 hours (jobs might start on different days)
                    if time_diff < 86400:  # 24 hours in seconds
                        # Only include jobs with meaningful runtime
                        if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY']:
                            elapsed_sec = parse_time_to_seconds(elapsed)
                            cpu_sec = parse_time_to_seconds(total_cpu)
                            max_rss_mb = parse_memory_to_mb(max_rss)
                            req_mem_mb = parse_memory_to_mb(req_mem)
                            alloc_cpus_int = int(alloc_cpus) if alloc_cpus.isdigit() else 0
                            
                            # Calculate GPU-hours
                            gpu_hours = (elapsed_sec * num_gpus) / 3600.0 if num_gpus > 0 else 0.0
                            cpu_efficiency = (cpu_sec / (elapsed_sec * alloc_cpus_int)) * 100 if elapsed_sec > 0 and alloc_cpus_int > 0 else 0
                            
                            results[condition].append({
                                'job_id': job_id,
                                'job_name': job_name,
                                'state': state,
                                'elapsed_seconds': elapsed_sec,
                                'cpu_seconds': cpu_sec,
                                'gpu_hours': gpu_hours,
                                'num_gpus': num_gpus,
                                'max_rss_mb': max_rss_mb,
                                'req_mem_mb': req_mem_mb,
                                'alloc_cpus': alloc_cpus_int,
                                'start': start,
                                'end': end,
                                'cpu_efficiency': cpu_efficiency,
                                'is_gpu_job': num_gpus > 0
                            })
                        break
                except Exception as e:
                    continue
    
    return results


def extract_per_question_data(directory: str, max_questions: int = None, num_gpus: int = 0, 
                              job_elapsed_seconds: float = None) -> Dict[str, Dict]:
    """
    Extract per-question data from entropy JSON files, handling duplicates.
    For duplicate questions, keeps only the latest complete result (by file modification time).
    
    IMPORTANT: A "question" represents the full task (up to 20 steps or less).
    Each step consists of: thought, action, and observation.
    - For no-sampling: 1 action per step (always)
    - For samples2: up to 2 actions per step (max)
    - For samples5: up to 5 actions per step (max)
    
    Args:
        directory: Directory containing entropy JSON files
        max_questions: Maximum number of questions to return
        num_gpus: Number of GPUs used (for calculating GPU-hours)
        job_elapsed_seconds: Total job elapsed time (for estimating runtime if not in files)
    
    Returns:
        Dict mapping question_id to dict with runtime, gpu_hours, steps, etc.
    """
    entropy_files = glob.glob(os.path.join(directory, "*_entropy.json"))
    
    # Group by question ID and keep only latest (by modification time)
    question_data = {}
    for file_path in entropy_files:
        try:
            # Extract question ID from filename (e.g., "easy-flight-0001_entropy.json" -> "easy-flight-0001")
            qid = os.path.basename(file_path).replace("_entropy.json", "")
            
            # Get file modification time to determine latest
            mtime = os.path.getmtime(file_path)
            
            # Load data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a complete result (has correct field and answer)
            if 'correct' in data and 'answer' in data:
                # If we haven't seen this question, or this file is newer, use it
                if qid not in question_data or mtime > question_data[qid]['mtime']:
                    runtime = data.get('runtime_seconds')
                    
                    # Extract computational complexity metrics for better estimation
                    action_samples = data.get('action_samples', [])
                    step_entropies = data.get('step_entropies', [])
                    answer_logprobs = data.get('answer_logprobs', [])
                    
                    # Calculate steps per question
                    # Steps represent the number of reasoning iterations (thought -> action -> observation cycles)
                    if action_samples:
                        max_step = max([action.get('step', 0) for action in action_samples])
                    elif step_entropies:
                        step_nums = [step.get('step', 0) for step in step_entropies if step.get('step') is not None]
                        max_step = max(step_nums) if step_nums else 0
                    else:
                        max_step = 0
                    
                    # Calculate num_action_samples (1 for no-sampling, 2 for samples2, 5 for samples5)
                    # This is the MAXIMUM number of action samples per step (may be less in practice)
                    if action_samples and len(action_samples) > 0:
                        num_action_samples = len(action_samples[0].get('samples', []))
                        if num_action_samples == 0:
                            num_action_samples = 1  # Fallback
                    else:
                        num_action_samples = 1  # Default to no-sampling if no action_samples
                    
                    # Calculate LLM calls: steps * action_samples + answer_samples
                    # Each step requires action_samples LLM calls for action selection
                    # Plus answer_samples LLM calls for final answer generation
                    num_answer_samples = len(answer_logprobs) if answer_logprobs else 0
                    llm_calls = max_step * num_action_samples + num_answer_samples
                    # ACTION calls = steps * action_samples (excluding answer samples)
                    # This counts only the LLM calls used for action selection during reasoning
                    action_calls = max_step * num_action_samples
                    
                    question_data[qid] = {
                        'qid': qid,
                        'runtime_seconds': runtime,  # May be None if estimating
                        'mtime': mtime,
                        'file_path': file_path,
                        'correct': data.get('correct', False),
                        'steps': max_step,
                        'num_action_samples': num_action_samples,
                        'llm_calls': llm_calls,
                        'action_calls': action_calls,  # Steps × action_samples (for action selection)
                        'num_answer_samples': num_answer_samples
                    }
        except Exception as e:
            continue
    
    # If we need to estimate runtime, use computational complexity (LLM calls) instead of simple division
    # Runtime should be proportional to LLM calls, not just number of questions
    # This is important because different questions have different numbers of steps and action samples,
    # so they take different amounts of time. Simply dividing total time by number of questions
    # would give incorrect estimates.
    num_questions_with_runtime = sum(1 for q_data in question_data.values() 
                                     if q_data.get('runtime_seconds') is not None and q_data['runtime_seconds'] > 0)
    
    if num_questions_with_runtime == 0 and job_elapsed_seconds is not None and job_elapsed_seconds > 0:
        # Estimate runtime based on LLM calls per question
        # Total LLM calls across all questions
        total_llm_calls = sum(q_data.get('llm_calls', 0) for q_data in question_data.values())
        
        if total_llm_calls > 0:
            # Time per LLM call = total job time / total LLM calls
            # This assumes that each LLM call takes roughly the same amount of time,
            # which is reasonable for similar model calls (action selection or answer generation)
            time_per_llm_call = job_elapsed_seconds / total_llm_calls
            
            # Estimate runtime for each question: llm_calls * time_per_llm_call
            # This gives a more accurate estimate than simple division because it accounts for
            # the fact that questions with more steps or more action samples take longer
            for qid, q_data in question_data.items():
                if q_data['runtime_seconds'] is None:
                    llm_calls = q_data.get('llm_calls', 0)
                    if llm_calls > 0:
                        q_data['runtime_seconds'] = llm_calls * time_per_llm_call
                    else:
                        # Fallback: if no LLM calls info, use simple average
                        num_questions = len(question_data)
                        q_data['runtime_seconds'] = job_elapsed_seconds / num_questions if num_questions > 0 else 0
        else:
            # Fallback: if no LLM calls data, use simple average
            num_questions = len(question_data)
            if num_questions > 0:
                estimated_runtime = job_elapsed_seconds / num_questions
                for qid, q_data in question_data.items():
                    if q_data['runtime_seconds'] is None:
                        q_data['runtime_seconds'] = estimated_runtime
    
    # Calculate GPU-hours for all questions
    # GPU-hours = (runtime_seconds × num_gpus) / 3600
    # This gives the total GPU time consumed for each question
    for qid, q_data in question_data.items():
        if q_data['runtime_seconds'] is not None and q_data['runtime_seconds'] > 0:
            q_data['gpu_hours'] = (q_data['runtime_seconds'] * num_gpus) / 3600.0 if num_gpus > 0 else 0.0
        else:
            q_data['gpu_hours'] = 0.0
    
    # Sort by question ID and limit if needed
    sorted_questions = sorted(question_data.items(), key=lambda x: x[0])
    if max_questions:
        sorted_questions = sorted_questions[:max_questions]
    
    return {qid: data for qid, data in sorted_questions}


def estimate_runtime_from_slurm_job(job_elapsed_seconds: float, num_questions_processed: int) -> float:
    """Estimate average runtime per question from total job elapsed time"""
    if num_questions_processed > 0:
        return job_elapsed_seconds / num_questions_processed
    return 0.0


def analyze_computational_cost(no_sampling_dir: str = None, samples2_dir: str = None, 
                               samples5_dir: str = None, output_dir: str = 'computational_cost_analysis',
                               max_questions: int = None):
    """
    Analyze computational cost from multiple sources.
    
    KEY CONCEPT: Understanding the relationship between steps, actions, and GPU-hours
    
    A "question" is the full task (up to 20 steps or less). Each step consists of:
    - Thought: reasoning about what to do next
    - Action: selecting and executing an action (may sample multiple candidates)
    - Observation: seeing the result
    
    Action sampling configuration:
    - no-sampling: 1 action per step (always)
    - samples2: up to 2 actions per step (max)
    - samples5: up to 5 actions per step (max)
    
    IMPORTANT: GPU-hours per question = GPU-hours per step × steps per question
    
    If one condition (e.g., no-sampling) has more steps on average, it will have
    higher GPU-hours per question even if GPU-hours per step is the same.
    
    Therefore, we compute BOTH metrics:
    1. GPU-hours per question (shows total cost per task)
    2. GPU-hours per step (normalizes for step count differences)
    
    If GPU-hours per step is similar across conditions, then differences in
    GPU-hours per question are explained by different step counts, not efficiency.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map condition names to directories
    condition_dirs = {}
    if no_sampling_dir:
        condition_dirs['no-sampling'] = no_sampling_dir
    if samples2_dir:
        condition_dirs['samples2'] = samples2_dir
    if samples5_dir:
        condition_dirs['samples5'] = samples5_dir
    
    print("=" * 80)
    print("COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)
    print()
    
    # Extract job IDs from .out files
    print("Extracting job IDs from .out files...")
    # Try to find workspace root - look for .out files in current directory or parent
    workspace_root = os.getcwd()
    # Check if we're in the QUILT directory
    if os.path.basename(workspace_root) != 'QUILT':
        # Try parent directory
        parent = os.path.dirname(workspace_root)
        if os.path.basename(parent) == 'QUILT':
            workspace_root = parent
        else:
            # Try to find QUILT directory by looking for .out files
            test_root = workspace_root
            for _ in range(3):  # Go up max 3 levels
                if glob.glob(os.path.join(test_root, "llama_toolqa_*.out")):
                    workspace_root = test_root
                    break
                test_root = os.path.dirname(test_root)
    
    job_ids_by_condition = extract_job_ids_from_out_files(workspace_root)
    
    print(f"  Found job IDs:")
    for condition, job_ids in job_ids_by_condition.items():
        print(f"    {condition}: {job_ids}")
    
    # Query Slurm for specific job IDs
    print("\nQuerying Slurm for job accounting data...")
    slurm_data = {}
    for condition, job_ids in job_ids_by_condition.items():
        if job_ids:
            jobs = query_slurm_jobs_by_id(job_ids)
            if jobs:
                slurm_data[condition] = jobs
                print(f"  {condition}: Found {len(jobs)} job(s)")
            else:
                print(f"  {condition}: No Slurm data found for job IDs {job_ids}")
        else:
            print(f"  {condition}: No job IDs found in .out files")
    
    # Extract per-question data from entropy files (handling duplicates)
    print("\nExtracting per-question data from entropy files (handling duplicates)...")
    per_question_data = {}
    
    # Define all conditions list
    all_conditions = ['no-sampling', 'samples2', 'samples5']
    
    # Get GPU count and job elapsed time from Slurm jobs for each condition
    condition_gpu_counts = {}
    condition_job_times = {}
    for condition in all_conditions:
        if condition in slurm_data and slurm_data[condition]:
            # Get GPU count from first job (should be same for all jobs in a condition)
            num_gpus = slurm_data[condition][0].get('num_gpus', 0)
            condition_gpu_counts[condition] = num_gpus
            # Sum elapsed time across all jobs for this condition
            total_elapsed = sum([j['elapsed_seconds'] for j in slurm_data[condition]])
            condition_job_times[condition] = total_elapsed
        else:
            condition_gpu_counts[condition] = 0
            condition_job_times[condition] = None
    
    if no_sampling_dir:
        num_gpus = condition_gpu_counts.get('no-sampling', 0)
        job_time = condition_job_times.get('no-sampling')
        per_question_data['no-sampling'] = extract_per_question_data(
            no_sampling_dir, max_questions, num_gpus, job_time)
        print(f"  no-sampling: {len(per_question_data['no-sampling'])} unique questions (after deduplication)")
    if samples2_dir:
        num_gpus = condition_gpu_counts.get('samples2', 0)
        job_time = condition_job_times.get('samples2')
        per_question_data['samples2'] = extract_per_question_data(
            samples2_dir, max_questions, num_gpus, job_time)
        print(f"  samples2: {len(per_question_data['samples2'])} unique questions (after deduplication)")
    if samples5_dir:
        num_gpus = condition_gpu_counts.get('samples5', 0)
        job_time = condition_job_times.get('samples5')
        per_question_data['samples5'] = extract_per_question_data(
            samples5_dir, max_questions, num_gpus, job_time)
        print(f"  samples5: {len(per_question_data['samples5'])} unique questions (after deduplication)")
    
    # Print Slurm job statistics
    print("\n" + "=" * 80)
    print("SLURM JOB STATISTICS")
    print("=" * 80)
    
    # Check if jobs use GPUs and add explanatory note
    has_gpu = False
    for condition in all_conditions:
        if condition in slurm_data:
            jobs = slurm_data[condition]
            if any(j.get('is_gpu_job', False) for j in jobs):
                has_gpu = True
                break
    
    if has_gpu:
        print("\nNOTE: These jobs use GPUs. Key metrics:")
        print("  - GPU-Hours per Question: Primary metric for computational cost")
        print("  - Wall-clock Time per Question: Real elapsed time per question")
        print("  - CPU Time: NOT shown (CPUs are not used for computation in GPU jobs)")
        print("  - All plots show PER-QUESTION metrics (not per-job)")
        print()
    for condition in all_conditions:
        if condition not in slurm_data or len(slurm_data[condition]) == 0:
            print(f"\n{condition.upper()}: No Slurm jobs found")
            continue
        
        jobs = slurm_data[condition]
        completed = [j for j in jobs if j['state'] == 'COMPLETED']
        other_states = [j for j in jobs if j['state'] != 'COMPLETED']
        
        print(f"\n{condition.upper()}:")
        print(f"  Total jobs found: {len(jobs)}")
        print(f"  Completed: {len(completed)}")
        if other_states:
            state_counts = {}
            for j in other_states:
                state_counts[j['state']] = state_counts.get(j['state'], 0) + 1
            for state, count in state_counts.items():
                print(f"  {state}: {count}")
        
        if jobs:
            elapsed_times = [j['elapsed_seconds'] for j in jobs]
            cpu_times = [j['cpu_seconds'] for j in jobs]
            gpu_hours_list = [j['gpu_hours'] for j in jobs if j.get('gpu_hours', 0) > 0]
            num_gpus_list = [j.get('num_gpus', 0) for j in jobs]
            max_rss = [j['max_rss_mb'] for j in jobs if j['max_rss_mb'] > 0]
            cpu_eff = [j['cpu_efficiency'] for j in jobs if j['cpu_efficiency'] > 0]
            is_gpu_job = any(j.get('is_gpu_job', False) for j in jobs)
            
            print(f"\n  Wall-clock Time (Elapsed - Real Time):")
            print(f"    Mean: {np.mean(elapsed_times):.2f} seconds ({np.mean(elapsed_times)/60:.2f} minutes)")
            print(f"    Median: {np.median(elapsed_times):.2f} seconds")
            print(f"    Std: {np.std(elapsed_times):.2f} seconds")
            print(f"    Min: {np.min(elapsed_times):.2f} seconds")
            print(f"    Max: {np.max(elapsed_times):.2f} seconds")
            
            if is_gpu_job and gpu_hours_list:
                print(f"\n  GPU-Hours (Primary Metric for GPU Jobs):")
                print(f"    Mean per job: {np.mean(gpu_hours_list):.2f} GPU-hours")
                print(f"    Total across all jobs: {np.sum(gpu_hours_list):.2f} GPU-hours")
                print(f"    Note: GPU-hours = wall-clock time × number of GPUs")
                if num_gpus_list:
                    print(f"    GPUs per job: {set(num_gpus_list)}")
            
            if max_rss:
                print(f"\n  Memory Usage (Max RSS):")
                print(f"    Mean: {np.mean(max_rss):.2f} MB ({np.mean(max_rss)/1024:.2f} GB)")
                print(f"    Median: {np.median(max_rss):.2f} MB")
                print(f"    Max: {np.max(max_rss):.2f} MB ({np.max(max_rss)/1024:.2f} GB)")
            
            if is_gpu_job:
                print(f"\n  Note: CPU time not shown - CPUs are not used for computation in GPU jobs")
                print(f"        (CPU time only measures overhead like data loading, not actual compute)")
    
    # Print per-question statistics
    print("\n" + "=" * 80)
    print("PER-QUESTION COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)
    
    for condition in all_conditions:
        print(f"\n{condition.upper()}:")
        
        if condition in per_question_data and len(per_question_data[condition]) > 0:
            q_data = per_question_data[condition]
            runtimes = [q['runtime_seconds'] for q in q_data.values()]
            gpu_hours_list = [q['gpu_hours'] for q in q_data.values() if q['gpu_hours'] > 0]
            steps_list = [q.get('steps', 0) for q in q_data.values() if q.get('steps', 0) > 0]
            llm_calls_list = [q.get('llm_calls', 0) for q in q_data.values() if q.get('llm_calls', 0) > 0]
            action_samples_list = [q.get('num_action_samples', 1) for q in q_data.values()]
            
            print(f"  Total unique questions: {len(q_data)}")
            
            if steps_list:
                print(f"  Steps per Question:")
                print(f"    Mean: {np.mean(steps_list):.2f}")
                print(f"    Median: {np.median(steps_list):.2f}")
                print(f"    Std: {np.std(steps_list):.2f}")
            
            if action_samples_list:
                most_common_samples = Counter(action_samples_list).most_common(1)[0][0]
                print(f"  Action Samples per Step: {most_common_samples}")
            
            action_calls_list = [q.get('action_calls', 0) for q in q_data.values() if q.get('action_calls', 0) > 0]
            if action_calls_list:
                print(f"  ACTION Calls per Question (steps × action_samples):")
                print(f"    Mean: {np.mean(action_calls_list):.2f}")
                print(f"    Median: {np.median(action_calls_list):.2f}")
                print(f"    Std: {np.std(action_calls_list):.2f}")
                print(f"    Total: {np.sum(action_calls_list):.0f} ACTION calls across all questions")
            
            if llm_calls_list:
                print(f"  LLM Calls per Question (steps × action_samples + answer_samples):")
                print(f"    Mean: {np.mean(llm_calls_list):.2f}")
                print(f"    Median: {np.median(llm_calls_list):.2f}")
                print(f"    Std: {np.std(llm_calls_list):.2f}")
                print(f"    Total: {np.sum(llm_calls_list):.0f} LLM calls across all questions")
            
            print(f"  Wall-clock Time per Question:")
            print(f"    Mean: {np.mean(runtimes):.2f} seconds ({np.mean(runtimes)/60:.2f} minutes)")
            print(f"    Median: {np.median(runtimes):.2f} seconds ({np.median(runtimes)/60:.2f} minutes)")
            print(f"    Std: {np.std(runtimes):.2f} seconds")
            print(f"    Min: {np.min(runtimes):.2f} seconds ({np.min(runtimes)/60:.2f} minutes)")
            print(f"    Max: {np.max(runtimes):.2f} seconds ({np.max(runtimes)/60:.2f} minutes)")
            
            if has_gpu and gpu_hours_list:
                print(f"  GPU-Hours per Question (Primary Metric):")
                print(f"    Mean: {np.mean(gpu_hours_list):.4f} GPU-hours")
                print(f"    Median: {np.median(gpu_hours_list):.4f} GPU-hours")
                print(f"    Total: {np.sum(gpu_hours_list):.4f} GPU-hours")
                print(f"    Note: GPU-hours = runtime_seconds × num_GPUs / 3600")
                
                # Calculate GPU-hours per step to normalize for different step counts
                # This helps understand if differences in GPU-hours per question are due to
                # different numbers of steps vs. different efficiency per step
                gpu_hours_per_step_list = []
                for qid, q_info in q_data.items():
                    if q_info.get('gpu_hours', 0) > 0 and q_info.get('steps', 0) > 0:
                        gpu_hours_per_step = q_info['gpu_hours'] / q_info['steps']
                        gpu_hours_per_step_list.append(gpu_hours_per_step)
                
                if gpu_hours_per_step_list:
                    print(f"  GPU-Hours per Step (Normalized Metric):")
                    print(f"    Mean: {np.mean(gpu_hours_per_step_list):.6f} GPU-hours")
                    print(f"    Median: {np.median(gpu_hours_per_step_list):.6f} GPU-hours")
                    print(f"    Std: {np.std(gpu_hours_per_step_list):.6f} GPU-hours")
                    print(f"    Note: GPU-hours per step = GPU-hours per question / steps per question")
                    print(f"          If this is similar across conditions, differences in GPU-hours")
                    print(f"          per question are due to different numbers of steps, not efficiency.")
                    
                    # Calculate and print the relationship
                    mean_steps = np.mean(steps_list) if steps_list else 0
                    mean_gpu_hours_per_q = np.mean(gpu_hours_list)
                    mean_gpu_hours_per_step = np.mean(gpu_hours_per_step_list)
                    print(f"  Relationship Analysis:")
                    print(f"    Mean steps per question: {mean_steps:.2f}")
                    print(f"    Mean GPU-hours per question: {mean_gpu_hours_per_q:.6f}")
                    print(f"    Mean GPU-hours per step: {mean_gpu_hours_per_step:.6f}")
                    print(f"    Verification: {mean_steps:.2f} steps × {mean_gpu_hours_per_step:.6f} GPU-h/step = {mean_steps * mean_gpu_hours_per_step:.6f} GPU-h/question")
        else:
            print(f"  No per-question data found (missing runtime_seconds in entropy files)")
    
    # Summary: Compare GPU-hours per step across conditions to determine if differences
    # in GPU-hours per question are due to step counts or efficiency
    if has_gpu:
        print("\n" + "=" * 80)
        print("CROSS-CONDITION COMPARISON: GPU-Hours per Step")
        print("=" * 80)
        print("This analysis determines if differences in GPU-hours per question are")
        print("due to different step counts (if GPU-hours per step is similar) or")
        print("different efficiency per step (if GPU-hours per step differs).")
        print()
        print("NOTE: GPU-hours per step is calculated as: (Total Job GPU-Hours) / (Total Steps)")
        print("      This uses ACTUAL job time from Slurm, not estimated per-question runtime.")
        print("      Estimated per-question runtime is flawed because it divides job time by")
        print("      LLM calls, which makes conditions with more LLM calls appear more efficient.")
        print()
        
        condition_gpu_hours_per_step = {}
        condition_steps = {}
        condition_gpu_hours_per_q = {}
        
        for condition in all_conditions:
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                steps_list = []
                gpu_hours_list = []
                
                # Collect steps and GPU-hours from per-question data
                for q_info in q_data.values():
                    if q_info.get('steps', 0) > 0:
                        steps_list.append(q_info['steps'])
                    if q_info.get('gpu_hours', 0) > 0:
                        gpu_hours_list.append(q_info['gpu_hours'])
                
                # Calculate GPU-hours per step using ACTUAL job time, not estimated per-question time
                # This is more accurate because estimated per-question time is flawed (divides job time by LLM calls)
                if condition in slurm_data and slurm_data[condition] and steps_list:
                    # Get actual job elapsed time and GPU count
                    total_job_elapsed = sum([j['elapsed_seconds'] for j in slurm_data[condition]])
                    num_gpus = slurm_data[condition][0].get('num_gpus', 1)
                    total_job_gpu_hours = (total_job_elapsed * num_gpus) / 3600.0
                    
                    # Total steps across all questions
                    total_steps = sum(steps_list)
                    
                    if total_steps > 0:
                        # GPU-hours per step = total job GPU-hours / total steps
                        # This is the correct way to calculate it, using actual job time
                        condition_gpu_hours_per_step[condition] = total_job_gpu_hours / total_steps
                        condition_steps[condition] = np.mean(steps_list)
                        condition_gpu_hours_per_q[condition] = np.mean(gpu_hours_list) if gpu_hours_list else 0.0
                elif steps_list and gpu_hours_list:
                    # Fallback: use estimated per-question data (less accurate)
                    gpu_hours_per_step_list = []
                    for q_info in q_data.values():
                        if q_info.get('gpu_hours', 0) > 0 and q_info.get('steps', 0) > 0:
                            gpu_hours_per_step_list.append(q_info['gpu_hours'] / q_info['steps'])
                    
                    if gpu_hours_per_step_list:
                        condition_gpu_hours_per_step[condition] = np.mean(gpu_hours_per_step_list)
                        condition_steps[condition] = np.mean(steps_list)
                        condition_gpu_hours_per_q[condition] = np.mean(gpu_hours_list)
        
        if len(condition_gpu_hours_per_step) > 1:
            print("Mean GPU-Hours per Step (Lower is Better):")
            for condition in sorted(condition_gpu_hours_per_step.keys()):
                val = condition_gpu_hours_per_step[condition]
                print(f"  {condition:15s}: {val:.6f} GPU-hours/step")
            
            print("\nMean Steps per Question:")
            for condition in sorted(condition_steps.keys()):
                val = condition_steps[condition]
                print(f"  {condition:15s}: {val:.2f} steps")
            
            print("\nMean GPU-Hours per Question:")
            for condition in sorted(condition_gpu_hours_per_q.keys()):
                val = condition_gpu_hours_per_q[condition]
                print(f"  {condition:15s}: {val:.6f} GPU-hours")
            
            # Check if GPU-hours per step are similar (within 20% of each other)
            values = list(condition_gpu_hours_per_step.values())
            if values:
                min_val = min(values)
                max_val = max(values)
                ratio = max_val / min_val if min_val > 0 else float('inf')
                
                print(f"\nAnalysis:")
                if ratio < 1.2:  # Within 20% of each other
                    print(f"  ✓ GPU-hours per step are similar across conditions (ratio: {ratio:.2f}x)")
                    print(f"  → Differences in GPU-hours per question are primarily due to step counts")
                    print(f"  → No-sampling has higher GPU-hours per question because it takes more steps")
                else:
                    print(f"  ⚠ GPU-hours per step differ significantly across conditions (ratio: {ratio:.2f}x)")
                    print(f"  → Differences in GPU-hours per question are due to BOTH step counts AND efficiency")
    
    # Create visualizations if plotting is available
    if HAS_PLOTTING and per_question_data:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS (PER-QUESTION METRICS)")
        print("=" * 80)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 14
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # 1. Wall-clock Time per Question Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot - per question
        ax = axes[0]
        data_to_plot = []
        labels = []
        for condition in all_conditions:
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                runtimes = [q['runtime_seconds']/60 for q in q_data.values()]  # Convert to minutes
                data_to_plot.append(runtimes)
                labels.append(condition.replace('-', ' ').title())
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Wall-clock Time per Question (minutes)', fontsize=14, fontweight='bold')
            ax.set_title('Wall-clock Time per Question Comparison', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bar plot with means - per question
        ax = axes[1]
        conditions_plot = []
        means = []
        stds = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                runtimes = [q['runtime_seconds']/60 for q in q_data.values()]
                conditions_plot.append(condition.replace('-', ' ').title())
                means.append(np.mean(runtimes))
                stds.append(np.std(runtimes))
                colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds, capsize=8)
            ax.set_ylabel('Mean Wall-clock Time per Question (minutes)', fontsize=14, fontweight='bold')
            ax.set_title('Mean Wall-clock Time per Question', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                       f'{mean:.2f} ± {std:.2f} min',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_wallclock_time_per_question.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. GPU-Hours per Question Comparison (Primary metric for GPU jobs)
        # NOTE: This metric can be misleading if different conditions have different numbers of steps.
        # If no-sampling takes more steps on average, it will have higher GPU-hours per question
        # even if GPU-hours per step is the same. See plot 2b for GPU-hours per step.
        if has_gpu:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Box plot - GPU-hours per question
            ax = axes[0]
            data_to_plot = []
            labels = []
            for condition in all_conditions:
                if condition in per_question_data and len(per_question_data[condition]) > 0:
                    q_data = per_question_data[condition]
                    gpu_hours = [q['gpu_hours'] for q in q_data.values() if q['gpu_hours'] > 0]
                    if gpu_hours:
                        data_to_plot.append(gpu_hours)
                        labels.append(condition.replace('-', ' ').title())
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i % len(colors)])
                    patch.set_alpha(0.7)
                ax.set_ylabel('GPU-Hours per Question', fontsize=14, fontweight='bold')
                ax.set_title('GPU-Hours per Question\n(May vary if step counts differ)', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            # Bar plot with means - GPU-hours per question
            ax = axes[1]
            conditions_plot = []
            means = []
            stds = []
            colors_bar = []
            for i, condition in enumerate(all_conditions):
                if condition in per_question_data and len(per_question_data[condition]) > 0:
                    q_data = per_question_data[condition]
                    gpu_hours = [q['gpu_hours'] for q in q_data.values() if q['gpu_hours'] > 0]
                    if gpu_hours:
                        conditions_plot.append(condition.replace('-', ' ').title())
                        means.append(np.mean(gpu_hours))
                        stds.append(np.std(gpu_hours))
                        colors_bar.append(colors[i % len(colors)])
            
            if conditions_plot:
                bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                             edgecolor='black', linewidth=2, yerr=stds, capsize=8)
                ax.set_ylabel('Mean GPU-Hours per Question', fontsize=14, fontweight='bold')
                ax.set_title('Mean GPU-Hours per Question', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.01,
                           f'{mean:.4f} ± {std:.4f}',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '2_gpu_hours_per_question.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2b. GPU-Hours per Step Comparison (Normalized metric)
            # This normalizes GPU-hours by the number of steps, which helps understand
            # if differences in GPU-hours per question are due to:
            # - Different numbers of steps (if GPU-hours per step is similar)
            # - Different efficiency per step (if GPU-hours per step differs)
            # If all conditions have similar GPU-hours per step, then the differences
            # in GPU-hours per question are explained by different step counts.
            # IMPORTANT: Uses ACTUAL job time / total steps, not estimated per-question time
            # (estimated per-question time is flawed because it divides job time by LLM calls)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar plot with means - GPU-hours per step (calculated from actual job time)
            conditions_plot = []
            means = []
            colors_bar = []
            for i, condition in enumerate(all_conditions):
                if condition in per_question_data and len(per_question_data[condition]) > 0:
                    q_data = per_question_data[condition]
                    
                    # Calculate GPU-hours per step using ACTUAL job time, not estimated per-question time
                    if condition in slurm_data and slurm_data[condition]:
                        # Get actual job elapsed time and GPU count
                        total_job_elapsed = sum([j['elapsed_seconds'] for j in slurm_data[condition]])
                        num_gpus = slurm_data[condition][0].get('num_gpus', 1)
                        total_job_gpu_hours = (total_job_elapsed * num_gpus) / 3600.0
                        
                        # Total steps across all questions
                        total_steps = sum([q.get('steps', 0) for q in q_data.values() if q.get('steps', 0) > 0])
                        
                        if total_steps > 0:
                            # GPU-hours per step = total job GPU-hours / total steps
                            gpu_hours_per_step = total_job_gpu_hours / total_steps
                            conditions_plot.append(condition.replace('-', ' ').title())
                            means.append(gpu_hours_per_step)
                            colors_bar.append(colors[i % len(colors)])
            
            if conditions_plot:
                bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                             edgecolor='black', linewidth=2)
                ax.set_ylabel('GPU-Hours per Step', fontsize=14, fontweight='bold')
                ax.set_title('GPU-Hours per Step (from Actual Job Time)\n(If similar, differences in per-question metric are due to step counts)', 
                            fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.01,
                           f'{mean:.6f}',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '2b_gpu_hours_per_step.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Memory Usage Comparison - per job (memory is a job-level metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = []
        labels = []
        for condition in all_conditions:
            if condition in slurm_data:
                jobs = [j for j in slurm_data[condition] if j['max_rss_mb'] > 0]
                if jobs:
                    mem = [j['max_rss_mb']/1024 for j in jobs]  # Convert to GB
                    data_to_plot.append(mem)
                    labels.append(condition.replace('-', ' ').title())
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Peak Memory Usage (GB)', fontsize=14, fontweight='bold')
            ax.set_title('Peak Memory Usage per Job', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_memory_usage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Steps per Question Comparison
        # Steps represent the number of reasoning iterations (thought -> action -> observation cycles)
        # More steps = more reasoning iterations needed to solve the question
        # This is important because GPU-hours per question = GPU-hours per step × steps per question
        # If one condition has more steps on average, it will have higher GPU-hours per question
        # even if GPU-hours per step is the same (see plot 2b for normalization)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot - steps per question
        ax = axes[0]
        data_to_plot = []
        labels = []
        for condition in all_conditions:
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                steps = [q.get('steps', 0) for q in q_data.values() if q.get('steps', 0) > 0]
                if steps:
                    data_to_plot.append(steps)
                    labels.append(condition.replace('-', ' ').title())
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Steps per Question', fontsize=14, fontweight='bold')
            ax.set_title('Steps per Question Comparison', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bar plot with means - steps per question
        ax = axes[1]
        conditions_plot = []
        means = []
        stds = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                steps = [q.get('steps', 0) for q in q_data.values() if q.get('steps', 0) > 0]
                if steps:
                    conditions_plot.append(condition.replace('-', ' ').title())
                    means.append(np.mean(steps))
                    stds.append(np.std(steps))
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds, capsize=8)
            ax.set_ylabel('Mean Steps per Question', fontsize=14, fontweight='bold')
            ax.set_title('Mean Steps per Question', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                       f'{mean:.2f} ± {std:.2f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_steps_per_question.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. LLM Calls per Question Comparison (steps × action_samples + answer_samples)
        # LLM calls = (steps × action_samples) + answer_samples
        # - Each step requires 'action_samples' LLM calls to generate action candidates
        # - Plus 'answer_samples' LLM calls for final answer generation
        # This metric correlates with runtime since each LLM call takes time
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with means - LLM calls per question
        conditions_plot = []
        means = []
        stds = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                llm_calls = [q.get('llm_calls', 0) for q in q_data.values() if q.get('llm_calls', 0) > 0]
                if llm_calls:
                    conditions_plot.append(condition.replace('-', ' ').title())
                    means.append(np.mean(llm_calls))
                    stds.append(np.std(llm_calls))
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds, capsize=8)
            ax.set_ylabel('Mean LLM Calls per Question', fontsize=14, fontweight='bold')
            ax.set_title('Mean LLM Calls per Question', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.02,
                       f'{mean:.1f} ± {std:.1f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_llm_calls_per_question.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5b. ACTION Calls per Question Comparison (steps × action_samples, excluding answer samples)
        # ACTION calls = steps × action_samples (excluding answer generation)
        # This counts only the LLM calls used for action selection during reasoning steps
        # - no-sampling: 1 action call per step
        # - samples2: 2 action calls per step (max)
        # - samples5: 5 action calls per step (max)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with means - ACTION calls per question
        conditions_plot = []
        means = []
        stds = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                action_calls = [q.get('action_calls', 0) for q in q_data.values() if q.get('action_calls', 0) > 0]
                if action_calls:
                    conditions_plot.append(condition.replace('-', ' ').title())
                    means.append(np.mean(action_calls))
                    stds.append(np.std(action_calls))
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds, capsize=8)
            ax.set_ylabel('Mean ACTION Calls per Question', fontsize=14, fontweight='bold')
            ax.set_title('Mean ACTION Calls per Question', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.02,
                       f'{mean:.1f} ± {std:.1f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5b_action_calls_per_question.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Scatter: Steps vs Runtime
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                steps = [q.get('steps', 0) for q in q_data.values()]
                runtimes = [q.get('runtime_seconds', 0)/60 for q in q_data.values()]  # Convert to minutes
                
                # Filter out invalid data
                valid_pairs = [(s, r) for s, r in zip(steps, runtimes) if s > 0 and r > 0]
                if valid_pairs:
                    steps_vals, runtimes_vals = zip(*valid_pairs)
                    ax.scatter(steps_vals, runtimes_vals, alpha=0.6, s=100, 
                             label=condition.replace('-', ' ').title(), 
                             color=colors[i % len(colors)], edgecolors='black', linewidths=1)
        
        ax.set_xlabel('Steps per Question', fontsize=14, fontweight='bold')
        ax.set_ylabel('Runtime per Question (minutes)', fontsize=14, fontweight='bold')
        ax.set_title('Steps vs Runtime per Question', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_steps_vs_runtime.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Scatter: LLM Calls vs Runtime
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                llm_calls = [q.get('llm_calls', 0) for q in q_data.values()]
                runtimes = [q.get('runtime_seconds', 0)/60 for q in q_data.values()]  # Convert to minutes
                
                # Filter out invalid data
                valid_pairs = [(l, r) for l, r in zip(llm_calls, runtimes) if l > 0 and r > 0]
                if valid_pairs:
                    llm_vals, runtimes_vals = zip(*valid_pairs)
                    ax.scatter(llm_vals, runtimes_vals, alpha=0.6, s=100, 
                             label=condition.replace('-', ' ').title(), 
                             color=colors[i % len(colors)], edgecolors='black', linewidths=1)
        
        ax.set_xlabel('LLM Calls per Question', fontsize=14, fontweight='bold')
        ax.set_ylabel('Runtime per Question (minutes)', fontsize=14, fontweight='bold')
        ax.set_title('LLM Calls vs Runtime per Question', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7_llm_calls_vs_runtime.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Action Samples per Step (should be 1, 2, or 5)
        # This shows the configuration: how many action samples are generated per step
        # - no-sampling: 1 action per step (always)
        # - samples2: up to 2 actions per step (max)
        # - samples5: up to 5 actions per step (max)
        # Note: This is the MAXIMUM; actual number may be less if fewer candidates are needed
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions_plot = []
        action_samples_vals = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                # Get unique action sample counts (should be consistent within condition)
                action_samples_set = set([q.get('num_action_samples', 1) for q in q_data.values()])
                if action_samples_set:
                    # Use the most common value
                    action_samples_list = [q.get('num_action_samples', 1) for q in q_data.values()]
                    most_common = Counter(action_samples_list).most_common(1)[0][0]
                    conditions_plot.append(condition.replace('-', ' ').title())
                    action_samples_vals.append(most_common)
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, action_samples_vals, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2)
            ax.set_ylabel('Action Samples per Step', fontsize=14, fontweight='bold')
            ax.set_title('Action Samples per Step (Configuration)', fontsize=16, fontweight='bold')
            ax.set_ylim([0, max(action_samples_vals) * 1.2])
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, action_samples_vals):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(action_samples_vals)*0.02,
                       f'{int(val)}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_action_samples_per_step.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. Efficiency Analysis: GPU Hours per ACTION Call and per Step
        # NOTE: This uses actual job elapsed time divided by ACTION calls/steps, not estimated per-question runtime
        # This gives a more accurate picture of efficiency since it accounts for batching/parallelization
        # across questions within a job. The per-question estimates may not capture this accurately.
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate efficiency metrics using actual job time (not estimated per-question time)
        conditions_plot = []
        efficiency_vals = []  # GPU hours per ACTION call (from job time)
        efficiency_per_step = []  # GPU hours per step (from job time)
        colors_bar = []
        
        for i, condition in enumerate(all_conditions):
            if condition in per_question_data and len(per_question_data[condition]) > 0:
                q_data = per_question_data[condition]
                action_calls = [q.get('action_calls', 0) for q in q_data.values() if q.get('action_calls', 0) > 0]
                steps = [q.get('steps', 0) for q in q_data.values() if q.get('steps', 0) > 0]
                
                # Get actual job elapsed time from Slurm (more accurate than estimated per-question time)
                if condition in slurm_data and slurm_data[condition]:
                    total_job_elapsed = sum([j['elapsed_seconds'] for j in slurm_data[condition]])
                    num_gpus = slurm_data[condition][0].get('num_gpus', 1)
                    total_job_gpu_hours = (total_job_elapsed * num_gpus) / 3600.0
                else:
                    # Fallback: use estimated per-question time
                    gpu_hours = [q.get('gpu_hours', 0) for q in q_data.values() if q.get('gpu_hours', 0) > 0]
                    if gpu_hours:
                        total_job_gpu_hours = sum(gpu_hours)
                    else:
                        continue
                
                if action_calls and steps:
                    total_action_calls = sum(action_calls)
                    total_steps = sum(steps)
                    
                    if total_action_calls > 0 and total_steps > 0:
                        conditions_plot.append(condition.replace('-', ' ').title())
                        # Use actual job GPU-hours divided by ACTION calls (more accurate)
                        efficiency_vals.append(total_job_gpu_hours / total_action_calls)
                        efficiency_per_step.append(total_job_gpu_hours / total_steps)
                        colors_bar.append(colors[i % len(colors)])
        
        # Plot 1: GPU Hours per ACTION Call
        ax = axes[0]
        if conditions_plot:
            bars = ax.bar(conditions_plot, efficiency_vals, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2)
            ax.set_ylabel('GPU Hours per ACTION Call', fontsize=14, fontweight='bold')
            ax.set_title('Efficiency: GPU Hours per ACTION Call\n(Lower is Better)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Convert to seconds for labels
            for bar, eff in zip(bars, efficiency_vals):
                height = bar.get_height()
                seconds = eff * 3600
                ax.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_vals)*0.02,
                       f'{eff:.6f}\n({seconds:.1f}s)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 2: GPU Hours per Step
        ax = axes[1]
        if conditions_plot:
            bars = ax.bar(conditions_plot, efficiency_per_step, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2)
            ax.set_ylabel('GPU Hours per Step', fontsize=14, fontweight='bold')
            ax.set_title('Efficiency: GPU Hours per Step\n(Lower is Better)', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Convert to seconds for labels
            for bar, eff in zip(bars, efficiency_per_step):
                height = bar.get_height()
                seconds = eff * 3600
                ax.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_per_step)*0.02,
                       f'{eff:.6f}\n({seconds:.1f}s)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '9_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}/")
    
    # Save detailed data to JSON
    output_data = {
        'slurm_jobs': {k: v for k, v in slurm_data.items()},
        'per_question_data': {k: {qid: {**q, 'mtime': None, 'file_path': None} for qid, q in v.items()} 
                              for k, v in per_question_data.items()}  # Remove file paths for JSON
    }
    
    output_json = os.path.join(output_dir, 'computational_cost_data.json')
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed data saved to: {output_json}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze computational cost from Slurm and entropy files')
    parser.add_argument('--no_sampling_dir', type=str, default=None,
                       help='Directory with no-sampling entropy files')
    parser.add_argument('--samples2_dir', type=str, default=None,
                       help='Directory with samples2 entropy files')
    parser.add_argument('--samples5_dir', type=str, default=None,
                       help='Directory with samples5 entropy files')
    parser.add_argument('--output_dir', type=str, default='computational_cost_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--max_questions', type=int, default=None,
                       help='Maximum number of questions to analyze')
    
    args = parser.parse_args()
    
    analyze_computational_cost(
        no_sampling_dir=args.no_sampling_dir,
        samples2_dir=args.samples2_dir,
        samples5_dir=args.samples5_dir,
        output_dir=args.output_dir,
        max_questions=args.max_questions
    )


if __name__ == '__main__':
    main()

