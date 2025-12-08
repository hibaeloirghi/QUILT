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
from collections import defaultdict
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
    
    # Build sacct command for specific job IDs
    cmd = ['sacct', '-j', ','.join(job_ids), 
           '--format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,Start,End,ReqMem', 
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
        if len(fields) < 9:
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
        
        # Skip batch/array job headers
        if '.batch' in job_id or '.extern' in job_id or ('.' in job_id and job_id.split('.')[-1].isdigit()):
            continue
        
        # Only process llama-toolqa jobs
        if 'llama' not in job_name.lower():
            continue
        
        # Include all states
        elapsed_sec = parse_time_to_seconds(elapsed)
        cpu_sec = parse_time_to_seconds(total_cpu)
        max_rss_mb = parse_memory_to_mb(max_rss)
        req_mem_mb = parse_memory_to_mb(req_mem)
        alloc_cpus_int = int(alloc_cpus) if alloc_cpus.isdigit() else 0
        
        results.append({
            'job_id': job_id,
            'job_name': job_name,
            'state': state,
            'elapsed_seconds': elapsed_sec,
            'cpu_seconds': cpu_sec,
            'max_rss_mb': max_rss_mb,
            'req_mem_mb': req_mem_mb,
            'alloc_cpus': alloc_cpus_int,
            'start': start,
            'end': end,
            'cpu_efficiency': (cpu_sec / (elapsed_sec * alloc_cpus_int)) * 100 if elapsed_sec > 0 and alloc_cpus_int > 0 else 0
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
    
    # Build sacct command
    cmd = ['sacct', '--format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,Start,End,ReqMem', 
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
        if len(fields) < 9:
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
        
        # Skip batch/array job headers (they have .batch or .0 suffix)
        if '.batch' in job_id or ('.' in job_id and job_id.split('.')[-1].isdigit()):
            continue
        
        # Only process llama-toolqa jobs
        if 'llama' not in job_name.lower():
            continue
        
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
                            
                            results[condition].append({
                                'job_id': job_id,
                                'job_name': job_name,
                                'state': state,
                                'elapsed_seconds': elapsed_sec,
                                'cpu_seconds': cpu_sec,
                                'max_rss_mb': max_rss_mb,
                                'req_mem_mb': req_mem_mb,
                                'alloc_cpus': alloc_cpus_int,
                                'start': start,
                                'end': end,
                                'cpu_efficiency': (cpu_sec / (elapsed_sec * alloc_cpus_int)) * 100 if elapsed_sec > 0 and alloc_cpus_int > 0 else 0
                            })
                        break
                except Exception as e:
                    continue
    
    return results


def extract_runtime_from_entropy_files(directory: str, max_questions: int = None) -> List[float]:
    """Extract runtime_seconds from entropy JSON files if available"""
    runtimes = []
    entropy_files = sorted(glob.glob(os.path.join(directory, "*_entropy.json")))
    
    if max_questions:
        entropy_files = entropy_files[:max_questions]
    
    # Try to get runtime from JSON
    for file_path in entropy_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                runtime = data.get('runtime_seconds')
                if runtime is not None and runtime > 0:
                    runtimes.append(runtime)
        except Exception as e:
            continue
    
    return runtimes


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
    
    # Extract runtime from entropy files (if available)
    runtime_data = {}
    if no_sampling_dir:
        print(f"\nExtracting runtime from entropy files: {no_sampling_dir}")
        runtime_data['no-sampling'] = extract_runtime_from_entropy_files(no_sampling_dir, max_questions)
    if samples2_dir:
        print(f"Extracting runtime from entropy files: {samples2_dir}")
        runtime_data['samples2'] = extract_runtime_from_entropy_files(samples2_dir, max_questions)
    if samples5_dir:
        print(f"Extracting runtime from entropy files: {samples5_dir}")
        runtime_data['samples5'] = extract_runtime_from_entropy_files(samples5_dir, max_questions)
    
    # Print Slurm job statistics
    print("\n" + "=" * 80)
    print("SLURM JOB STATISTICS")
    print("=" * 80)
    
    all_conditions = ['no-sampling', 'samples2', 'samples5']
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
            max_rss = [j['max_rss_mb'] for j in jobs if j['max_rss_mb'] > 0]
            cpu_eff = [j['cpu_efficiency'] for j in jobs if j['cpu_efficiency'] > 0]
            
            print(f"\n  Wall-clock Time (Elapsed):")
            print(f"    Mean: {np.mean(elapsed_times):.2f} seconds ({np.mean(elapsed_times)/60:.2f} minutes)")
            print(f"    Median: {np.median(elapsed_times):.2f} seconds")
            print(f"    Std: {np.std(elapsed_times):.2f} seconds")
            print(f"    Min: {np.min(elapsed_times):.2f} seconds")
            print(f"    Max: {np.max(elapsed_times):.2f} seconds")
            
            print(f"\n  Total CPU Time:")
            print(f"    Mean: {np.mean(cpu_times):.2f} seconds ({np.mean(cpu_times)/60:.2f} minutes)")
            print(f"    Median: {np.median(cpu_times):.2f} seconds")
            print(f"    Total: {np.sum(cpu_times):.2f} seconds ({np.sum(cpu_times)/3600:.2f} CPU-hours)")
            
            if max_rss:
                print(f"\n  Memory Usage (Max RSS):")
                print(f"    Mean: {np.mean(max_rss):.2f} MB ({np.mean(max_rss)/1024:.2f} GB)")
                print(f"    Median: {np.median(max_rss):.2f} MB")
                print(f"    Max: {np.max(max_rss):.2f} MB ({np.max(max_rss)/1024:.2f} GB)")
            
            if cpu_eff:
                print(f"\n  CPU Efficiency:")
                print(f"    Mean: {np.mean(cpu_eff):.1f}%")
                print(f"    Median: {np.median(cpu_eff):.1f}%")
    
    # Print runtime from entropy files (if available)
    print("\n" + "=" * 80)
    print("RUNTIME ANALYSIS")
    print("=" * 80)
    
    for condition in all_conditions:
        print(f"\n{condition.upper()}:")
        
        # Runtime from entropy files
        if condition in runtime_data and len(runtime_data[condition]) > 0:
            runtimes = runtime_data[condition]
            print(f"  Runtime from entropy JSON files:")
            print(f"    Questions with runtime data: {len(runtimes)}")
            print(f"    Mean runtime: {np.mean(runtimes):.2f} seconds ({np.mean(runtimes)/60:.2f} minutes)")
            print(f"    Median runtime: {np.median(runtimes):.2f} seconds")
            print(f"    Std: {np.std(runtimes):.2f} seconds")
            print(f"    Total runtime: {np.sum(runtimes):.2f} seconds ({np.sum(runtimes)/3600:.2f} hours)")
        else:
            print(f"  Runtime from entropy JSON files: No runtime_seconds field found")
        
        # Runtime estimated from Slurm jobs
        if condition in slurm_data and len(slurm_data[condition]) > 0:
            all_jobs = slurm_data[condition]
            completed_jobs = [j for j in all_jobs if j['state'] == 'COMPLETED']
            
            if all_jobs:
                print(f"  Runtime from Slurm job data:")
                # Count questions processed
                condition_dir = condition_dirs.get(condition)
                if condition_dir:
                    entropy_files = glob.glob(os.path.join(condition_dir, "*_entropy.json"))
                    num_questions = len(entropy_files)
                    
                    for job in all_jobs:
                        elapsed_min = job['elapsed_seconds'] / 60
                        cpu_min = job['cpu_seconds'] / 60
                        state_str = job['state']
                        print(f"    Job {job['job_id']} ({state_str}): {elapsed_min:.2f} minutes wall-clock ({cpu_min:.2f} min CPU)")
                        if num_questions > 0 and job['elapsed_seconds'] > 0:
                            avg_per_question = job['elapsed_seconds'] / num_questions
                            print(f"      Average per question: {avg_per_question:.2f} seconds ({avg_per_question/60:.2f} minutes)")
                            print(f"      Questions processed: {num_questions}")
                        if job['max_rss_mb'] > 0:
                            print(f"      Peak memory: {job['max_rss_mb']/1024:.2f} GB")
                else:
                    for job in all_jobs:
                        elapsed_min = job['elapsed_seconds'] / 60
                        cpu_min = job['cpu_seconds'] / 60
                        print(f"    Job {job['job_id']} ({job['state']}): {elapsed_min:.2f} minutes wall-clock ({cpu_min:.2f} min CPU)")
    
    # Create visualizations if plotting is available
    if HAS_PLOTTING and slurm_data:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 14
        
        # 1. Wall-clock Time Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot - include all jobs (COMPLETED, OUT_OF_MEMORY, etc.)
        ax = axes[0]
        data_to_plot = []
        labels = []
        for condition in all_conditions:
            if condition in slurm_data:
                jobs = slurm_data[condition]
                if jobs:
                    elapsed = [j['elapsed_seconds']/60 for j in jobs]  # Convert to minutes
                    data_to_plot.append(elapsed)
                    labels.append(condition.replace('-', ' ').title())
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Wall-clock Time (minutes)', fontsize=14, fontweight='bold')
            ax.set_title('Job Wall-clock Time Comparison', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bar plot with means - include all jobs
        ax = axes[1]
        conditions_plot = []
        means = []
        stds = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in slurm_data:
                jobs = slurm_data[condition]
                if jobs:
                    elapsed = [j['elapsed_seconds']/60 for j in jobs]
                    conditions_plot.append(condition.replace('-', ' ').title())
                    means.append(np.mean(elapsed))
                    stds.append(np.std(elapsed))
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, means, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds, capsize=8)
            ax.set_ylabel('Mean Wall-clock Time (minutes)', fontsize=14, fontweight='bold')
            ax.set_title('Mean Job Wall-clock Time', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                       f'{mean:.1f} ± {std:.1f} min',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_wallclock_time_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. CPU Time Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot - include all jobs
        ax = axes[0]
        data_to_plot = []
        labels = []
        for condition in all_conditions:
            if condition in slurm_data:
                jobs = slurm_data[condition]
                if jobs:
                    cpu = [j['cpu_seconds']/3600 for j in jobs]  # Convert to CPU-hours
                    data_to_plot.append(cpu)
                    labels.append(condition.replace('-', ' ').title())
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            ax.set_ylabel('Total CPU Time (CPU-hours)', fontsize=14, fontweight='bold')
            ax.set_title('Total CPU Time per Job', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bar plot with totals - include all jobs
        ax = axes[1]
        conditions_plot = []
        totals = []
        colors_bar = []
        for i, condition in enumerate(all_conditions):
            if condition in slurm_data:
                jobs = slurm_data[condition]
                if jobs:
                    total_cpu = sum([j['cpu_seconds']/3600 for j in jobs])
                    conditions_plot.append(condition.replace('-', ' ').title())
                    totals.append(total_cpu)
                    colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, totals, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2)
            ax.set_ylabel('Total CPU Time (CPU-hours)', fontsize=14, fontweight='bold')
            ax.set_title('Total CPU Time Across All Jobs', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, total in zip(bars, totals):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(totals)*0.01,
                       f'{total:.2f} CPU-h',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_cpu_time_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Memory Usage Comparison - include all jobs
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
            ax.set_ylabel('Max Memory Usage (GB)', fontsize=14, fontweight='bold')
            ax.set_title('Peak Memory Usage per Job', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_memory_usage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Per-Question Runtime Comparison (wall-clock time / questions processed)
        fig, ax = plt.subplots(figsize=(12, 6))
        conditions_plot = []
        per_question_times = []
        colors_bar = []
        stds_per_q = []
        
        for i, condition in enumerate(all_conditions):
            if condition in slurm_data:
                jobs = slurm_data[condition]
                condition_dir = condition_dirs.get(condition)
                if condition_dir and jobs:
                    # Count questions processed
                    entropy_files = glob.glob(os.path.join(condition_dir, "*_entropy.json"))
                    num_questions = len(entropy_files)
                    
                    if num_questions > 0:
                        # Calculate per-question runtime for each job
                        per_q_times = []
                        for job in jobs:
                            if job['elapsed_seconds'] > 0:
                                per_q = job['elapsed_seconds'] / num_questions
                                per_q_times.append(per_q / 60)  # Convert to minutes
                        
                        if per_q_times:
                            conditions_plot.append(condition.replace('-', ' ').title())
                            per_question_times.append(np.mean(per_q_times))
                            stds_per_q.append(np.std(per_q_times))
                            colors_bar.append(colors[i % len(colors)])
        
        if conditions_plot:
            bars = ax.bar(conditions_plot, per_question_times, color=colors_bar, alpha=0.7, 
                         edgecolor='black', linewidth=2, yerr=stds_per_q, capsize=8)
            ax.set_ylabel('Average Runtime per Question (minutes)', fontsize=14, fontweight='bold')
            ax.set_title('Average Computational Cost per Question', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, mean, std in zip(bars, per_question_times, stds_per_q):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                       f'{mean:.2f} ± {std:.2f} min',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_per_question_runtime.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}/")
    
    # Save detailed data to JSON
    output_data = {
        'slurm_jobs': {k: v for k, v in slurm_data.items()},
        'runtime_from_entropy': {k: v for k, v in runtime_data.items()}
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

