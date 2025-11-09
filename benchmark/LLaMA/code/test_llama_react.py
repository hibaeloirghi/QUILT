import os
import argparse
import jsonlines
import datetime
import sys
import numpy as np

# Add ReAct code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ReAct/code'))
from util import summarize_react_trial, log_react_trial, remove_fewshot

from llama_react_agent import ReactAgent

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="flight", 
                    help="Dataset name: flight, coffee, yelp, airbnb, dblp, gsm8k, scirex, agenda")
parser.add_argument("--hardness", type=str, default="easy", choices=["easy", "hard"],
                    help="Difficulty level: easy or hard")
parser.add_argument("--path", type=str, default="/fs/nexus-scratch/eloirghi/QUILT",
                    help="Path to ToolQA root directory")
parser.add_argument("--wolframalpha_api_key", type=str, default=None,
                    help="WolframAlpha API key for calculator tool")
parser.add_argument("--debug", action="store_true",
                    help="Debug mode: run single question")
parser.add_argument("--debug_id", type=int, default=0,
                    help="Question index for debug mode")
parser.add_argument("--prompt", type=str, default="easy", choices=["easy", "hard"],
                    help="Prompt type: easy or hard")
parser.add_argument("--llama_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="Llama model name from HuggingFace")
parser.add_argument("--hf_token", type=str, default=None,
                    help="HuggingFace token for accessing Llama models")
parser.add_argument("--max_steps", type=int, default=20,
                    help="Maximum number of steps for the agent")
parser.add_argument("--max_questions", type=int, default=None,
                    help="Maximum number of questions to process (None = all)")
parser.add_argument("--entropy_threshold", type=float, default=6.0,
                    help="Entropy threshold for action backtracking (higher = less backtracking, set to 999 to disable)")
parser.add_argument("--max_retries_per_step", type=int, default=3,
                    help="Maximum number of retries per step when entropy is too high")
parser.add_argument("--num_action_samples", type=int, default=5,
                    help="Number of action samples to generate for entropy-based selection (default: 5, original code had 1)")
parser.add_argument("--num_answer_samples", type=int, default=10,
                    help="Number of final answer samples for entropy computation (default: 10, set to 0 to disable)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for logs and results")

args = parser.parse_args()

# Normalize "flights" to "flight" (singular)
if args.dataset == "flights":
    args.dataset = "flight"

# Set environment variables
if args.wolframalpha_api_key:
    os.environ['WOLFRAMALPHA_API_KEY'] = args.wolframalpha_api_key

# Set output directory (include num_action_samples to distinguish runs)
if args.output_dir is None:
    samples_suffix = f"samples{args.num_action_samples}" if args.num_action_samples != 1 else "no-sampling"
    args.output_dir = os.path.join(args.path, 'benchmark/LLaMA/logs', 
                                   f'llama-3.1-8b-{datetime_string}-{samples_suffix}', 
                                   f'{args.dataset}-{args.hardness}')

# Load questions
file_path = os.path.join(args.path, "data/questions", args.hardness, f"{args.dataset}-{args.hardness}.jsonl")
if not os.path.exists(file_path):
    print(f"Error: Question file not found: {file_path}")
    sys.exit(1)

with open(file_path, 'r') as f:
    contents = []
    for item in jsonlines.Reader(f):
        contents.append(item)

print(f"Loaded {len(contents)} questions from {file_path}")

if args.debug:
    # Debug mode: run single question
    random_indices = args.debug_id
    if random_indices >= len(contents):
        print(f"Error: debug_id {random_indices} is out of range (max: {len(contents)-1})")
        sys.exit(1)
    
    # Load model once for debug mode too (to match normal mode behavior)
    print("Loading model...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from huggingface_hub.hf_api import HfFolder
    
    if args.hf_token:
        HfFolder.save_token(args.hf_token)
    
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model,
        dtype=dtype,
        device_map="auto",
        max_memory={0: "20GiB"},
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=dtype,
        model_kwargs={"temperature": 0.0, "do_sample": False}
    )
    print("Model loaded successfully.")
    
    test_q = contents[random_indices]['question']
    test_a = contents[random_indices]['answer']
    print(f"\n=== Debug Mode: Question {random_indices} ===")
    print(f"Question: {test_q}")
    print(f"Answer: {test_a}\n")
    
    # Use preloaded model in debug mode too
    agent = ReactAgent(args, test_q, test_a, max_steps=args.max_steps,
                      llama_model_name=args.llama_model, hf_token=args.hf_token,
                      preloaded_pipe=pipe, preloaded_tokenizer=tokenizer)
    agent.run()
    
    print(f"\n=== Agent Scratchpad ===")
    print(agent._build_agent_prompt())
    print(f"\n=== Results ===")
    print(f"Ground-Truth: {test_a}")
    print(f"Model Answer: {agent.answer}")
    print(f"Correct: {agent.is_correct()}")
    
    # Compute entropy in debug mode too
    print("\n=== Computing Entropy ===")
    import sys
    sys.stdout.flush()
    # Compute final answer entropy if enabled
    if args.num_answer_samples > 0:
        entropy_data = agent.compute_final_answer_entropy(n_samples=args.num_answer_samples, temperature=0.8)
    else:
        entropy_data = agent.entropy_data
        print("Skipping final answer entropy computation (num_answer_samples=0)")
    
    print(f'\nEntropy Metrics:')
    print(f'  Predictive H(Y|Z,x): {entropy_data["predictive_entropy"]:.4f}')
    print(f'  Semantic H_c(Y|Z,x): {entropy_data["semantic_entropy"]:.4f}')
    print(f'  Tool H(Z|a): {sum(entropy_data["tool_entropies"]):.4f}')
    print(f'  STA-P: {entropy_data["sta_predictive"]:.4f}')
    print(f'  STA-S: {entropy_data["sta_semantic"]:.4f}')
    print(f'\nFew-Shot Analysis:')
    print(f'  Few-shot examples: {entropy_data.get("fewshot_examples", 0)}')
    print(f'  Step-wise entropies: {len(entropy_data.get("step_entropies", []))} steps tracked')
    if entropy_data.get("step_entropies"):
        avg_step_entropy = np.mean([s["predictive_entropy"] for s in entropy_data["step_entropies"] if s.get("predictive_entropy") is not None])
        print(f'  Avg step entropy: {avg_step_entropy:.4f}')
    
    # Save entropy data in debug mode too
    import json
    import os
    debug_output_dir = os.path.join(args.path, "benchmark/LLaMA/logs/debug")
    os.makedirs(debug_output_dir, exist_ok=True)
    entropy_file = os.path.join(debug_output_dir, f"debug-{random_indices}_entropy.json")
    with open(entropy_file, 'w') as f:
        json.dump({
            'qid': f"debug-{random_indices}",
            'question': test_q,
            'answer': agent.answer,
            'ground_truth': test_a,
            'correct': agent.is_correct(),
            **entropy_data
        }, f, indent=2)
    
    # Also save a .txt log with discarded actions
    log_file = os.path.join(debug_output_dir, f"debug-{random_indices}.txt")
    log = f"""
########################################
BEGIN TRIAL debug-{random_indices}
#######################################
"""
    log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {test_a}\n\n'
    
    # Add action samples section (all 5 samples with selected/discarded info)
    if entropy_data.get('action_samples') and len(entropy_data['action_samples']) > 0:
        log += f'\n{"="*60}\n'
        log += f'ACTION SAMPLES (All samples per step)\n'
        log += f'{"="*60}\n'
        for action_sample_group in entropy_data['action_samples']:
            step = action_sample_group['step']
            samples = action_sample_group.get('samples', [])
            log += f"Step {step}:\n"
            for sample in samples:
                status = "SELECTED" if sample.get('selected', False) else "discarded"
                log += f"  [{status}] Sample {sample['index']}: {sample.get('text', '')[:80]}\n"
                log += f"    Entropy: {sample.get('entropy', 0):.4f}\n"
            log += f"\n"
        log += f'{"="*60}\n\n'
    
    with open(log_file, 'w') as f:
        f.write(log)
    
    print(f'\nEntropy data saved to: {entropy_file}')
    print(f'Log file saved to: {log_file}')
else:
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load model once and reuse across questions to save memory
    print("Loading model...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from huggingface_hub.hf_api import HfFolder
    
    if args.hf_token:
        HfFolder.save_token(args.hf_token)
    
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_model,
        dtype=dtype,
        device_map="auto",
        max_memory={0: "20GiB"},
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=dtype,
        model_kwargs={"temperature": 0.0, "do_sample": False}
    )
    print("Model loaded successfully.")
    
    # Process all questions (or just one if debug mode)
    agents = []
    unanswered_questions = []
    
    # If debug mode, only process one question
    if args.debug:
        indices = [args.debug_id] if args.debug_id < len(contents) else [0]
        print(f"DEBUG MODE: Processing only question {args.debug_id}")
    else:
        # Limit to max_questions if specified
        if args.max_questions is not None:
            indices = range(min(args.max_questions, len(contents)))
            print(f"Processing first {len(indices)} questions (max_questions={args.max_questions})")
        else:
            indices = range(len(contents))
            print(f"Processing all {len(contents)} questions")
    
    for i in indices:
        qid = contents[i]['qid']
        question = contents[i]['question']
        answer = contents[i]['answer']
        
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(contents)}: {qid}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Create agent with pre-loaded model
        agent = ReactAgent(args, question, answer, max_steps=args.max_steps,
                          llama_model_name=args.llama_model, hf_token=args.hf_token,
                          preloaded_pipe=pipe, preloaded_tokenizer=tokenizer)
        
        try:
            print("Running agent...")
            import sys
            sys.stdout.flush()  # Force output to appear immediately
            agent.run()
            print("Agent finished. Computing entropy...")
            sys.stdout.flush()
            
            # Compute final answer entropy if enabled
            if args.num_answer_samples > 0:
                entropy_data = agent.compute_final_answer_entropy(n_samples=args.num_answer_samples, temperature=0.8)
            else:
                entropy_data = agent.entropy_data
                print("Skipping final answer entropy computation (num_answer_samples=0)")
            print("Entropy computation finished.")
            sys.stdout.flush()
            
            print(f'\nAnswer: {agent.answer}')
            print(f'Ground Truth: {agent.key}')
            print(f'Correct: {agent.is_correct()}')
            print(f'\nEntropy Metrics:')
            print(f'  Predictive H(Y|Z,x): {entropy_data["predictive_entropy"]:.4f}')
            print(f'  Semantic H_c(Y|Z,x): {entropy_data["semantic_entropy"]:.4f}')
            print(f'  Tool H(Z|a): {sum(entropy_data["tool_entropies"]):.4f}')
            print(f'  STA-P: {entropy_data["sta_predictive"]:.4f}')
            print(f'  STA-S: {entropy_data["sta_semantic"]:.4f}')
            print(f'\nFew-Shot Analysis:')
            print(f'  Few-shot examples: {entropy_data.get("fewshot_examples", 0)}')
            print(f'  Step-wise entropies: {len(entropy_data.get("step_entropies", []))} steps tracked')
            if entropy_data.get("step_entropies"):
                avg_step_entropy = np.mean([s["predictive_entropy"] for s in entropy_data["step_entropies"] if s.get("predictive_entropy") is not None])
                print(f'  Avg step entropy: {avg_step_entropy:.4f}')
            print('-'*60)
            
            # Save log
            log = f"""
########################################
BEGIN TRIAL {qid}
#######################################
"""
            log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
            
            # Add action samples section (all 5 samples with selected/discarded info)
            if entropy_data.get('action_samples') and len(entropy_data['action_samples']) > 0:
                num_samples = len(entropy_data['action_samples'][0].get('samples', [])) if entropy_data['action_samples'][0].get('samples') else 5
                log += f'\n{"="*60}\n'
                log += f'ACTION SAMPLES (All {num_samples} samples per step)\n'
                log += f'{"="*60}\n'
                for action_sample_group in entropy_data['action_samples']:
                    step = action_sample_group['step']
                    samples = action_sample_group.get('samples', [])
                    log += f"Step {step}:\n"
                    for sample in samples:
                        status = "SELECTED" if sample.get('selected', False) else "discarded"
                        log += f"  [{status}] Sample {sample['index']}: {sample.get('text', '')[:80]}\n"
                        log += f"    Entropy: {sample.get('entropy', 0):.4f}\n"
                    log += f"\n"
                log += f'{"="*60}\n\n'
            
            log_file = os.path.join(args.output_dir, f"{qid}.txt")
            with open(log_file, 'w') as f:
                f.write(log)
            
            # Save entropy data as JSON
            import json
            entropy_file = os.path.join(args.output_dir, f"{qid}_entropy.json")
            with open(entropy_file, 'w') as f:
                json.dump({
                    'qid': qid,
                    'question': question,
                    'answer': agent.answer,
                    'ground_truth': agent.key,
                    'correct': agent.is_correct(),
                    **entropy_data
                }, f, indent=2)
                
        except Exception as e:
            print(f'Error when computing answer for {qid}: {e}')
            import traceback
            traceback.print_exc()
            print('-'*60)
            
            # Save error log
            log = f"""
########################################
BEGIN TRIAL {qid}
#######################################
"""
            try:
                log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
                # Add action samples if available
                if hasattr(agent, 'entropy_data') and agent.entropy_data.get('action_samples'):
                    log += f'\n{"="*60}\n'
                    log += f'ACTION SAMPLES (All samples per step)\n'
                    log += f'{"="*60}\n'
                    for action_sample_group in agent.entropy_data['action_samples']:
                        step = action_sample_group['step']
                        samples = action_sample_group.get('samples', [])
                        log += f"Step {step}:\n"
                        for sample in samples:
                            status = "SELECTED" if sample.get('selected', False) else "discarded"
                            log += f"  [{status}] Sample {sample['index']}: {sample.get('text', '')[:80]}\n"
                            log += f"    Entropy: {sample.get('entropy', 0):.4f}\n"
                        log += f"\n"
                    log += f'{"="*60}\n\n'
            except:
                log += f'Question: {question}\nCorrect answer: {answer}\n\n'
            log += f'ERROR: {str(e)}\n'
            log_file = os.path.join(args.output_dir, f"{qid}.txt")
            with open(log_file, 'w') as f:
                f.write(log)
            
            unanswered_questions.append(qid)
        
        agents.append(agent)
        
        # Clean up GPU memory between questions to prevent OOM # kept running into this problemo
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    # Summary
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'\n{"="*60}')
    print(f'Summary')
    print(f'{"="*60}')
    print(f'Total questions: {len(agents)}')
    print(f'Correct: {len(correct)} ({len(correct)/len(agents)*100:.1f}%)')
    print(f'Incorrect: {len(incorrect)} ({len(incorrect)/len(agents)*100:.1f}%)')
    print(f'Halted: {len(halted)} ({len(halted)/len(agents)*100:.1f}%)')
    print(f'Unanswered questions: {len(unanswered_questions)}')
    if unanswered_questions:
        print(f'  {unanswered_questions}')
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Hardness: {args.hardness}\n")
        f.write(f"Model: {args.llama_model}\n")
        f.write(f"Total questions: {len(agents)}\n")
        f.write(f"Correct: {len(correct)} ({len(correct)/len(agents)*100:.1f}%)\n")
        f.write(f"Incorrect: {len(incorrect)} ({len(incorrect)/len(agents)*100:.1f}%)\n")
        f.write(f"Halted: {len(halted)} ({len(halted)/len(agents)*100:.1f}%)\n")
        f.write(f"Unanswered questions: {len(unanswered_questions)}\n")
        if unanswered_questions:
            f.write(f"  {unanswered_questions}\n")
    
    print(f"\nResults saved to: {args.output_dir}")

