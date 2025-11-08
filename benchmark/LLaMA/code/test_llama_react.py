import os
import argparse
import jsonlines
import datetime
import sys

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
parser.add_argument("--debug", type=bool, default=False,
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
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for logs and results")

args = parser.parse_args()

# Normalize "flights" to "flight" (singular)
if args.dataset == "flights":
    args.dataset = "flight"

# Set environment variables
if args.wolframalpha_api_key:
    os.environ['WOLFRAMALPHA_API_KEY'] = args.wolframalpha_api_key

# Set output directory
if args.output_dir is None:
    args.output_dir = os.path.join(args.path, 'benchmark/LLaMA/logs', 
                                   f'llama-3.1-8b-{datetime_string}', 
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
    
    test_q = contents[random_indices]['question']
    test_a = contents[random_indices]['answer']
    print(f"\n=== Debug Mode: Question {random_indices} ===")
    print(f"Question: {test_q}")
    print(f"Answer: {test_a}\n")
    
    agent = ReactAgent(args, test_q, test_a, max_steps=args.max_steps,
                      llama_model_name=args.llama_model, hf_token=args.hf_token)
    agent.run()
    
    print(f"\n=== Agent Scratchpad ===")
    print(agent._build_agent_prompt())
    print(f"\n=== Results ===")
    print(f"Ground-Truth: {test_a}")
    print(f"Model Answer: {agent.answer}")
    print(f"Correct: {agent.is_correct()}")
else:
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Process all questions
    agents = []
    unanswered_questions = []
    
    for i in range(len(contents)):
        qid = contents[i]['qid']
        question = contents[i]['question']
        answer = contents[i]['answer']
        
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(contents)}: {qid}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        agent = ReactAgent(args, question, answer, max_steps=args.max_steps,
                          llama_model_name=args.llama_model, hf_token=args.hf_token)
        
        try:
            agent.run()
            
            # Compute final answer entropy
            entropy_data = agent.compute_final_answer_entropy(n_samples=10, temperature=0.8)
            
            print(f'\nAnswer: {agent.answer}')
            print(f'Ground Truth: {agent.key}')
            print(f'Correct: {agent.is_correct()}')
            print(f'\nEntropy Metrics:')
            print(f'  Predictive H(Y|Z,x): {entropy_data["predictive_entropy"]:.4f}')
            print(f'  Semantic H_c(Y|Z,x): {entropy_data["semantic_entropy"]:.4f}')
            print(f'  Tool H(Z|a): {sum(entropy_data["tool_entropies"]):.4f}')
            print(f'  STA-P: {entropy_data["sta_predictive"]:.4f}')
            print(f'  STA-S: {entropy_data["sta_semantic"]:.4f}')
            print('-'*60)
            
            # Save log
            log = f"""
########################################
BEGIN TRIAL {qid}
#######################################
"""
            log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
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
            except:
                log += f'Question: {question}\nCorrect answer: {answer}\n\n'
            log += f'ERROR: {str(e)}\n'
            log_file = os.path.join(args.output_dir, f"{qid}.txt")
            with open(log_file, 'w') as f:
                f.write(log)
            
            unanswered_questions.append(qid)
        
        agents.append(agent)
    
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

