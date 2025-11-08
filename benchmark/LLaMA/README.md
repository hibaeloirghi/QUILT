# Llama-3.1-8B-Instruct ReAct Agent for ToolQA

This directory contains the implementation of a ReAct (Reasoning + Acting) agent using Llama-3.1-8B-Instruct for the ToolQA benchmark, following the approach from the Tools-in-the-loop paper.

## Setup

### 1. Install Requirements

First, install the required dependencies:

```bash
cd /fs/nexus-scratch/eloirghi/QUILT
pip install -r requirements.txt
```

### 2. Set Up HuggingFace Token

You need a HuggingFace token to access the Llama-3.1-8B-Instruct model:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or set it in the shell script (see below).

### 3. Optional: Set Up WolframAlpha API Key (for Calculator Tool)

If you want to use the calculator tool, set the WolframAlpha API key:

```bash
export WOLFRAMALPHA_API_KEY="your_wolframalpha_key_here"
```

## Usage

### Running with SLURM (Recommended)

Use the provided shell script to submit a job:

```bash
cd /fs/nexus-scratch/eloirghi/QUILT
sbatch benchmark/LLaMA/scripts/run_llama_react.sh [dataset] [hardness] [prompt_type]
```

Examples:
```bash
# Run easy flight questions
sbatch benchmark/LLaMA/scripts/run_llama_react.sh flights easy easy

# Run hard coffee questions
sbatch benchmark/LLaMA/scripts/run_llama_react.sh coffee hard hard
```

**Parameters:**
- `dataset`: One of `flights`, `coffee`, `yelp`, `airbnb`, `dblp`, `gsm8k`, `scirex`, `agenda`
- `hardness`: `easy` or `hard`
- `prompt_type`: `easy` or `hard` (prompt template type)

### Running Directly

You can also run the test script directly:

```bash
cd /fs/nexus-scratch/eloirghi/QUILT
python benchmark/LLaMA/code/test_llama_react.py \
    --dataset flights \
    --hardness easy \
    --path /fs/nexus-scratch/eloirghi/QUILT \
    --prompt easy \
    --llama_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --hf_token $HF_TOKEN \
    --max_steps 20
```

### Debug Mode

To test with a single question:

```bash
python benchmark/LLaMA/code/test_llama_react.py \
    --dataset flights \
    --hardness easy \
    --debug True \
    --debug_id 0 \
    --llama_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --hf_token $HF_TOKEN
```

## Output

Results are saved to:
```
benchmark/LLaMA/logs/llama-3.1-8b-<timestamp>/<dataset>-<hardness>/
```

Each question gets its own log file (`<qid>.txt`), and a summary is saved in `summary.txt`.

## Implementation Details

The implementation includes:

1. **LlamaLLM**: A custom LangChain LLM wrapper that uses the transformers pipeline to interface with Llama-3.1-8B-Instruct
2. **ReactAgent**: A ReAct agent that can use multiple tools:
   - Calculate (WolframAlpha)
   - RetrieveAgenda / RetrieveScirex (text retrieval)
   - LoadDB / FilterDB / GetValue (database operations)
   - LoadGraph / NeighbourCheck / NodeCheck / EdgeCheck (graph operations)
   - SQLInterpreter / PythonInterpreter (code execution)
   - Finish (answer submission)

The agent follows the ReAct pattern: Thought → Action → Observation, iterating until it finds the answer or reaches the maximum number of steps.

## Notes

- The model is loaded with `bfloat16` precision and `device_map="auto"` for efficient GPU memory usage
- Maximum steps default to 20, but can be adjusted
- The agent uses the same tool implementations as the original ReAct baseline
- Token counting uses tiktoken with GPT-4 encoding as an approximation

