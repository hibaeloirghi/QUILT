#!/bin/bash

#SBATCH --job-name=llama-toolqa
#SBATCH --output=llama_toolqa_%j.out
#SBATCH --error=llama_toolqa_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=20gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger

module load cuda

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set HuggingFace token and cache
export HF_HOME="/fs/nexus-scratch/eloirghi/.cache/huggingface"
# HF_TOKEN must be set as an environment variable - do not hardcode tokens in this file
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo "Please set it with: export HF_TOKEN=your_token"
    exit 1
fi
export PIP_CACHE_DIR="/fs/nexus-scratch/eloirghi/.cache/pip"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST" 
echo "Started at: $(date)"

# Set working directory
cd /fs/nexus-scratch/eloirghi/QUILT

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/fs/nexus-scratch/eloirghi/QUILT/benchmark/ReAct/code:/fs/nexus-scratch/eloirghi/QUILT/benchmark/LLaMA/code"

# Default arguments - modify these as needed
TOOLQA_PATH="/fs/nexus-scratch/eloirghi/QUILT"
DATASET="${1:-flight}"  # flight, coffee, yelp, airbnb, dblp, gsm8k, scirex, agenda
# Normalize "flights" to "flight" (singular)
if [ "$DATASET" = "flights" ]; then
    DATASET="flight"
fi
HARDNESS="${2:-easy}"    # easy or hard
PROMPT="${3:-easy}"       # easy or hard (prompt type)
LLAMA_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_STEPS=20
MAX_QUESTIONS="${4:-50}"  # Number of questions to process (default: 50)

# Optional: WolframAlpha API key for calculator tool
# WOLFRAMALPHA_API_KEY="your_key_here"

# HF_TOKEN check is done earlier in the script

# Print configuration
echo "Configuration:"
echo "  ToolQA path: $TOOLQA_PATH"
echo "  Dataset: $DATASET"
echo "  Hardness: $HARDNESS"
echo "  Prompt type: $PROMPT"
echo "  Llama model: $LLAMA_MODEL"
echo "  Max steps: $MAX_STEPS"
echo "  Max questions: $MAX_QUESTIONS"
echo ""

# Check if question file exists
QUESTION_FILE="${TOOLQA_PATH}/data/questions/${HARDNESS}/${DATASET}-${HARDNESS}.jsonl"
if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file not found: $QUESTION_FILE"
    exit 1
fi

echo "=== Running Llama-3.1-8B-Instruct ReAct on ToolQA ==="
echo ""

# Build and display the command
CMD_ARGS=(
    --dataset "$DATASET"
    --hardness "$HARDNESS"
    --path "$TOOLQA_PATH"
    --prompt "$PROMPT"
    --llama_model "$LLAMA_MODEL"
    --hf_token "$HF_TOKEN"
    --max_steps "$MAX_STEPS"
    --max_questions "$MAX_QUESTIONS"
)

# Add debug flags if specified (for testing single questions)
if [ ! -z "$DEBUG" ] && [ "$DEBUG" = "1" ]; then
    CMD_ARGS+=(--debug)
    if [ ! -z "$DEBUG_ID" ]; then
        CMD_ARGS+=(--debug_id "$DEBUG_ID")
    fi
fi

if [ ! -z "$WOLFRAMALPHA_API_KEY" ]; then
    CMD_ARGS+=(--wolframalpha_api_key "$WOLFRAMALPHA_API_KEY")
fi

echo "Running command:"
# Create a safe display version that masks the HF token
SAFE_CMD_ARGS=()
SKIP_NEXT=false
for arg in "${CMD_ARGS[@]}"; do
    if [ "$SKIP_NEXT" = true ]; then
        SAFE_CMD_ARGS+=("***MASKED***")
        SKIP_NEXT=false
    elif [[ "$arg" == "--hf_token" ]]; then
        SAFE_CMD_ARGS+=("--hf_token")
        SKIP_NEXT=true
    else
        SAFE_CMD_ARGS+=("$arg")
    fi
done
echo "python benchmark/LLaMA/code/test_llama_react.py ${SAFE_CMD_ARGS[*]}"
echo ""

# Execute the command
python benchmark/LLaMA/code/test_llama_react.py "${CMD_ARGS[@]}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Job completed successfully ==="
    echo "Results saved to: benchmark/LLaMA/logs/"
else
    echo ""
    echo "=== Job failed ==="
    exit 1
fi

echo "Finished at: $(date)"

