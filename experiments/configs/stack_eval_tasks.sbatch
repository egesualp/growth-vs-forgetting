#!/bin/bash
#SBATCH --job-name=eval_stack_7b_rc_only        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/eval_stack_7b_rc_only%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/eval_stack_7b_rc_only_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:2                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=00:20:00                   # Time limit

#!/bin/bash

# Activate conda environment
source "/dss/dsshome1/02/ra95kix2/miniconda3/bin/activate" evaluator || { echo "Conda activation failed"; exit 1; }

# Change to script's directory
cd "src/utils" || { echo "Directory change failed"; exit 1; }

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

# Set environment variables for distributed training and cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/transformers"
export HF_HOME="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/datasets"

# Debugging: Log GPU Info
nvidia-smi

#!/bin/bash

# Declare associative arrays for models and tasks
declare -A MODELS
MODELS=(
    ["stack_7b_m3"]="/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/llm_7b_m2_prompt/checkpoint-3546"
    #["stack_7b_m1"]="/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/stack_7b_m1_prompt/checkpoint-6189"
    #["stack_7b_m0"]="llm-stacking/StackLLM_7B_300BToken"
)

declare -A TASKS
TASKS=(
    #["Rs"]="piqa,boolq,winogrande,hellaswag,mathqa,mutual"
    ["RC"]="race"
    #["DK"]="mmlu_stem,mmlu_humanities,mmlu_social_sciences,mmlu_other"
    #["Bias"]="crows_pairs"
)

# Output CSV
SUMMARY_FILE="evaluation_summary.csv"
echo "Model,Tasks,Accuracy,StdDev" > "$SUMMARY_FILE"

# Loop over models and tasks
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    output_dir="/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/stack_evaluations"
    mkdir -p "$output_dir" || exit 1

    for task_key in "${!TASKS[@]}"; do
        task_string="${TASKS[$task_key]}"
        output_file="${output_dir}/results_${task_key}.json"
        echo "Evaluating ${model_name} on ${task_key} (${task_string})..."

        num_fewshot=0
        [[ "$task_string" =~ "mmlu" ]] && num_fewshot=5

        accelerate launch --num_processes 2 -m lm_eval --model hf \
            --model_args pretrained="${model_path}" \
            --tasks "${task_string}" \
            --batch_size 8 \
            --num_fewshot "$num_fewshot" \
            --trust_remote_code \
            --output_path "$output_file" || continue

        [ ! -f "$output_file" ] && { echo "Output file $output_file not found"; continue; }

        # Split tasks and extract metrics
        IFS=',' read -r -a task_array <<< "$task_string"
        for task in "${task_array[@]}"; do
            ACCURACY=$(jq -r ".results.\"$task\".acc // empty" "$output_file")
            STDDEV=$(jq -r ".results.\"$task\".acc_stderr // empty" "$output_file")
            [ -z "$ACCURACY" ] || [ -z "$STDDEV" ] && { echo "Metrics missing for $task"; continue; }
            echo "${model_name},${task},${ACCURACY},${STDDEV}" >> "$SUMMARY_FILE"
        done
    done
done

echo "Done. Results in ${SUMMARY_FILE}."