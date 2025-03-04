#!/bin/bash
#SBATCH --job-name=inf_inqQG_baseline        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/inf_inqQG_baseline_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/inf_inqQG_baseline_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:4                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=02:00:00                   # Time limit

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clean

# Change directory to where the script is located
cd src/utils

echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/transformers"
export HF_HOME="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/datasets"

# Log GPU Info
nvidia-smi

# Baseline models to test
models=(
    "llm-stacking/LLM_7B_300BToken"
    "llm-stacking/StackLLM_7B_300BToken"
)

for model_path in "${models[@]}"; do
    # Create a sanitized model name for output directory
    model_name=$(echo "$model_path" | tr '/' '_')
    output_dir="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/${model_name}/test_inqQG"
    mkdir -p "$output_dir"

    deepspeed --num_gpus=4 finetune_v3.py \
        --model_name_or_path "$model_path" \
        --dataset "../../data/preprocessed/inqQG.json" \
        --dataset_format "prompt" \
        --output_dir "$output_dir" \
        --run_name "${model_name}_inqqg" \
        --trust_remote_code true \
        --overwrite_output_dir false \
        --do_train false \
        --do_eval false \
        --do_predict true \
        --predict_with_generate true \
        --per_device_eval_batch_size 8 \
        --dataloader_num_workers 4 \
        --seed 42 \
        --report_to "none" \
        --bf16 true \
        --dataloader_pin_memory true \
        --gradient_checkpointing false \
        --num_beams 4 \
        --length_penalty 0.9 \
        --do_sample false \
        --no_repeat_ngram_size 2 \
        --remove_unused_columns false \
        --compute_rouge true \
        --compute_bert_score true \
        --compute_bleu true \
        --max_new_tokens 64 \
        --task "inqqg"

    echo "Completed inference for baseline model: $model_path"
done

echo "Job Completed!"
