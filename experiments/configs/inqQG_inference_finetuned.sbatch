#!/bin/bash
#SBATCH --job-name=inf_inqQG_finetuned        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/inf_inqQG_finetuned_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/inf_inqQG_finetuned_%j.err     # Error output
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
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Log GPU Info
nvidia-smi

# Array of model checkpoints for InqQG
models=(
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m3_prompt_2/checkpoint-1173"
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m3_prompt/checkpoint-1173"
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m3_prompt_2/checkpoint-782"
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m3_prompt/checkpoint-782"
)

# Loop over each model
for model_path in "${models[@]}"; do
    # Extract the parent model directory and checkpoint name
    model_dir="${model_path%/checkpoint-*}"
    checkpoint=$(basename "$model_path")
    output_dir="${model_dir}/test_${checkpoint}"
    mkdir -p "$output_dir"
    
    deepspeed --num_gpus=4 finetune_v3.py \
        --model_name_or_path "$model_path" \
        --dataset "../../data/preprocessed/inqQG.json" \
        --dataset_format "prompt" \
        --output_dir "$output_dir" \
        --run_name "${checkpoint}_inqqg" \
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
    
    echo "Completed inference for model: $model_path"
done

echo "Job Completed!"

