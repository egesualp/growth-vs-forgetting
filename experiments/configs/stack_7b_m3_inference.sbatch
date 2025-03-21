#!/bin/bash
#SBATCH --job-name=inf_stack_7b_m3_p        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/inf_stack_7b_m3_p_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/inf_stack_7b_m3_p_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:4                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=02:00:00                   # Time limit

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clean

# Change directory to where the script is located
cd src/utils

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

# Set environment variables for distributed training
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Debugging: Log GPU Info
nvidia-smi

deepspeed --num_gpus=4 finetune_v3.py \
    --model_name_or_path "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m3_prompt_2/checkpoint-782" \
    --dataset "../../data/preprocessed/inqQG.json" \
    --dataset_format "prompt" \
    --output_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m3_prompt_2/test_782" \
    --run_name "stack_7b_m3_inf_prompt" \
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
    --task "empd"


echo "Job Completed!"
