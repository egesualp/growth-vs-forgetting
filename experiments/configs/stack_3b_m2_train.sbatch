#!/bin/bash
#SBATCH --job-name=ft_stack_3b_m2        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/ft_stack_3b_m2_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/ft_stack_3b_m2_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:3                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=04:00:00                   # Time limit

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clean

# Change directory to where the script is located
cd src/utils

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

# Set environment variables for distributed training
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1               # Use 2 GPUs
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Debugging: Log GPU Info
nvidia-smi

deepspeed --num_gpus=3 finetune_v3.py \
    --model_name_or_path "../models/stack_3b_m1/checkpoint-1547" \
    --dataset "../../data/preprocessed/empd.json" \
    --dataset_format "input-output" \
    --output_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_3b_m2" \
    --checkpoint_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_3b_m2" \
    --run_name "stack_3b_m2_exp3" \
    --trust_remote_code true \
    --max_length 512 \
    --max_eval_samples 256 \
    --overwrite_output_dir true \
    --do_train true \
    --do_eval true \
    --do_predict false \
    --evaluation_strategy "steps" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 16 \
    --dataloader_num_workers 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --lr_scheduler_type "constant" \
    --logging_steps 100 \
    --save_strategy "epoch" \
    --eval_steps 200 \
    --save_total_limit 3 \
    --seed 42 \
    --report_to "wandb" \
    --fp16 true \
    --bf16 false \
    --dataloader_pin_memory true \
    --task "empd" \
    --predict_with_generate false \
    --train_on_source false \
    --deepspeed "ds_config_2.json" \
    --gradient_checkpointing true \
    --logging_first_step true \
    --save_only_model true

echo "Job Completed!"
