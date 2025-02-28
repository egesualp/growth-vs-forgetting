#!/bin/bash
#SBATCH --job-name=ft_stack_3b_m1_p        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/ft_stack_3b_m1_p_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/ft_stack_3b_m1_p_%j.err     # Error output
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
export CUDA_VISIBLE_DEVICES=0,1,2               # Use 2 GPUs
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Debugging: Log GPU Info
nvidia-smi

deepspeed --num_gpus=3 new_finetune.py \
    --model_name_or_path "llm-stacking/StackLLM_3B_300BToken" \
	--data_path "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/data/raw" \
	--output_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_3b_m1_prompt" \
	--run_name "stack_3b_m1_prompt_exp2" \
    --num_train_epochs 3 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 16 \
	--save_strategy 'epoch' \
    --evaluation_strategy 'steps' \
    --eval_steps 100 \
	--save_total_limit 3 \
	--learning_rate 2e-5 \
    --warmup_steps 1 \
    --logging_steps 10 \
	--lr_scheduler_type 'constant' \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --fp16 True \
    --report_to "wandb" \
    --logging_first_step true \
	

echo "Job Completed!"