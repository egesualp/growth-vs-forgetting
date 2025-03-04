#!/bin/bash
#SBATCH --job-name=ft_llm_7b_m3_p        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/ft_llm_7b_m3_p_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/ft_llm_7b_m3_p_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:8                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=06:00:00                   # Time limit

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clean

# Change directory to where the script is located
cd src/utils

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

# Set environment variables for distributed training
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7               # Use 2 GPUs
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/transformers"
export HF_HOME="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface/datasets"

# Debugging: Log GPU Info
nvidia-smi

deepspeed --num_gpus=8 finetune_v3.py \
    --model_name_or_path "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/llm_7b_m2_prompt/checkpoint-3546" \
    --dataset "../../data/preprocessed/inqQG.json" \
    --dataset_format "prompt" \
	--output_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m3_prompt" \
	--run_name "llm_7b_m3_prompt" \
    --max_train_samples 100000 \
    --num_train_epochs 3 \
	--per_device_train_batch_size 16 \
	--gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 16 \
    --max_steps -1 \
	--save_strategy 'epoch' \
    --evaluation_strategy 'steps' \
    --eval_steps 100 \
	--save_total_limit 2 \
	--learning_rate 2e-5 \
    --logging_steps 10 \
	--lr_scheduler_type 'cosine' \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --gradient_checkpointing true \
    --deepspeed 'ds_config_3.json' \
    --bf16 true \
    --report_to "wandb" \
    --logging_first_step true \
	--seed 42 \
    --do_train true \
    --do_eval true \
    --do_predict false \
    --predict_with_generate false \
    --train_on_source false \
    --trust_remote_code true \
    --task 'inqqg'

echo "Job Completed!"
