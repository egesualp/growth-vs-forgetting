#!/bin/bash
#SBATCH --job-name=inf_stack_3b_m0        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/inf_stack_3b_m0_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/inf_stack_3b_m0_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:2                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=03:00:00                   # Time limit

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clean

# Change directory to where the script is located
cd src/utils

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"

# Set environment variables for distributed training
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRANSFORMERS_CACHE=~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Debugging: Log GPU Info
nvidia-smi

deepspeed --num_gpus=2 finetune_v3.py \
    --model_name_or_path "llm-stacking/StackLLM_3B_300BToken" \
    --dataset "../../data/preprocessed/simp_wiki_auto_new.json" \
    --dataset_format "input-output" \
    --output_dir "../models/stack_3b_m0/test" \
    --checkpoint_dir "../models/stack_3b_m0" \
    --run_name "stack_3b_m0_inf_exp1" \
    --trust_remote_code true \
    --overwrite_output_dir false \
    --do_train false \
    --do_eval true \
    --do_predict true \
    --predict_with_generate true \
    --per_device_eval_batch_size 8 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --report_to "none" \
    --fp16 true \
    --bf16 false \
    --dataloader_pin_memory true \
    --gradient_checkpointing false \
    --num_beams 3 \
    --temperature 0.9 \
    --top_k 50 \
    --top_p 0.9 \
    --length_penalty 0.8 \
    --remove_unused_columns false \
    --compute_rouge true \
    --compute_sari true \
    --compute_bleu true \
    --max_new_tokens 250 \
    --task "Simp"

echo "Job Completed!"
