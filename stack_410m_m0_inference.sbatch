#!/bin/bash
#SBATCH --job-name=stack_410m_m0_inference_sc       # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=logs/inf_stack_410m_m0_sc_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=logs/inf_stack_410m_m0_sc_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:2                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --time=2:00:00                   # Time limit
#SBATCH --mem=128G                       # Total memory allocation

# Load necessary modules (if required, otherwise skip)
# module load cuda/12.2

# Activate conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate clearning

# Change directory to where the script is located
cd src/utils

# Ensure `nvidia-smi` works and CUDA is detected
echo "Running on host $(hostname)"
echo "Using Python from $(which python)"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count()}')"

# Run the training script with accelerate
accelerate launch --config_file /dss/dsshome1/02/ra95kix2/.cache/huggingface/accelerate/default_config.yaml \
    --multi_gpu finetune_v2.py --config stack_410m_m0_inference_simp_sanity_check.yaml
