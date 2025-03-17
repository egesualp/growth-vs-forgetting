# Gradual Growth vs Naive Learning in Continuous Learning Scenarios

This repository contains the implementation and experiments for comparing **gradual growth methods**, specifically the G_stack approach, with **naive models trained from scratch**. The project focuses on addressing **catastrophic forgetting** and improving model performance in continuous learning scenarios.

keywords: probabilistic continuous learning, deterministic and probabilistic fine tuning, catastrophic forgetting, model growth, g_stack, decoder-only

---

## Objectives

1. Compare the effectiveness of G_stack and naive models in continuous learning tasks. (Done)
2. Investigate the impact of **deterministic** and **probabilistic fine-tuning** on model performance. (WIP)
3. Analyze how these methods mitigate catastrophic forgetting when revisiting the same task and dataset over time. (Done)

---

## Repository Structure

```
gradual-growth-vs-naive-learning/
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   ├── preprocessed/          # Preprocessed datasets for finetuning
│   ├── results/               # Experiment outputs
├── src/                       # Source code
│   ├── logs/                  # Logs of model related investigations
│   ├── models/                # Utilities for data handling and metrics (might be empty due to storage limitation)
│   ├── utils/                 # Training and evaluation scripts
│   ├── debuggers/             # Debuggers used during the project 
├── experiments/               # Experiment configurations and logs
    ├── configs/               # Configurations
    ├── logs/                  # Logs of experiments
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment file (optional)
└── README.md                  # Project overview
```

---

## Installation

### Prerequisites
- Python 3.11 or later
- Recommended: [Conda](https://docs.conda.io/en/latest/) for environment management

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/growth-vs-forgetting.git
   cd growth-vs-forgetting
   ```
2. Install dependencies:
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Using Conda:
     ```bash
     conda env create -f environment.yml
     conda activate gradual-growth-env
     ```

---

## Usage

### 1. Train Models
To finetune models with the tasks, first preprocess raw datasets:
```bash
python src/utils/data_formatting.py --folder_path wiki_auto --output_path simp_wiki_auto.json --eval_dataset_size 0.01
```
Then, you can run fine-tuning process using shell script. Don't forget to set dataset_format = "prompt" to have a general prompt added to each training sample. Feel free to play around with deepspeed config, based on your system. 
```bash
cd src/utils

deepspeed --num_gpus=6 finetune.py \
    --model_name_or_path "llm-stacking/LLM_7B_300BToken" \
    --dataset "../../data/preprocessed/simp_wiki_auto_new.json" \
    --dataset_format "prompt" \
	  --output_dir "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m1_prompt" \
	  --run_name "llm_7b_m1_prompt_exp2" \
    --num_train_epochs 3 \
    --max_steps -1 \
	  --per_device_train_batch_size 8 \
	  --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 16 \
	  --save_strategy 'epoch' \
    --evaluation_strategy 'steps' \
    --eval_steps 300 \
  	--save_total_limit 1 \
	  --learning_rate 2e-5 \
    --logging_steps 150 \
	  --lr_scheduler_type 'constant' \
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
    --task 'simp'
```

### 2. Test Models
You can also test the model' inference performance using the same script above, with different arguments

```bash
deepspeed --num_gpus=4 finetune_v3.py \
    --model_name_or_path "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m1_prompt/checkpoint-6189" \
    --dataset "../../data/preprocessed/simp_wiki_auto_new.json" \
    --dataset_format "prompt" \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/llm_7b_m1_prompt/test \
    --run_name "llm_7b_m0_inf_prompt" \
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
    --num_beams 3 \
    --length_penalty 0.8 \
    --remove_unused_columns false \
    --compute_rouge true \
    --compute_sari true \
    --compute_bleu true \
    --max_new_tokens 184 \
    --task "simp"
```
### 3. Evaluate Models (Benchmark tasks)

We have used lm-evaluation-harness to assess whether catastrophic forgetting. Each model has been evaluated iteratively with one shell script. Please refer to llm_eval_tasks.sbatch or stack_eval_tasks.sbatch.

### 4. Visualize Results
Use the provided Jupyter notebooks:
```bash
jupyter notebook notebooks/results_visualization.ipynb
```

---

## Results
Detailed results, including metrics like accuracy and r@1 scores, are stored in the `data/results/` directory and visualized in the `notebooks/`. For the inference results in each task, please contact. 

---

## Contact
For questions or collaboration, contact [Ege Süalp](mailto:e.sualp@campus.lmu.de).


