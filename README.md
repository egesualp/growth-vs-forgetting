# Gradual Growth vs Naive Learning in Continuous Learning Scenarios

This repository contains the implementation and experiments for comparing **gradual growth methods**, specifically the G_stack approach, with **naive models trained from scratch**. The project focuses on addressing **catastrophic forgetting** and improving model performance in continuous learning scenarios.

keywords: probabilistic continuous learning, deterministic and probabilistic fine tuning, catastrophic forgetting, model growth, g_stack, FiLM-Ensemble

---

## Objectives

1. Compare the effectiveness of G_stack and naive models in continuous learning tasks.
2. Investigate the impact of **deterministic** and **probabilistic fine-tuning** on model performance.
3. Analyze how these methods mitigate catastrophic forgetting when revisiting the same task and dataset over time.

---

## Repository Structure

```
gradual-growth-vs-naive-learning/
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   ├── results/               # Experiment outputs
├── src/                       # Source code
│   ├── models/                # Model implementations
│   ├── utils/                 # Utilities for data handling and metrics
│   ├── training/              # Training and evaluation scripts
├── experiments/               # Experiment configurations and logs
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment file (optional)
└── README.md                  # Project overview
```

---

## Installation

### Prerequisites
- Python 3.8 or later
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
#### Train G_stack Model:
```bash
python src/training/ftune_g_stack.py --config experiments/configs/g_stack_config.json
```

#### Train Naive Model:
```bash
python src/training/ftune_naive.py --config experiments/configs/naive_model_config.json
```

### 3. Evaluate Models
```bash
python src/training/evaluate.py --model g_stack --metrics ECE
```

### 4. Visualize Results
Use the provided Jupyter notebooks:
```bash
jupyter notebook notebooks/results_visualization.ipynb
```

---

## Key Features

- **Gradual Growth (G_stack)**: Plese refer to https://llm-stacking.github.io/
- **Naive Model Training**: Baseline for evaluating G_stack’s effectiveness.
- **Deterministic and Probabilistic Fine-Tuning**: Explore their impact on calibration and robustness.
- **Metrics**: Evaluate catastrophic forgetting, Expected Calibration Error (ECE), and task accuracy.

---

## Results
Detailed results, including metrics like ECE and diversity scores, are stored in the `data/results/` directory and visualized in the `notebooks/`.

---

## Contact
For questions or collaboration, contact [Ege Süalp](mailto:e.sualp@campus.lmu.de).


