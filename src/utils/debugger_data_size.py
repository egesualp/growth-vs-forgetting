import json
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
def clean_text(text):
    return text.replace('-RRB-', ')').replace('-LRB-', '(').replace('-RSB-', ']').replace('-LSB-', '[')

# Define file path
file_path = "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/data/preprocessed/inqQG.json"
print(f'File path: {file_path}')
train_dataset = Dataset.from_json(path_or_paths=file_path, field='train')
dataset = train_dataset
dataset = dataset.map(lambda x: {
                    'input': f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{clean_text(x['input'])}\n\n### Response:",
                    'output': clean_text(x['output'])
                })
print(f"Dataset size: {len(dataset['input'])}")
# Function to calculate input and output lengths
def check_lengths(dataset, input_key="input", output_key="output"):
    
    input_lengths = [len(item[input_key].split()) for item in dataset]
    output_lengths = [len(item[output_key].split()) for item in dataset]
    combined_lengths = [len(item[input_key].split()) + len(item[output_key].split()) for item in dataset]

    print(f"Input Lengths: Min={min(input_lengths)}, Max={max(input_lengths)}, Avg={sum(input_lengths)/len(input_lengths):.2f}")
    print(f"Output Lengths: Min={min(output_lengths)}, Max={max(output_lengths)}, Avg={sum(output_lengths)/len(output_lengths):.2f}")
    print(f"Combined Lengths: Min={min(combined_lengths)}, Max={max(combined_lengths)}, Avg={sum(combined_lengths)/len(combined_lengths):.2f}")


# Run the function on the dataset
check_lengths(dataset)

# Load a tokenizer (Replace with your model’s tokenizer)
tokenizer = AutoTokenizer.from_pretrained("llm-stacking/LLM_7B_300BToken")  # Change model if needed

# Function to check tokenized lengths
import numpy as np

def check_token_lengths(dataset, tokenizer, input_key="input", output_key="output"):
    # Tokenize input, output, and combined sequences
    input_lengths = [len(tokenizer(item[input_key], truncation=False)["input_ids"]) for item in dataset]
    output_lengths = [len(tokenizer(item[output_key], truncation=False)["input_ids"]) for item in dataset]
    combined_lengths = [len(tokenizer(item[input_key] + item[output_key], truncation=False)["input_ids"]) for item in dataset]

    # Define quantiles to compute
    quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    # Print summary statistics
    print(f"Tokenized Input Lengths: Min={min(input_lengths)}, Max={max(input_lengths)}, Avg={np.mean(input_lengths):.2f}")
    print(f"  Quantiles: {dict(zip(quantiles, np.quantile(input_lengths, quantiles)))}")

    print(f"Tokenized Output Lengths: Min={min(output_lengths)}, Max={max(output_lengths)}, Avg={np.mean(output_lengths):.2f}")
    print(f"  Quantiles: {dict(zip(quantiles, np.quantile(output_lengths, quantiles)))}")

    print(f"Combined Lengths: Min={min(combined_lengths)}, Max={max(combined_lengths)}, Avg={np.mean(combined_lengths):.2f}")
    print(f"  Quantiles: {dict(zip(quantiles, np.quantile(combined_lengths, quantiles)))}")



# Run the function on the dataset
check_token_lengths(dataset, tokenizer)
