from transformers import AutoModelForCausalLM
import torch

def load_model(model_name_or_path, force_download=False):
    """Load a causal language model using AutoModelForCausalLM."""
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, force_download)
    return model

def compare_parameters(pretrained_model, finetuned_model):
    """Compare parameters of two models."""
    changed_params = []
    unchanged_params = []

    pretrained_params = pretrained_model.state_dict()
    finetuned_params = finetuned_model.state_dict()

    for key in pretrained_params:
        if key not in finetuned_params:
            print(f"Warning: Parameter {key} not found in fine-tuned model.")
            continue

        if torch.equal(pretrained_params[key], finetuned_params[key]):
            unchanged_params.append(key)
        else:
            changed_params.append(key)

    return changed_params, unchanged_params

def save_results_to_file(changed_params, unchanged_params, output_file):
    """Save the comparison results to a file."""
    with open(output_file, "w") as f:
        f.write(f"Number of changed parameters: {len(changed_params)}\n")
        f.write(f"Number of unchanged parameters: {len(unchanged_params)}\n\n")

        if changed_params:
            f.write("Changed parameters:\n")
            for param in changed_params:
                f.write(f" - {param}\n")

        if unchanged_params:
            f.write("\nUnchanged parameters:\n")
            for param in unchanged_params:
                f.write(f" - {param}\n")

def main():
    # Paths or names of the pre-trained and fine-tuned models
    pretrained_model_name_or_path = "llm-stacking/StackLLM_3B_300BToken"   # Replace with your pre-trained model path or name
    finetuned_model_name_or_path = "../models/stack_3b_m1/checkpoint-100"  # Replace with your fine-tuned model path

    # Output file to save results
    output_file = "naive_vs_ft_3b_model.txt"

    # Load models
    pretrained_model= AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, force_download=True)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name_or_path)
    
    # Compare parameters
    changed_params, unchanged_params = compare_parameters(pretrained_model, finetuned_model)

    # Save results to file
    save_results_to_file(changed_params, unchanged_params, output_file)
    print(f"Comparison results saved to {output_file}")

if __name__ == "__main__":
    main()