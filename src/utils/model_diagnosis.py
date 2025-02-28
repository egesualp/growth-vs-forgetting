import torch
import argparse
from transformers import AutoModelForCausalLM

def check_model(model_name):
    output_lines = []
    output_lines.append(f"\n🚀 Checking model: {model_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    output_lines.append(f"🧠 Total Parameters: {total_params:,}")
    output_lines.append(f"🔄 Trainable Parameters: {trainable_params:,}")
    output_lines.append(f"❄️ Frozen Parameters: {total_params - trainable_params:,}")

    # Count frozen layers
    frozen_layers = [name for name, param in model.named_parameters() if not param.requires_grad]
    if frozen_layers:
        output_lines.append(f"🛑 Frozen Layers: {len(frozen_layers)} (Partial Freezing Detected)")
    else:
        output_lines.append("✅ No Frozen Layers (Full Fine-Tuning)")

    # Check for weight sharing (stacked models may reuse weights)
    weight_counts = {}
    for name, param in model.named_parameters():
        weight_counts[name] = param.data_ptr()

    unique_weights = len(set(weight_counts.values()))
    output_lines.append(f"🔄 Unique Weights: {unique_weights} / {len(weight_counts)}")

    if unique_weights < len(weight_counts):
        output_lines.append("⚠️ Possible Weight Sharing Detected (Stacked Architecture?)")

    # Print model architecture (first 10 layers for overview)
    output_lines.append("\n🔍 Model Architecture Overview:")
    output_lines.append(str(model)[:1000])  # Print first 1000 chars to avoid huge output

    output_lines.append(f"\n PS: # of Layers: {print(len(model.model.layers))}")
    output_lines.append("\n✅ Diagnostics Complete!\n")
    output_file = f'model_diagnostic_llm_7b_m1.txt'

    # Save to file
    with open(output_file, "a") as f:
        f.write("\n".join(output_lines) + "\n" + "="*80 + "\n")

    # Print output to console
    print("\n".join(output_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Model Training Setup")
    parser.add_argument("--model_name", type=str, required=True, help="Path or Hugging Face model name")
    #parser.add_argument("--output_file", type=str, default="model_diagnostics.txt", help="File to save outputs")
    args = parser.parse_args()

    check_model(args.model_name)
