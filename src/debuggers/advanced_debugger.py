import os
import json
import torch
from transformers import AutoModelForCausalLM

# Update these paths before running
BASE_MODEL_PATH = "/dss/dsshome1/02/ra95kix2/.cache/huggingface/models--llm-stacking--StackLLM_7B_300BToken/snapshots/1dffaf0e8b023be315beba18fd75cf1d63434af9/pytorch_model.bin"
FINETUNED_MODEL_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m2_prompt/checkpoint-3546"
DEEPSPEED_CONFIG_PATH = "ds_config_3.json"  # Update this if you have a DeepSpeed config file

def list_checkpoint_files(model_path):
    """List all files in checkpoint directory to check for sharding."""
    print("\n📂 Checking files in checkpoint directory:")
    files = os.listdir(model_path)
    for file in files:
        file_size = round(os.path.getsize(os.path.join(model_path, file)) / (1024**3), 2)
        print(f"- {file} ({file_size} GB)")

def load_model_params(model_path):
    """Load model weights from .bin files."""
    print(f"\n🔍 Loading model from: {model_path}")

    if os.path.isdir(model_path):
        # Check for sharded files
        index_file = os.path.join(model_path, "pytorch_model.bin.index.json")
        if os.path.exists(index_file):
            print("🛠️ Detected sharded model, loading automatically...")
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
            return model.state_dict()

        # Load all .bin files manually
        params = {}
        for file in os.listdir(model_path):
            if file.endswith(".bin") and "pytorch_model" in file:
                shard_path = os.path.join(model_path, file)
                shard_data = torch.load(shard_path, map_location="cpu")
                params.update(shard_data)
        return params
    else:
        # Load single .bin file
        return torch.load(model_path, map_location="cpu")

def compare_model_parameters(base_params, finetuned_params):
    """Compare base vs. fine-tuned model parameters."""
    print("\n📊 Comparing model parameters...")
    print(f"- Base model has {len(base_params)} parameters")
    print(f"- Fine-tuned model has {len(finetuned_params)} parameters")

    missing_keys = set(base_params.keys()) - set(finetuned_params.keys())
    extra_keys = set(finetuned_params.keys()) - set(base_params.keys())

    if missing_keys:
        print(f"❌ {len(missing_keys)} missing parameters in fine-tuned model")
        print("Example missing keys:", list(missing_keys)[:5])
    if extra_keys:
        print(f"⚠️ {len(extra_keys)} extra parameters in fine-tuned model")
        print("Example extra keys:", list(extra_keys)[:5])

    return missing_keys, extra_keys

def check_tensor_dtype(model_params, model_name):
    """Check dtype of model tensors."""
    dtypes = set(v.dtype for v in model_params.values())
    print(f"\n🧬 Tensor dtypes in {model_name}: {dtypes}")

def check_deepspeed_sharding(model_path):
    """Check if DeepSpeed sharded the model across multiple files."""
    part_files = [f for f in os.listdir(model_path) if "pytorch_model" in f and ".bin" in f]
    
    if len(part_files) > 1:
        print(f"🛠️ Detected {len(part_files)} sharded model files:")
        for file in part_files:
            file_size = round(os.path.getsize(os.path.join(model_path, file)) / (1024**3), 2)
            print(f"- {file} ({file_size} GB)")
    else:
        print("✅ Model is not sharded.")

def check_deepspeed_config(ds_config_path):
    """Check DeepSpeed config for offloading settings."""
    if not os.path.exists(ds_config_path):
        print("\n⚠️ DeepSpeed config file not found. Skipping DeepSpeed checks.")
        return

    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    print("\n🔍 Checking DeepSpeed ZeRO-3 configuration:")
    if "zero_optimization" in ds_config:
        zero_config = ds_config["zero_optimization"]
        print(f"- ZeRO stage: {zero_config.get('stage', 'Unknown')}")
        print(f"- Offload Parameters: {zero_config.get('offload_param', 'Not specified')}")
        print(f"- Offload Optimizer: {zero_config.get('offload_optimizer', 'Not specified')}")
        print(f"- Pin Memory: {zero_config.get('offload_param_pin_memory', 'Not specified')}")
    else:
        print("❌ No ZeRO-3 configuration found.")

if __name__ == "__main__":
    print("🚀 Running Advanced Debugger Script")

    # Check checkpoint directory structure
    list_checkpoint_files(FINETUNED_MODEL_PATH)

    # Load base model
    print("\n🔍 Loading base model...")
    base_params = load_model_params(BASE_MODEL_PATH)
    print(f"✅ Base model loaded with {len(base_params)} tensors.")

    # Load fine-tuned model
    print("\n🔍 Loading fine-tuned model...")
    finetuned_params = load_model_params(FINETUNED_MODEL_PATH)
    print(f"✅ Fine-tuned model loaded with {len(finetuned_params)} tensors.")

    # Compare model parameters
    missing_keys, extra_keys = compare_model_parameters(base_params, finetuned_params)

    # Check tensor dtypes
    check_tensor_dtype(base_params, "Base Model")
    check_tensor_dtype(finetuned_params, "Fine-Tuned Model")

    # Check DeepSpeed sharding
    check_deepspeed_sharding(FINETUNED_MODEL_PATH)

    # Check DeepSpeed configuration
    check_deepspeed_config(DEEPSPEED_CONFIG_PATH)

    print("\n✅ Debugging complete!")
