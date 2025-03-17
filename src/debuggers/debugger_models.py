import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Correct paths
CHECKPOINT_DIR = "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/llm_3b_m1/checkpoint-1547"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")
CONFIG_FILE = os.path.join(CHECKPOINT_DIR, "config.json")

def check_file_exists(file_path):
    """Check if a specific file exists."""
    if os.path.isfile(file_path):
        print(f"✅ File exists: {file_path}")
        return True
    else:
        print(f"❌ File does NOT exist: {file_path}")
        return False

def check_checkpoint_integrity(checkpoint_path):
    """Check if the checkpoint file is valid."""
    if not check_file_exists(checkpoint_path):
        return

    try:
        print(f"🔍 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("✅ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")

def check_model_loading(checkpoint_dir):
    """Attempt to load the model using Hugging Face's `from_pretrained`."""
    if not check_file_exists(CONFIG_FILE):
        return

    try:
        print(f"🔍 Loading model from: {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

if __name__ == "__main__":
    check_checkpoint_integrity(CHECKPOINT_FILE)
    check_model_loading(CHECKPOINT_DIR)
