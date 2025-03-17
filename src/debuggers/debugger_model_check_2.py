from transformers import AutoModelForCausalLM

BASE_MODEL_PATH = "/dss/dsshome1/02/ra95kix2/.cache/huggingface/models--llm-stacking--StackLLM_7B_300BToken/snapshots/1dffaf0e8b023be315beba18fd75cf1d63434af9/pytorch_model.bin"

FINETUNED_MODEL_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m2_prompt/checkpoint-3546"
SAVE_PATH = "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/stack_7b_m2_prompt_full"

# Update these paths before running
BASE_MODEL_PATH = "/dss/dsshome1/02/ra95kix2/.cache/huggingface/models--llm-stacking--StackLLM_7B_300BToken/snapshots/1dffaf0e8b023be315beba18fd75cf1d63434af9"
FINETUNED_MODEL_PATH = "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models/stack_7b_m2_prompt_full"

def check_lm_head(model, model_name):
    """Check if `lm_head.weight` exists in the model."""
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        print(f"✅ `{model_name}` has `lm_head.weight` with shape: {model.lm_head.weight.shape}")
    else:
        print(f"❌ `{model_name}` is missing `lm_head.weight`!")

if __name__ == "__main__":
    print("🚀 Checking `lm_head.weight` in models...")

    # Load base model
    print("\n🔍 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="cpu")
    check_lm_head(base_model, "Base Model")

    # Load fine-tuned model
    print("\n🔍 Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, device_map="cpu")
    check_lm_head(finetuned_model, "Fine-Tuned Model")

    print("\n✅ Check complete!")


