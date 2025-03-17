import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "../models/stack_3b_m1/checkpoint-1547"
model_name = "llm-stacking/StackLLM_3B_300BToken"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half().cuda()
model.eval()

# Test prompts
test_prompts = [
    "Write a summary of this text: 'Einstein developed the theory of relativity.'",
    "Generate a simple definition for 'Photosynthesis'.",
    "Tell me a short fun fact about space.",
    "Explain why the sky is blue in simple words.",
    "Reformulate this text in simpler words: 'The government implemented new policies to improve the economy.'",
]

# Run inference
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id, num_beams=3, top_k=50, top_p=0.9, temperature=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("\n🔍 **Testing for Hallucination Issues:**\n")
for prompt in test_prompts:
    response = generate_response(prompt)
    print(f"📝 **Prompt:** {prompt}")
    print(f"🔹 **Response:** {response}\n")
    print("=" * 80)

