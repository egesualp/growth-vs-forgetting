import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os

checkpoint_path = "./test_checkpoints/checkpoint-125"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ The checkpoint path '{checkpoint_path}' does not exist. Please check the path.")


# ✅ Set the checkpoint path (update if needed) 
# ✅ Load the tokenizer and model from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

# ✅ Ensure the model is in eval mode
model.eval()

# ✅ Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ Sample input text
input_text = "Translate the following sentence to French: The weather is nice today."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# ✅ Generate output
with torch.no_grad():
    output_tokens = model.generate(**inputs, max_length=50)

# ✅ Decode and print output
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(f"📝 Generated Output: {output_text}")
