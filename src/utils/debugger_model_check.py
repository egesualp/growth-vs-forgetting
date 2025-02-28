from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Define model directory (update with your actual path)
model_path = "../models/stack_3b_m1/checkpoint-1547" # Example: "./models/llama-finetuned"
model_path_old = "llm-stacking/StackLLM_3B_300BToken"

# Load the fine-tuned model
model_new = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model_old = AutoModelForCausalLM.from_pretrained(model_path_old, torch_dtype=torch.float16, device_map="auto")


# Load tokenizer

tokenizer_new = AutoTokenizer.from_pretrained(model_path)
tokenizer_old = AutoTokenizer.from_pretrained(model_path_old)


# Create inference pipeline
generator_new = pipeline("text-generation", model=model_new, tokenizer=tokenizer_new, device=0)  # Use device=0 for GPU
generator_old = pipeline("text-generation", model=model_old, tokenizer=tokenizer_old, device=0)  # Use device=0 for GPU

print('Checking new model...')
# Run inference with a sample prompt
prompt_1 = '''
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
        The Mercedes-Benz Axor was a truck manufactured by Mercedes-Benz designed to fill the gap between the premium Actros tractors and the mostly rigid Atego trucks and was targeted at fleet customers .\n\n\n===\n\nGiven this text, write it with, simpler words:
            '''

prompt_2 = '''
            The Mercedes-Benz Axor was a truck manufactured by Mercedes-Benz designed to fill the gap between the premium Actros tractors and the mostly rigid Atego trucks and was targeted at fleet customers .\n\n\n===\n\nGiven this text, write it with, simpler words:
           '''

prompt_1 = '''
            Make this text simpler: "A romantic friendship , passionate friendship , or affectionate friendship is a very close but typically non-sexual relationship between friends , often involving a degree of physical closeness beyond that which is common in the contemporary Western societies .\n"
            '''

prompt_2 = "It is incapable of compressing files , although it is able to extract compressed ones .\n\n\n===\n\nGiven the above text, reformulate it so it is easier to read:"
output_1 = generator_new(prompt_1, max_length=256, temperature=0.3, top_k=50)
output_2 = generator_new(prompt_2, max_length=256, temperature=0.3, top_k=50)

# Print results
print(f"Prompt with SYS: {prompt_1}")
print("Generated Text:", output_1[0]['generated_text'])

print(f"Prompt without SYS: {prompt_2}")
print("Generated Text:", output_2[0]['generated_text'])
#print(tokenizer.chat_template)  # Prints system prompt if it exists
print('=======================================')
print('Checking old...')

output_1 = generator_old(prompt_1, max_length=256, temperature=0.3, top_k=50)
output_2 = generator_old(prompt_2, max_length=256, temperature=0.3, top_k=50)
# Print results
print(f"Prompt with SYS: {prompt_1}")
print("Generated Text:", output_1[0]['generated_text'])

print(f"Prompt without SYS: {prompt_2}")
print("Generated Text:", output_2[0]['generated_text'])