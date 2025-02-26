import os
import glob
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# ✅ Load a small dataset (subset of wikitext)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Small dataset

# ✅ Load a small encoder-decoder model (T5-small)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ✅ Tokenize dataset with labels for loss computation
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # Assign `input_ids` and `labels`
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)

# ✅ Define Training Arguments
output_dir = "./test_checkpoints"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    predict_with_generate=True,  # Important for seq2seq models
    fp16=False,
    report_to="none",
)

# ✅ Custom Seq2SeqTrainer to Exclude Optimizer
class NoOptimizerSeq2SeqTrainer(Seq2SeqTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to prevent saving optimizer state."""
        checkpoint_folder = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")

        # Save only model weights, tokenizer, and config (skip optimizer)
        self.model.save_pretrained(checkpoint_folder)
        self.tokenizer.save_pretrained(checkpoint_folder)

        # Save training state (without optimizer)
        self.state.save_to_json(os.path.join(checkpoint_folder, "trainer_state.json"))

        print(f"✅ Checkpoint saved without optimizer at {checkpoint_folder}")

# ✅ Initialize Trainer
trainer = NoOptimizerSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# ✅ Run training
trainer.train()

# ✅ Verify that optimizer files are NOT saved
print("\n✅ Checking if optimizer files exist:")
optimizer_files = glob.glob(f"{output_dir}/checkpoint-*/optimizer.pt")
if not optimizer_files:
    print("🎉 Optimizer files are NOT saved! Everything is working correctly.")
else:
    print(f"⚠️ Found optimizer files: {optimizer_files}. The fix did not work.")
