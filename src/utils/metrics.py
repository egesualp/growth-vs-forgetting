import json
import os
from evaluate import load
import numpy as np

#from bert_score import score as bert_score

# Load evaluation metrics
sari = load("sari")
bleu = load("bleu")
rouge = load("rouge")
bert_score = load("bertscore")

# 📌 Set the path to the predictions file
PREDICTIONS_FILE = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m2_prompt/test_2364/predictions.jsonl"  # Update with actual path

# Check if the file exists
if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_FILE}")

# Load predictions and references
predictions = []
predictions_wo_input = []
references = []
sources = []

with open(PREDICTIONS_FILE, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        predictions.append(entry["prediction_with_input"].strip())  # Full prediction
        predictions_wo_input.append(entry["prediction"].strip())  # Cleaned prediction
        if "simple_sentence" in entry['src_info']:
            references.append([entry['src_info']["simple_sentence"]])
        elif "output" in entry:
            references.append([entry["output"]])
        if "normal_sentence" in entry['src_info']:
            sources.append(entry['src_info']["normal_sentence"])

# 📌 Compute BLEU
bleu_score = bleu.compute(predictions=predictions_wo_input, references=references)
print(f"BLEU Score: {bleu_score['bleu']:.4f}")

# 📌 Compute ROUGE
rouge_score = rouge.compute(predictions=predictions_wo_input, references=references)
print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")

# 📌 Compute BERTScore
bscore = bert_score.compute(predictions=predictions_wo_input, references=references, lang='en')

print(f"BScore-F1: {np.mean(bscore['f1']):.4f}")  # ✅ Works correctly
print(f"BScore-P: {np.mean(bscore['precision']):.4f}")
print(f"BScore-R: {np.mean(bscore['recall']):.4f}")

# 📌 Compute SARI (if simplification task)
if sources:
    sari_score = sari.compute(sources=sources, predictions=predictions_wo_input, references=references)
    print(f"SARI Score: {sari_score['sari']:.4f}")

# 📌 Save results to a JSON file
metrics = {
    "bleu": bleu_score["bleu"],
    "rouge1": rouge_score["rouge1"],
    "rouge2": rouge_score["rouge2"],
    "rougeL": rouge_score["rougeL"],
    "bertscore_f1": np.mean(bscore['f1']),
    "bertscore_P": np.mean(bscore['precision']),
    "bertscore_R": np.mean(bscore['recall'])
}

if sources:
    metrics["sari"] = sari_score["sari"]

with open("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m2_prompt/test_2364/metrics_2.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation metrics saved to /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stack_7b_m2_prompt/test_2364/metrics_2.json")
