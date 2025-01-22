import json
from datasets import Dataset 

file_path = "../../data/preprocessed/simp_wiki_auto.json"

from datasets import DatasetDict, Dataset


# Load each split separately into a DatasetDict
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset_dict = DatasetDict({
    split: Dataset.from_dict(examples)
    for split, examples in data.items()
})

print(dataset_dict)