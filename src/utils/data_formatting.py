import json
import logging
from datasets import Dataset, DatasetDict
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_and_merge_datasets(folder_path, output_path, eval_dataset_size=0.1):
    """
    Processes training and test data from a folder, splits training data into train and eval subsets,
    and merges them into a single JSON file. Adds an index column to track rows in the raw dataset.

    Args:
        folder_path (str): Path to the folder containing training and test dataset JSON files.
        output_path (str): Path to save the merged dataset JSON file.
        eval_dataset_size (float): Ratio of training data to use for evaluation (e.g., 0.1 for 10%).
    """
    train_data = []
    test_data = []

    # Iterate over files in the folder
    logger.info(f"Scanning folder: {folder_path}...")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith(".json"):
            continue

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                
                # Handle top-level dictionary
                if isinstance(data, dict):
                    logger.info(f"Top-level dictionary found in {file_path}. Extracting values...")
                    data = list(data.values())

                if not isinstance(data, list):
                    raise ValueError(f"Expected a list of dictionaries in {file_path}, but got {type(data).__name__}")

                if "train" in file_name:
                    logger.info(f"Loading training data from {file_path}...")
                    train_data.extend(data)
                elif "test" in file_name:
                    logger.info(f"Loading test data from {file_path}...")
                    test_data.extend(data)
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    if not train_data:
        raise ValueError("No training data found in the folder.")
    if not test_data:
        raise ValueError("No test data found in the folder.")

    # Verify and format training data
    formatted_train_data = []
    for idx, item in enumerate(train_data):
        if not isinstance(item, dict) or "src" not in item or "tgt" not in item:
            logger.error(f"Invalid data format in training data at index {idx}: {item}")
            raise ValueError("Training data must be a list of dictionaries with 'src' and 'tgt' keys.")
        formatted_train_data.append({"input": item["src"], "output": item["tgt"], "index": idx, "src_info": item["src_info"]})

    # Verify and format test data
    formatted_test_data = []
    for idx, item in enumerate(test_data):
        if not isinstance(item, dict) or "src" not in item or "tgt" not in item:
            logger.error(f"Invalid data format in test data at index {idx}: {item}")
            raise ValueError("Test data must be a list of dictionaries with 'src' and 'tgt' keys.")
        formatted_test_data.append({"input": item["src"], "output": item["tgt"], "index": idx, 'src_info': item["src_info"]})

    # Create a Hugging Face dataset from the training data
    train_dataset = Dataset.from_list(formatted_train_data)

    # Split the training dataset into train and eval subsets
    logger.info("Splitting training data into train and eval subsets...")
    train_eval_split = train_dataset.train_test_split(
        test_size=eval_dataset_size, shuffle=True, seed=42
    )

    # Create the test dataset
    test_dataset = Dataset.from_list(formatted_test_data)

    # Combine train, eval, and test datasets into a DatasetDict
    combined_dataset = DatasetDict({
        "train": train_eval_split["train"],
        "eval": train_eval_split["test"],
        "test": test_dataset
    })

    # Save the merged dataset to a JSON file
    logger.info(f"Saving combined dataset to {output_path}...")
    merged_data = {
        "train": combined_dataset["train"].to_list(),
        "eval": combined_dataset["eval"].to_list(),
        "test": combined_dataset["test"].to_list()
    }
    with open(output_path, "w") as file:
        json.dump(merged_data, file, indent=4)

    logger.info("Dataset processing and merging completed!")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and merge datasets for training.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing dataset JSON files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged dataset JSON file.")
    parser.add_argument("--eval_dataset_size", type=float, default=0.1, help="Ratio of training data to use for evaluation.")

    args = parser.parse_args()

    process_and_merge_datasets(
        folder_path=''.join(['../../data/raw/', args.folder_path]),
        output_path=''.join(['../../data/preprocessed/', args.output_path]),
        eval_dataset_size=args.eval_dataset_size
    )
