import os
import json
import pandas as pd

import os
import json
import pandas as pd

def gather_test_metrics(model_roots, output_csv="all_test_metrics.csv"):
    """
    Iterates over multiple model root directories, finds subdirectories named 'test*',
    loads 'metrics.json' and 'metrics_2.json' (if it exists) from each test directory,
    and aggregates all data into a single CSV with a new column indicating the source JSON.
    """
    rows = []

    # model_roots is a list of folder paths, e.g.:
    # ["/path/to/models1", "/path/to/models2"]
    for root_dir in model_roots:
        if not os.path.isdir(root_dir):
            print(f"Warning: {root_dir} is not a directory or doesn't exist.")
            continue
        
        # Iterate over subdirectories in the current root_dir
        for model_dir in os.listdir(root_dir):
            full_model_dir = os.path.join(root_dir, model_dir)
            if not os.path.isdir(full_model_dir):
                continue  # skip if it's not a directory

            # Look for subdirectories that start with "test"
            for subdir in os.listdir(full_model_dir):
                if subdir.startswith("test"):
                    test_dir_path = os.path.join(full_model_dir, subdir)
                    
                    # List of JSON files to check
                    json_files = [("metrics.json", "metrics"), ("metrics_2.json", "metrics_2")]
                    for file_name, json_name in json_files:
                        file_path = os.path.join(test_dir_path, file_name)
                        if os.path.isfile(file_path):
                            # Load the JSON data
                            with open(file_path, "r") as f:
                                metrics_data = json.load(f)

                            # Prepare a row of data
                            row = {
                                "root": root_dir,       # which root directory
                                "model_folder": model_dir,
                                "test_dir": subdir,
                                "json_name": json_name  # indicate which JSON file was loaded
                            }
                            # Add all key/value pairs from the loaded JSON file
                            for key, value in metrics_data.items():
                                row[key] = value

                            rows.append(row)

    # Convert the list of rows into a DataFrame and save it as CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated results to {output_csv}")

if __name__ == "__main__":
    # Provide both model paths here
    model_paths = [
        "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models",
        "/dss/dsshome1/02/ra95kix2/seminar_fma/growth-vs-forgetting/src/models"
    ]
    gather_test_metrics(model_paths, output_csv="combined_metrics_new.csv")