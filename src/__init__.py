from pathlib import Path

# Define the base directory for the `src` folder
DIR = Path(__file__).parent

# Define a path to the `configs` folder
CONFIG_DIR = DIR.parent / "experiments" / "configs"

# Define parent dir
PARENT_DIR = DIR.parent