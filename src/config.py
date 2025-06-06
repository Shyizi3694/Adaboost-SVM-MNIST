# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# define the directories and file names for data
DATA_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# define the directories and file names for source code
SRC_DIR = PROJECT_ROOT / "src"
DATA_PREPROCESS_DIR = SRC_DIR / "data_preprocess"
MODELS_DIR = SRC_DIR / "models"



# define the directories and file names for results
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_TASK1_DIR = PLOTS_DIR / "task1"
PLOTS_TASK2_DIR = PLOTS_DIR / "task2"
METRICS_DIR = RESULTS_DIR / "metrics"
TRAINED_MODEL_DIR = RESULTS_DIR / "trained_models"


# define the directories and file names for utils
UTILS_DIR = PROJECT_ROOT/"utils"


# define hyperparameters for models

RANDOM_STATE = 114514