"""
RetentionAI project configuration.
"""

from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Main directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


# Dataset location
DATA_FILE = DATA_DIR / "customer_churn.csv"


# ML configuration
TARGET_COLUMN = "churn"

RANDOM_STATE = 42
TEST_SIZE = 0.20