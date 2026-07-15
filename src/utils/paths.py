import os

# Project root resolution
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KB_DIR = os.path.join(DATA_DIR, "kb")
DEV_DIR = os.path.join(DATA_DIR, "dev")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Knowledge Base and Context Paths
EXCEL_PATH = os.path.join(KB_DIR, "ICD10.xlsx")
DB_PATH = os.path.join(KB_DIR, "metadata.db")
JSON_PATH = os.path.join(KB_DIR, "icd10_dictionary.json")
ICD10_TXT_PATH = os.path.join(KB_DIR, "icd10_context.txt")
RXNORM_TXT_PATH = os.path.join(KB_DIR, "rxnorm_context.txt")
