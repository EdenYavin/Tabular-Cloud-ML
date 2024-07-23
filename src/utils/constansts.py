import os
import sys

WINDOWS_OS_STR = "nt"
IS_WINDOWS_OS = (os.name == WINDOWS_OS_STR)

PROJECT_DIR = os.environ['PROJECT_DIR'] if 'PROJECT_DIR' in os.environ else os.path.dirname(
    sys.modules['__main__'].__file__)

# -----------------
# FILE PATHS
# -----------------
OUTPUT_DIR = 'output'
OUTPUT_DIR_PATH = os.path.join(PROJECT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
GRAPH_OUTPUT_FILE_NAME = "graph.pkl"
GRAPH_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, GRAPH_OUTPUT_FILE_NAME)

INPUT_DIR = "input"
INPUT_PATH = os.path.join(PROJECT_DIR, INPUT_DIR)
os.makedirs(INPUT_PATH, exist_ok=True)
CONFIG_FILE_NAME = "config.ini"
CONFIG_PATH = os.path.join(INPUT_PATH, CONFIG_FILE_NAME)

STORE_DIR = "store"
STORE_PATH = os.path.join(PROJECT_DIR, STORE_DIR)
os.makedirs(STORE_PATH, exist_ok=True)

MODELS_DIR = "models"
TEMPLATES_PATH = os.path.join(STORE_PATH, MODELS_DIR)
os.makedirs(TEMPLATES_PATH, exist_ok=True)

