import os
import sys
import enum

class EMBEDDING_TYPES(str, enum.Enum):
    STRING = "w2v_embedding"
    IMAGE = "image_embedding"
    DIGIT_TO_IMAGE_TO_EMBEDDING = "numerical_table_embedding"
    DNN = "dnn_embedding"


class ENCODERS_TYPES(str, enum.Enum):
    tabular_dc = "tabular_dc"
    DENSE = "dense"
    EFFICIENT = "efficientnet"
    DCONV = "dc"

class IIM_MODELS(str, enum.Enum):
    NEURAL_NET = "neural_network"
    XGBOOST = "xgboost"


class CLOUD_MODELS(str, enum.Enum):
    ENSEMBLE_OF_TABULAR_MODELS = "ensemble"
    NEURAL_NET_DENSE_LAYERS = "dense"
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"
    VGG16 = "vgg16"

class DATASETS(str, enum.Enum):
    HELOC = "heloc"
    ADULT = "adult"
    STUDENTS_DROPOUT = "students_dropout"
    BANK_MARKETING = "bank_marketing"
    GESTURE_PHASE = "gesture_phase"


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

PROMPTS_DIR = 'prompts'
PROMPTS_DIR_PATH = os.path.join(PROJECT_DIR, PROMPTS_DIR)
os.makedirs(PROMPTS_DIR, exist_ok=True)

INPUT_DIR = "input"
INPUT_PATH = os.path.join(PROJECT_DIR, INPUT_DIR)
os.makedirs(INPUT_PATH, exist_ok=True)
CONFIG_FILE_NAME = "config.yaml"
CONFIG_PATH = os.path.join(INPUT_PATH, CONFIG_FILE_NAME)

CANVAS_FILE = "canvas.JPEG"
CANVAS_PATH = os.path.join(INPUT_PATH, CANVAS_FILE)

STORE_DIR = "store"
STORE_PATH = os.path.join(PROJECT_DIR, STORE_DIR)
os.makedirs(STORE_PATH, exist_ok=True)

MODELS_DIR = "models"
MODELS_PATH = os.path.join(STORE_PATH, MODELS_DIR)
os.makedirs(MODELS_PATH, exist_ok=True)

DATASETS_DIR = "data"
DATASETS_PATH = os.path.join(PROJECT_DIR, DATASETS_DIR)

REPORT_NAME = 'report.csv'
REPORT_PATH = os.path.join(OUTPUT_DIR_PATH, REPORT_NAME)

DATA_CACHE_PATH = os.path.join(STORE_PATH, "dataset")
os.makedirs(DATA_CACHE_PATH, exist_ok=True)

# ---------------------
# CLOUD MODELS SECTIONS
# ---------------------
CONFIG_CLOUD_MODEL_SECTION = "CLOUD"
CONFIG_CLOUD_MODELS_TOKEN = "models"
CONFIG_CLOUD_MODELS_PATH_TOKEN = "path"
LLM_CACHE_DIR = "llm_cache"
LLM_CACHE_PATH = os.path.join(STORE_PATH, LLM_CACHE_DIR)
os.makedirs(LLM_CACHE_PATH, exist_ok=True)
LLM_PRED_VECTOR_TYPE_CONFIG_TOKEN = 'logits'
LLM_TOP_K_CONFIG_TOKEN = "top_k"

# ---------------------
# INTERNAL MODEL SECTIONS
# ---------------------
CONFIG_INN_SECTION = "IIM"

# ---------------------
# DATASET SECTIONS
# ---------------------
CONFIG_DATASET_SECTION = "DATASET"
CONFIG_DATASET_NAME_TOKEN = "names"
CONFIG_DATASET_ONEHOT_TOKEN = "one_hot"
CONFIG_DATASET_SHUFFLE_TOKEN = "shuffle"
CONFIG_DATASET_SPLIT_RATIO_TOKEN = "ratio"
CONFIG_DATASET_FORCE_CREATION_TOKEN = "force"
DB_TRAIN_INDEX_TOKEN = "train_index"
DB_IMM_TRAIN_INDEX_TOKEN = "imm_train_index"
DB_TEST_INDEX_TOKEN = "test_index"
DB_EMBEDDING_TOKEN = "embedding"
DB_RAW_FEATURES_TOKEN = "X"
DB_LABEL_TOKEN = "y"
XGBOOST_BASELINE = "xgboost"
NEURAL_NET_BASELINE = "neural_network"

# ---------------------
# INTERNAL INFERENCE MODEL  SECTIONS
# ---------------------
CONFIG_IIM_SECTION = "IIM"
CONFIG_IMM_NAME_TOKEN = "name"

# ---------------------
# ENCODER SECTIONS
# ---------------------
CONFIG_ENCRYPTOR_SECTION = "ENCRYPTOR"
GPU_MODELS = ['dc', 'resnet', 'efficientnet']
CONFIG_ENCRYPTOR_NAME_TOKEN = "name"

# ---------------------
# EXPERIMENT SECTIONS
# ---------------------
CONFIG_EXPERIMENT_SECTION = "EXPERIMENT"
K_FOLDS_TOKEN = "k_fold"


# ---------------------
# PROMPTS SECTIONS
# ---------------------
SYSTEM_PREDICTION_PROMPT = os.path.join(PROMPTS_DIR_PATH, "system_prediction_prompt.txt")
USER_PREDICTION_PROMPT = os.path.join(PROMPTS_DIR_PATH, "user_prediction_prompt.txt")