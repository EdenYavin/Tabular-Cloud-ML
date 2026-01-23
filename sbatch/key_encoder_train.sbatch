#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=30G

# ==========================================
# KEY ENCODER TRAINING SBATCH SCRIPT
# ==========================================

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Default values (override with environment variables)
NUM_KEYS=${NUM_KEYS:-500}
NUM_CAL_PAIRS=${NUM_CAL_PAIRS:-50}
EMBEDDING_DIM=${EMBEDDING_DIM:-64}
OUTPUT_EMB_DIM=${OUTPUT_EMB_DIM:-256}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
OUTPUT_DIR=${OUTPUT_DIR:-}

# Echo Configuration
echo "=========================================="
echo "KEY ENCODER PROOF-OF-CONCEPT TRAINING"
echo "=========================================="
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Number of Keys: $NUM_KEYS"
echo "Calibration Pairs per Key: $NUM_CAL_PAIRS"
echo "Embedding Dimension: $EMBEDDING_DIM"
echo "Output Embedding Dimension: $OUTPUT_EMB_DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output Directory: $OUTPUT_DIR"
else
    echo "Output Directory: (default - output/meta_learning/key_encoder_poc)"
fi
echo "=========================================="

# ==========================================
# 2. ENVIRONMENT SETUP
# ==========================================

module load anaconda
source activate tabml
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# Set PYTHONPATH to include project root
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "PYTHONPATH: $PYTHONPATH"
which python
echo ""

# ==========================================
# 3. TRAIN KEY ENCODER
# ==========================================

echo "Starting Key Encoder Training..."
echo "--------"

# Change to project root directory
cd "${SLURM_SUBMIT_DIR}"

if [ -n "$OUTPUT_DIR" ]; then
    python -m src.meta_learning.train_key_encoder \
        --num-keys $NUM_KEYS \
        --num-calibration-pairs $NUM_CAL_PAIRS \
        --embedding-dim $EMBEDDING_DIM \
        --output-embedding-dim $OUTPUT_EMB_DIM \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --output-dir $OUTPUT_DIR
else
    python -m src.meta_learning.train_key_encoder \
        --num-keys $NUM_KEYS \
        --num-calibration-pairs $NUM_CAL_PAIRS \
        --embedding-dim $EMBEDDING_DIM \
        --output-embedding-dim $OUTPUT_EMB_DIM \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE
fi

# Check exit code
if [ $? -ne 0 ]; then
    echo "❌ Key Encoder Training Failed!"
    exit 1
fi

echo "--------"
echo "✅ Key Encoder Training Completed Successfully!"