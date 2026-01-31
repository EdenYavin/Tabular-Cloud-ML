#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# =============================================================================
# T Network Training Job - Reconstruction-Only Experiment
# Trains Encoder + Decoder (T Network) without classification head
# =============================================================================

# --- Build flags from environment variables ---

# Triangulation Mode
if [ -n "$TRIANGULATION_MODE" ]; then
    TRIANG_MODE_FLAG="--triangulation-mode $TRIANGULATION_MODE"
else
    TRIANG_MODE_FLAG=""
fi

# Triangulation Embedding Model
if [ -n "$TRIANGULATION_EMBEDDING" ]; then
    TRIANG_EMB_FLAG="--triangulation-embedding-model $TRIANGULATION_EMBEDDING"
else
    TRIANG_EMB_FLAG=""
fi

# Triangulation Choosing Method
if [ -n "$CHOOSE" ]; then
    TRIANG_CHOOSE_FLAG="--triangulation-choosing $CHOOSE"
else
    TRIANG_CHOOSE_FLAG=""
fi

# Number of Triangulation Samples (Anchors)
if [ -n "$TRIANGULATIONS_NUMBER" ]; then
    TRIANG_SAMPLES_FLAG="--triangulation-samples $TRIANGULATIONS_NUMBER"
else
    TRIANG_SAMPLES_FLAG=""
fi

# Number of Epochs
if [ -n "$EPOCHS" ]; then
    EPOCHS_FLAG="--iim-epochs $EPOCHS"
else
    EPOCHS_FLAG="--iim-epochs 100"  # Default
fi

# Dataset Batch Size
if [ -n "$DATASET_BATCH_SIZE" ]; then
    BATCH_SIZE_FLAG="--dataset-batch-size $DATASET_BATCH_SIZE"
else
    BATCH_SIZE_FLAG=""
fi

# Use Cloud Models (optional)
if [ -n "$CLOUD_MODEL" ]; then
    CLOUD_FLAG="--use-cloud-models $CLOUD_MODEL"
else
    CLOUD_FLAG=""
fi

# Use Raw Features
if [ -n "$USE_RAW_FEATURES" ] && [ "$USE_RAW_FEATURES" = "1" ]; then
    RAW_FEATURES_FLAG="--use-raw-features"
else
    RAW_FEATURES_FLAG=""
fi

# Scaling Factor
if [ -n "$SCALING_FACTOR" ]; then
    SCALING_FLAG="--scaling-factor $SCALING_FACTOR"
else
    SCALING_FLAG=""
fi

# -----------------------------------------
# Echo configuration info
# -----------------------------------------

echo "==============================================="
echo "T Network Training Experiment"
echo "==============================================="
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Triangulation Mode: $TRIANGULATION_MODE"
echo "  Triangulation Embedding: $TRIANGULATION_EMBEDDING"
echo "  Triangulation Choosing: $CHOOSE"
echo "  Number of Anchors: $TRIANGULATIONS_NUMBER"
echo "  Epochs: $EPOCHS"
echo "  Cloud Model: ${CLOUD_MODEL:-None}"
echo "  Use Raw Features: ${USE_RAW_FEATURES:-0}"
echo "  Scaling Factor: ${SCALING_FACTOR:-5}"
echo "==============================================="
echo ""

# -----------------------------------------
# Load environment
# -----------------------------------------

module load anaconda
source activate tabml
echo "Currently active conda env: $CONDA_DEFAULT_ENV"
which python
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# -----------------------------------------
# Run T Network Training
# -----------------------------------------

echo "Starting T Network training..."
python main.py \
    --experiment-to-run t_network_training \
    --encoder-rotating-key \
    --datasets "$DATASET_NAME" \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $EPOCHS_FLAG \
    $BATCH_SIZE_FLAG \
    $CLOUD_FLAG \
    $RAW_FEATURES_FLAG \
    $SCALING_FLAG

echo ""
echo "T Network training completed!"
echo "Results saved to output/report.csv"
