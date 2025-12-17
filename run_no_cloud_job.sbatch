#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

if [ -n "$USE_RAW_FEATURES" ]; then
    RAW_FEATURES_FLAG="--use-raw-features"
else
    RAW_FEATURES_FLAG=""
fi

# --- NEW: Check for Triangulation Mode ---
if [ -n "$TRIANGULATION_MODE" ]; then
    TRIANG_MODE_FLAG="--triangulation-mode $TRIANGULATION_MODE"
else
    TRIANG_MODE_FLAG=""
fi

# --- NEW: Check for Embedding Model ---
if [ -n "$TRIANGULATION_EMBEDDING" ]; then
    TRIANG_EMB_FLAG="--triangulation-embedding-model $TRIANGULATION_EMBEDDING"
else
    TRIANG_EMB_FLAG=""
fi
# --------------------------------------

# Echo info
# ... existing echoes
echo "Triangulation Embedding: $TRIANGULATION_EMBEDDING"
# -----------------------------------------

# Echo info
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Running dataset: $DATASET_NAME"
echo "Running experiment": $EXPERIMENT_NAME
echo "Number of triangulation $TRIANGULATIONS_NUMBER"
echo "Use raw features flag: $RAW_FEATURES_FLAG"
echo "Triangulation Mode: $TRIANGULATION_MODE"

# Load environment
module load anaconda
source activate tabml
echo "Currently active conda env: $CONDA_DEFAULT_ENV"
which python

export TF_XLA_FLAGS=--tf_xla_auto_jit=0
# Run the Python script
python main.py \
    --experiment-to-run "$EXPERIMENT_NAME" \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG  \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$NUMBER_OF_DATASETS" \
    --iim-epochs 100 \
    --triangulation-choosing $CHOOSE \
    --dataset-use-cache \
    --triangulation-samples "$TRIANGULATIONS_NUMBER"