#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# ==========================================
# 1. SETUP & LOGGING
# ==========================================

if [ -n "$USE_RAW_FEATURES" ] && [ "$USE_RAW_FEATURES" = "1" ]; then
    RAW_FEATURES_FLAG="--use-raw-features"
elif [ "$USE_RAW_FEATURES" = "0" ]; then
    RAW_FEATURES_FLAG=""
else
    RAW_FEATURES_FLAG=""
fi

if [ -n "$TRIANGULATION_MODE" ]; then
    TRIANG_MODE_FLAG="--triangulation-mode $TRIANGULATION_MODE"
else
    TRIANG_MODE_FLAG=""
fi

if [ -n "$TRIANGULATION_EMBEDDING" ]; then
    TRIANG_EMB_FLAG="--triangulation-embedding-model $TRIANGULATION_EMBEDDING"
else
    TRIANG_EMB_FLAG=""
fi

if [ -n "$CALIB_VECTOR" ] && [ "$CALIB_VECTOR" = "1" ]; then
    USE_CALIB="--use-calibration-vector"
else
    USE_CALIB=""
fi

# Echo Configuration
echo "SLURM_JOBID=$SLURM_JOBID"
echo "Dataset: $DATASET_NAME"
echo "Triangulations: $TRIANGULATIONS_NUMBER | Method: $CHOOSE"
echo "Mode: $TRIANGULATION_MODE | Embedding: $TRIANGULATION_EMBEDDING"
echo "Use Calibration Vector: $CALIB_VECTOR"
echo "N_P (Pred Vectors): $N_P | K-Folds: $K"
echo "Using Cloud Model: $CLOUD_MODEL"

module load anaconda
source activate tabml
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# ==========================================
# 2. STEP 1: DATASET CREATION
# ==========================================
echo "----------------------------------------------------------------"
echo "STARTING STEP 1: DATASET CREATION"
echo "----------------------------------------------------------------"

python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $CA
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$N_P" \
    --use-cloud-models $CLOUD_MODEL \
    --triangulation-choosing "$CHOOSE" \
    --triangulation-samples "$TRIANGULATIONS_NUMBER" \
    --dataset-use-cache

# Check exit code to ensure creation succeeded
if [ $? -ne 0 ]; then
    echo "❌ Dataset Creation Failed! Exiting."
    exit 1
fi

# ==========================================
# 3. STEP 2: MODEL TRAINING
# ==========================================
echo "----------------------------------------------------------------"
echo "STARTING STEP 2: MODEL TRAINING"
echo "----------------------------------------------------------------"

python main.py \
    --experiment-to-run model_training \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $USE_CALIB \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$N_P" \
    --use-cloud-models $CLOUD_MODEL \
    --iim-name $MODELS \
    --iim-epochs 100 \
    --k-training "$K" \
    --triangulation-choosing "$CHOOSE" \
    --triangulation-samples "$TRIANGULATIONS_NUMBER" \
    --dataset-use-cache