#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# ==========================================
# K-FOLD CV PIPELINE (With Cloud)
# Step 1: Dataset Creation (with K folds)
# Step 2: K-Fold Training (each fold trained once)
# ==========================================

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



# Use trained key encoder for feature extraction
if [ -n "$USE_KEY_ENCODER" ] && [ "$USE_KEY_ENCODER" = "1" ]; then
    KEY_ENCODER_FLAG="--use-key-encoder"
else
    KEY_ENCODER_FLAG=""
fi

# Use trained key encoder for feature extraction
if [ -n "$DEEPSET" ] && [ "$DEEPSET" = "1" ]; then
    DEEPSET_FLAG="--deepset"
else
    DEEPSET_FLAG=""
fi

# Freeze T-Network if requested (auto-determines path from config)
if [ -n "$FREEZE_T_NETWORK" ] && [ "$FREEZE_T_NETWORK" = "1" ]; then
    FREEZE_T_FLAG="--freeze-t-network"
else
    FREEZE_T_FLAG=""
fi

# Optional: Manual T-Network path override
if [ -n "$PRETRAINED_T_NETWORK_PATH" ]; then
    T_NETWORK_PATH_FLAG="--pretrained-t-network $PRETRAINED_T_NETWORK_PATH"
else
    T_NETWORK_PATH_FLAG=""
fi

# Echo Configuration
echo "SLURM_JOBID=$SLURM_JOBID"
echo "Dataset: $DATASET_NAME"
echo "Triangulations: $TRIANGULATIONS_NUMBER | Method: $CHOOSE"
echo "Mode: $TRIANGULATION_MODE | Embedding: $TRIANGULATION_EMBEDDING"

echo "N_P (Pred Vectors): $N_P | K-Folds: $K"
echo "Using Cloud Model: $CLOUD_MODEL"
echo "Use Key Encoder: $USE_KEY_ENCODER"
echo "Use Deepset: $DEEPSET"
echo "Freeze T-Network: $FREEZE_T_NETWORK"
echo "T-Network Path: ${PRETRAINED_T_NETWORK_PATH:-auto}"
module load anaconda
source activate tabml
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# ==========================================
# 2. STEP 1: DATASET CREATION
# ==========================================
echo "----------------------------------------------------------------"
echo "STARTING STEP 1: DATASET CREATION (K-Fold)"
echo "----------------------------------------------------------------"

python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $KEY_ENCODER_FLAG \
    $DEEPSET_FLAG \
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$N_P" \
    --use-cloud-models $CLOUD_MODEL \
    --triangulation-choosing "$CHOOSE" \
    --triangulation-samples "$TRIANGULATIONS_NUMBER" \
    --k-training "$K" \
    --dataset-use-cache

# Check exit code to ensure creation succeeded
if [ $? -ne 0 ]; then
    echo "❌ Dataset Creation Failed! Exiting."
    exit 1
fi

# ==========================================
# 3. STEP 2: K-FOLD TRAINING
# ==========================================
echo "----------------------------------------------------------------"
echo "STARTING STEP 2: K-FOLD TRAINING (each fold trained once)"
echo "----------------------------------------------------------------"

python main.py \
    --experiment-to-run k_fold_training \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $KEY_ENCODER_FLAG \
    $DEEPSET_FLAG \
    $FREEZE_T_FLAG \
    $T_NETWORK_PATH_FLAG \
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$N_P" \
    --use-cloud-models $CLOUD_MODEL \
    --iim-name $MODELS \
    --iim-epochs 100 \
    --k-training "$K" \
    --triangulation-choosing "$CHOOSE" \
    --triangulation-samples "$TRIANGULATIONS_NUMBER" \
    --dataset-use-cache
