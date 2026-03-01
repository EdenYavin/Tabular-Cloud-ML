#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# =============================================================================
# OTP Experiment
# Creates an OTP-masked dataset (raw sparse-AE vector ⊕ random OTP → cloud)
# and trains an IIM on the cloud responses.
# =============================================================================

# --- Build flags from environment variables ---

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

if [ -n "$CHOOSE" ]; then
    TRIANG_CHOOSE_FLAG="--triangulation-choosing $CHOOSE"
else
    TRIANG_CHOOSE_FLAG=""
fi

if [ -n "$TRIANGULATIONS_NUMBER" ]; then
    TRIANG_SAMPLES_FLAG="--triangulation-samples $TRIANGULATIONS_NUMBER"
else
    TRIANG_SAMPLES_FLAG=""
fi

if [ -n "$N_P" ]; then
    N_P_FLAG="--number-of-prediction-vector $N_P"
else
    N_P_FLAG="--number-of-prediction-vector 1"
fi

if [ -n "$CLOUD_MODEL" ]; then
    CLOUD_FLAG="--use-cloud-models $CLOUD_MODEL"
else
    CLOUD_FLAG=""
fi

if [ -n "$USE_RAW_FEATURES" ] && [ "$USE_RAW_FEATURES" = "1" ]; then
    RAW_FEATURES_FLAG="--use-raw-features"
else
    RAW_FEATURES_FLAG=""
fi

if [ -n "$IIM_EPOCHS" ]; then
    EPOCHS_FLAG="--iim-epochs $IIM_EPOCHS"
else
    EPOCHS_FLAG="--iim-epochs 100"
fi

if [ -n "$K_FOLDS" ]; then
    K_FOLDS_FLAG="--k-training $K_FOLDS"
else
    K_FOLDS_FLAG="--k-training 1"
fi

if [ -n "$IIM_NAME" ]; then
    IIM_NAME_FLAG="--iim-name $IIM_NAME"
else
    IIM_NAME_FLAG="--iim-name lstm"
fi

# -----------------------------------------
# Echo configuration info
# -----------------------------------------

echo "==============================================="
echo "OTP Experiment"
echo "==============================================="
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  IIM Model: ${IIM_NAME:-lstm}"
echo "  Cloud Model: ${CLOUD_MODEL:-None}"
echo "  Triangulation Mode: $TRIANGULATION_MODE"
echo "  Triangulation Embedding: $TRIANGULATION_EMBEDDING"
echo "  Triangulation Choosing: $CHOOSE"
echo "  Number of Anchors: $TRIANGULATIONS_NUMBER"
echo "  N_P (Pred Vectors): $N_P"
echo "  K-Folds: ${K_FOLDS:-1}"
echo "  IIM Epochs: ${IIM_EPOCHS:-100}"
echo "  Use Raw Features: ${USE_RAW_FEATURES:-0}"
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
# Run OTP Experiment
# -----------------------------------------

echo ""
echo "==============================================="
echo "Running OTP Experiment Handler"
echo "==============================================="
echo ""

python main.py \
    --experiment-to-run otp_experiment \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $CLOUD_FLAG \
    $EPOCHS_FLAG \
    $K_FOLDS_FLAG \
    $IIM_NAME_FLAG \
    --datasets "$DATASET_NAME" \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ OTP experiment failed!"
    exit 1
fi

# -----------------------------------------
# Final Summary
# -----------------------------------------

echo ""
echo "==============================================="
echo "✅ OTP EXPERIMENT COMPLETE!"
echo "==============================================="
echo ""
echo "Results saved to output/otp/${DATASET_NAME}/k_report.csv"
echo ""
echo "K-fold iterations: ${K_FOLDS:-1}"
echo "Cloud model(s): ${CLOUD_MODEL:-None}"
echo ""
echo "==============================================="
