#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# =============================================================================
# Feature Ablation Experiment - Frozen T Network Study
# Trains FlexibleSINClassifier on all 4 feature combinations
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

# Number of Prediction Vectors
if [ -n "$N_P" ]; then
    N_P_FLAG="--number-of-prediction-vector $N_P"
else
    N_P_FLAG="--number-of-prediction-vector 1"
fi

# Cloud Models (for combo3/combo4)
if [ -n "$CLOUD_MODELS" ]; then
    CLOUD_FLAG="--use-cloud-models $CLOUD_MODELS"
else
    CLOUD_FLAG=""
fi

# Use Raw Features
if [ -n "$USE_RAW_FEATURES" ] && [ "$USE_RAW_FEATURES" = "1" ]; then
    RAW_FEATURES_FLAG="--use-raw-features"
else
    RAW_FEATURES_FLAG=""
fi

# Pretrained T Network Path (Optional - uses default path if not specified)
if [ -n "$PRETRAINED_T_NETWORK_PATH" ]; then
    T_NETWORK_PATH_FLAG="--pretrained-t-network $PRETRAINED_T_NETWORK_PATH"
else
    T_NETWORK_PATH_FLAG=""
fi

# IIM Epochs
if [ -n "$IIM_EPOCHS" ]; then
    EPOCHS_FLAG="--iim-epochs $IIM_EPOCHS"
else
    EPOCHS_FLAG="--iim-epochs 100"
fi

# -----------------------------------------
# Echo configuration info
# -----------------------------------------

echo "==============================================="
echo "Feature Ablation Experiment - Frozen T Network"
echo "==============================================="
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  T Network Path: ${PRETRAINED_T_NETWORK_PATH:-default}"
echo "  Triangulation Mode: $TRIANGULATION_MODE"
echo "  Triangulation Embedding: $TRIANGULATION_EMBEDDING"
echo "  Triangulation Choosing: $CHOOSE"
echo "  Number of Anchors: $TRIANGULATIONS_NUMBER"
echo "  N_P (Pred Vectors): $N_P"
echo "  IIM Epochs: ${IIM_EPOCHS:-100}"
echo "  Cloud Models: ${CLOUD_MODELS:-None}"
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
# STEP 1: Create datasets for all 4 combos
# -----------------------------------------

echo ""
echo "==============================================="
echo "STEP 1: Creating datasets for all combinations"
echo "==============================================="

# Combo1: [p_x, q_x, T_context]
echo ""
echo "--- Creating combo1 dataset ---"
python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination combo1 \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ Combo1 dataset creation failed! Exiting."
    exit 1
fi

# Combo2: [q_x, T_context]
echo ""
echo "--- Creating combo2 dataset ---"
python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination combo2 \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ Combo2 dataset creation failed! Exiting."
    exit 1
fi

# Combo3: [p_x, q_x, T_context, cloud]
echo ""
echo "--- Creating combo3 dataset ---"
python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    $CLOUD_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination combo3 \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ Combo3 dataset creation failed! Exiting."
    exit 1
fi

# Combo4: [q_x, T_context, cloud]
echo ""
echo "--- Creating combo4 dataset ---"
python main.py \
    --experiment-to-run dataset_creation \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    $CLOUD_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination combo4 \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ Combo4 dataset creation failed! Exiting."
    exit 1
fi

echo ""
echo "✅ All dataset combinations created successfully!"

# -----------------------------------------
# STEP 2: Run ablation experiment
# -----------------------------------------

echo ""
echo "==============================================="
echo "STEP 2: Running Feature Ablation Training"
echo "==============================================="

python main.py \
    --experiment-to-run ablation_experiment \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    $CLOUD_FLAG \
    $EPOCHS_FLAG \
    --datasets "$DATASET_NAME" \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ Ablation experiment failed!"
    exit 1
fi

echo ""
echo "==============================================="
echo "✅ Feature Ablation Experiment Complete!"
echo "Results saved to output/report.csv"
echo "==============================================="