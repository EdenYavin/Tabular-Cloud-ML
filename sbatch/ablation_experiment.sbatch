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
# Process each combination: create dataset then train
# -----------------------------------------

echo ""
echo "==============================================="
echo "Processing Feature Combinations Sequentially"
echo "==============================================="

# -----------------------------------------
# Combination 1: baseline_no_cloud
# -----------------------------------------
echo ""
echo "═══════════════════════════════════════════════"
echo "COMBINATION 1/4: baseline_no_cloud"
echo "═══════════════════════════════════════════════"

echo ""
echo "--- Step 1.1: Creating dataset ---"
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
    --feature-combination baseline_no_cloud \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ baseline_no_cloud dataset creation failed! Exiting."
    exit 1
fi

echo ""
echo "--- Step 1.2: Training model ---"
python main.py \
    --experiment-to-run model_training \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    $EPOCHS_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination baseline_no_cloud \
    --iim-name flexible_sin \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ baseline_no_cloud training failed! Exiting."
    exit 1
fi

echo "✅ baseline_no_cloud complete!"

# -----------------------------------------
# Combination 2: no_raw_embedding
# -----------------------------------------
echo ""
echo "═══════════════════════════════════════════════"
echo "COMBINATION 2/4: no_raw_embedding"
echo "═══════════════════════════════════════════════"

echo ""
echo "--- Step 2.1: Creating dataset ---"
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
    --feature-combination no_raw_embedding \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ no_raw_embedding dataset creation failed! Exiting."
    exit 1
fi

echo ""
echo "--- Step 2.2: Training model ---"
python main.py \
    --experiment-to-run model_training \
    --encoder-rotating-key \
    $RAW_FEATURES_FLAG \
    $TRIANG_MODE_FLAG \
    $TRIANG_EMB_FLAG \
    $TRIANG_CHOOSE_FLAG \
    $TRIANG_SAMPLES_FLAG \
    $N_P_FLAG \
    $T_NETWORK_PATH_FLAG \
    $EPOCHS_FLAG \
    --datasets "$DATASET_NAME" \
    --feature-combination no_raw_embedding \
    --iim-name flexible_sin \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ no_raw_embedding training failed! Exiting."
    exit 1
fi

echo "✅ no_raw_embedding complete!"

# -----------------------------------------
# Combination 3: full_features
# -----------------------------------------
echo ""
echo "═══════════════════════════════════════════════"
echo "COMBINATION 3/4: full_features"
echo "═══════════════════════════════════════════════"

echo ""
echo "--- Step 3.1: Creating dataset ---"
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
    --feature-combination full_features \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ full_features dataset creation failed! Exiting."
    exit 1
fi

echo ""
echo "--- Step 3.2: Training model ---"
python main.py \
    --experiment-to-run model_training \
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
    --feature-combination full_features \
    --iim-name flexible_sin \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ full_features training failed! Exiting."
    exit 1
fi

echo "✅ full_features complete!"

# -----------------------------------------
# Combination 4: cloud_no_raw
# -----------------------------------------
echo ""
echo "═══════════════════════════════════════════════"
echo "COMBINATION 4/4: cloud_no_raw"
echo "═══════════════════════════════════════════════"

echo ""
echo "--- Step 4.1: Creating dataset ---"
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
    --feature-combination cloud_no_raw \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ cloud_no_raw dataset creation failed! Exiting."
    exit 1
fi

echo ""
echo "--- Step 4.2: Training model ---"
python main.py \
    --experiment-to-run model_training \
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
    --feature-combination cloud_no_raw \
    --iim-name flexible_sin \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo "❌ cloud_no_raw training failed! Exiting."
    exit 1
fi

echo "✅ cloud_no_raw complete!"

# -----------------------------------------
# Final Summary
# -----------------------------------------

echo ""
echo "==============================================="
echo "✅ ALL FEATURE COMBINATIONS COMPLETE!"
echo "==============================================="
echo ""
echo "Results saved to output/report.csv"
echo ""
echo "Completed combinations:"
echo "  1. baseline_no_cloud: [p_x, q_x, T_context]"
echo "  2. no_raw_embedding: [q_x, T_context]"
echo "  3. full_features: [p_x, q_x, T_context, cloud]"
echo "  4. cloud_no_raw: [q_x, T_context, cloud]"
echo ""
echo "==============================================="