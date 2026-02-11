#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# =============================================================================
# Feature Ablation Experiment - Frozen T Network Study
# Trains FlexibleSINClassifier on all 4 feature combinations using
# FeatureAblationExperimentHandler
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
    RAW_FEATURES_FLAG="--experiment-use-raw"
else
    RAW_FEATURES_FLAG=""
fi

# Pretrained T Network Path (Optional - uses default path if not specified)
if [ -n "$PRETRAINED_T_NETWORK_PATH" ]; then
    T_NETWORK_PATH_FLAG="--pretrained-t-network $PRETRAINED_T_NETWORK_PATH"
else
    T_NETWORK_PATH_FLAG=""
fi

# Freeze T Network (default: true for ablation studies)
FREEZE_T_FLAG="--freeze-t-network"

# IIM Epochs
if [ -n "$IIM_EPOCHS" ]; then
    EPOCHS_FLAG="--iim-epochs $IIM_EPOCHS"
else
    EPOCHS_FLAG="--iim-epochs 100"
fi

# K-Fold iterations (default: 1)
if [ -n "$K_FOLDS" ]; then
    K_FOLDS_FLAG="--k-training $K_FOLDS"
else
    K_FOLDS_FLAG="--k-training 1"
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
echo "  T Network Path: ${PRETRAINED_T_NETWORK_PATH:-auto-detect}"
echo "  Freeze T Network: true"
echo "  Triangulation Mode: $TRIANGULATION_MODE"
echo "  Triangulation Embedding: $TRIANGULATION_EMBEDDING"
echo "  Triangulation Choosing: $CHOOSE"
echo "  Number of Anchors: $TRIANGULATIONS_NUMBER"
echo "  N_P (Pred Vectors): $N_P"
echo "  K-Folds: ${K_FOLDS:-1}"
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
# Run Feature Ablation Experiment
# Processes all 4 combinations automatically:
#   1. baseline_no_cloud: [p_x, q_x, T_context]
#   2. no_raw_embedding: [q_x, T_context]
#   3. full_features: [p_x, q_x, T_context, cloud]
#   4. cloud_no_raw: [q_x, T_context, cloud]
# -----------------------------------------

echo ""
echo "==============================================="
echo "Running Feature Ablation Experiment Handler"
echo "==============================================="
echo ""

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
    $FREEZE_T_FLAG \
    $CLOUD_FLAG \
    $EPOCHS_FLAG \
    $K_FOLDS_FLAG \
    --datasets "$DATASET_NAME" \
    --dataset-use-cache

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Feature ablation experiment failed!"
    exit 1
fi

# -----------------------------------------
# Final Summary
# -----------------------------------------

echo ""
echo "==============================================="
echo "✅ FEATURE ABLATION EXPERIMENT COMPLETE!"
echo "==============================================="
echo ""
echo "Results saved to output/report.csv"
echo ""
echo "All 4 combinations processed:"
echo "  1. baseline_no_cloud: [p_x, q_x, T_context]"
echo "  2. no_raw_embedding: [q_x, T_context]"
echo "  3. full_features: [p_x, q_x, T_context, cloud]"
echo "  4. cloud_no_raw: [q_x, T_context, cloud]"
echo ""
echo "K-fold iterations: ${K_FOLDS:-1}"
echo "Results logged via log_k_results() for statistical analysis"
echo ""
echo "==============================================="