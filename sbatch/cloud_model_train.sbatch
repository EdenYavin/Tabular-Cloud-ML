#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# --- NEW: Check for Triangulation Mode ---
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


# --- NEW: Check for Dropout ---
if [ -n "$DROPOUT" ]; then
    DROPOUT_FLAG="--iim-dropout $DROPOUT"
else
    DROPOUT_FLAG=""
fi
# -----------------------------------------

# Echo info
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Running dataset: $DATASET_NAME"
echo "Running triangulation method $CHOOSE"
echo "Number of triangulation $TRIANGULATIONS_NUMBER"
echo "Number of prediction vectors: $N_P"
echo "Number of K Iteration $K"
echo "IIM Models to train $MODELS"
echo "Dropout Config: $DROPOUT" # Log the mode
echo "Cloud Models To Use: $CLOUD_MODELS"
echo "Triangulation Mode: $TRIANGULATION_MODE"
echo "Triangulation Embedding: $TRIANG_EMB_FLAG" # Log the mode


# Load environment
module load anaconda
source activate tabml
echo "Currently active conda env: $CONDA_DEFAULT_ENV"
which python

# Run the Python script

if [ -n "$USE_RAW_FEATURES" ] && [ "$USE_RAW_FEATURES" = "1" ]; then
    echo "Raw Features Flag is Turned ON"
    python main.py --experiment-to-run model_training $DROPOUT_FLAG --encoder-rotating-key --use-raw-features $TRIANG_MODE_FLAG $TRIANG_EMB_FLAG --use-cloud-models $CLOUD_MODELS --datasets "$DATASET_NAME" --number-of-prediction-vector "$N_P" --iim-name $MODELS --iim-epochs 100 --triangulation-choosing "$CHOOSE" --triangulation-samples "$TRIANGULATIONS_NUMBER" --k-training $K
else
    echo "Raw Features Flag is Turned OFF"
    python main.py --experiment-to-run model_training $DROPOUT_FLAG --encoder-rotating-key $TRIANG_MODE_FLAG $TRIANG_EMB_FLAG --use-cloud-models $CLOUD_MODELS --datasets "$DATASET_NAME" --number-of-prediction-vector "$N_P" --iim-name $MODELS --iim-epochs 100 --triangulation-choosing classes --triangulation-samples "$TRIANGULATIONS_NUMBER" --k-training $K
fi