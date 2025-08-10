#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=50G

# Echo info
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Running dataset: $DATASET_NAME"
echo "Using Cloud Model: $CLOUD_MODEL"
echo "Running experiment: $EXPERIMENT_NAME"

# Load environment
module load anaconda
source activate tabml
echo "Currently active conda env: $CONDA_DEFAULT_ENV"
which python

# Run the Python script
python main.py \
    --experiment-to-run "$EXPERIMENT_NAME" \
    --encoder-rotating-key \
    --use-raw-features \
    --datasets "$DATASET_NAME" \
    --number-of-prediction-vector "$NUMBER_OF_DATASETS" \
    --use-cloud-models $CLOUD_MODEL \
    --iim-epochs 100 \
    --triangulation-choosing "$CHOOSE" \
    --dataset-use-cache