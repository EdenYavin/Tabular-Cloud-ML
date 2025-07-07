#!/bin/bash
#SBATCH --partition=main                             # Use 'main' partition unless told otherwise
#SBATCH --time=0-30:00:00                             # Set job time limit to 30 hours
#SBATCH --job-name=tabml_${DATASET_NAME}_training_loop      # Dynamic job name
#SBATCH --output=tabml_${DATASET_NAME}_cloud-%J.out         # Output log includes dataset name
#SBATCH --mail-user=edenyav@post.bgu.ac.il            # Email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL                    # Notify on job begin, end, or fail
#SBATCH --gpus=rtx_3090:1                             # Request 1 GPU
#SBATCH --mem=32G                                     # Request 32 GB of RAM

# Echo SLURM job and dataset info
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Running dataset: $DATASET_NAME"

# Load Anaconda module
module load anaconda

# Activate the Conda environment
source activate tabml

# Run the experiment using the dataset from env var
python main.py \
    --experiment-to-run training_loop \
    --encoder-rotating-key \
    --datasets "$DATASET_NAME" \
    --use-cloud-models xception \
    --number-of-prediction-vector 1 \
    --iim-epochs 100
