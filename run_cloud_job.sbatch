#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-30:00:00
#SBATCH --mail-user=edenyav@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=36G

# Echo info
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Running dataset: $DATASET_NAME"

# Load environment
module load anaconda
source activate tabml

# Run the Python script
python main.py \
    --experiment-to-run training_loop \
    --encoder-rotating-key \
    --datasets "$DATASET_NAME" \
    --use-cloud-models xception \
    --number-of-prediction-vector 1 \
    --iim-epochs 100
