#!/bin/bash
#SBATCH --partition=main                     # Use 'main' partition unless told otherwise
#SBATCH --time=0-20:00:00                    # Set job time limit to 10 hours (adjust as needed)
#SBATCH --job-name=tabml_mushroom            # Name your job for easier tracking
#SBATCH --output=tabml_mushroom-%J.out       # Output log file, %J will be replaced by job ID
#SBATCH --mail-user=your.email@post.bgu.ac.il  # Change to your own BGU email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL           # Get email on job start, end, or failure
#SBATCH --gpus=1                             # Request 1 GPU

# Print SLURM job info for debugging
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# Load Anaconda module (as per cluster setup)
module load anaconda

# Activate your Conda environment
source activate tabml

# Run your experiment
python main.py --experiment-to-run dataset_creation --encoder-rotating-key --datasets mushroom --number-of-prediction-vector 4 --use-cloud-models xception
