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
echo "Number of triangulation $TRIANGULATIONS_NUMBER"
echo "Number of prediction vectors: $N_P"
echo "Number of K Iteration $K"
echo "Use raw features flag: $RAW_FEATURES_FLAG"

# Load environment
module load anaconda
source activate tabml
echo "Currently active conda env: $CONDA_DEFAULT_ENV"
which python

# Run the Python script

if [ -n "$USE_RAW_FEATURES" ]; then
    echo "Raw Features Flag is Turned ON"
    python main.py --experiment-to-run model_training --encoder-rotating-key --use-raw-features --use-cloud-models xception --datasets "$DATASET_NAME" --number-of-prediction-vector "$N_P" --iim-name lstm --iim-epochs 100 --triangulation-choosing classes --triangulation-samples 2 --k-training $K
else
    echo "Raw Features Flag is Turned OFF"
    python main.py --experiment-to-run model_training --encoder-rotating-key --use-cloud-models xception --datasets "$DATASET_NAME" --number-of-prediction-vector "$N_P" --iim-name lstm --iim-epochs 100 --triangulation-choosing classes --triangulation-samples 2 --k-training $K
fi

