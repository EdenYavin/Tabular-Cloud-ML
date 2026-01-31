# Key Encoder SBATCH Integration Fix

## Problem

The `key_encoder_train.sbatch` script was failing with:
```
ModuleNotFoundError: No module named 'src'
AttributeError: module '__main__' has no attribute '__file__'
```

This occurred because:
1. Python couldn't resolve the `src` module when running directly via `python -m src.meta_learning.train_key_encoder`
2. The `constansts.py` file tries to get `__file__` from `__main__` module, which doesn't exist in module execution mode

## Solution

Instead of trying to run `train_key_encoder.py` directly, integrate it into the main experiment orchestration system:

### Changes Made

#### 1. Created KeyEncoderTrainingHandler (`src/experiments/key_encoder_training_handler.py`)
- New experiment handler that extends `ExperimentHandler`
- Orchestrates key encoder training with configuration from main config system
- Accepts all parameters (num_keys, embedding_dim, epochs, etc.) from config object
- Properly initializes as a context manager like other handlers

#### 2. Updated EXPERIMENTS Enum (`src/utils/constansts.py`)
- Added `KEY_ENCODER_TRAINING = "key_encoder_training"` to EXPERIMENTS enum
- Allows `--experiment-to-run key_encoder_training` to be recognized

#### 3. Updated main.py
- Imported KeyEncoderTrainingHandler
- Added command-line arguments for key encoder parameters:
  - `--num-keys`
  - `--num-calibration-pairs`
  - `--embedding-dim`
  - `--output-embedding-dim`
  - `--epochs`
  - `--batch-size`
  - `--learning-rate`
  - `--output-dir`
- Added handler dispatch logic for `KEY_ENCODER_TRAINING` experiment type
- Attached key encoder parameters to config object for handler access

#### 4. Updated sbatch Script (`sbatch/key_encoder_train.sbatch`)
- Now calls `python main.py --experiment-to-run key_encoder_training` instead of direct module execution
- Passes all parameters as command-line arguments to main.py
- Still respects environment variables for configuration (NUM_KEYS, EPOCHS, etc.)
- Removed problematic `python -m` module execution

## Usage

Run the sbatch script as before:
```bash
sbatch sbatch/key_encoder_train.sbatch
```

Or with custom parameters via environment variables:
```bash
export NUM_KEYS=1000
export EPOCHS=100
sbatch sbatch/key_encoder_train.sbatch
```

Or manually run main.py:
```bash
python main.py --experiment-to-run key_encoder_training --num-keys 500 --epochs 50
```

## Benefits

1. **Consistency**: Key encoder training uses same orchestration as other experiments
2. **Proper initialization**: `constansts.py` is loaded correctly through main.py's proper execution
3. **Configuration**: All parameters flow through the centralized config system
4. **Error handling**: Leverages existing experiment handler error handling and reporting
5. **Maintainability**: No special-case code paths for key encoder

## Files Modified

- `src/experiments/key_encoder_training_handler.py` - **NEW**
- `src/utils/constansts.py` - Added KEY_ENCODER_TRAINING to EXPERIMENTS enum
- `main.py` - Added handler import, dispatch logic, and command-line arguments
- `sbatch/key_encoder_train.sbatch` - Changed to use main.py instead of direct module execution
