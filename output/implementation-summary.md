# Implementation Summary: FiLM-Conditioned IIM + Multi-Calibration

**Date**: 2026-01-16
**Status**: Ready for Testing

## What Was Implemented

### 1. FiLM-Conditioned IIM Architecture

A new Internal Inference Model that uses Feature-wise Linear Modulation (FiLM) to better leverage calibration embeddings for rotating encryption keys.

**Key Innovation**: Instead of concatenating calibration vectors as additional features, FiLM uses them to generate scale (γ) and shift (β) parameters that dynamically modulate the network's feature processing.

**Files Modified**:
- `src/internal_model/model.py` - Added `FiLMConditionedIIM` class
- `src/internal_model/__init__.py` - Registered model in factory
- `src/utils/constansts.py` - Added `FILM_CONDITIONED` to enum

**Usage**:
```bash
python main.py --iim-name film_conditioned --use-calibration-vector ...
```

### 2. Multi-Distribution Calibration Vectors

Instead of a single calibration vector, the system now supports **multiple calibration vectors** with different statistical distributions, creating a richer "key fingerprint."

**Available Distributions**:
- `uniform`: Uniform random [0, 1]
- `gaussian`: Normal distribution centered at 0.5
- `sparse`: 75% zeros, 25% random (tests sparse patterns)
- `bimodal`: Two peaks at 0.2 and 0.8 (tests contrasting signals)
- `edges`: Binary 0/1 values (tests extremes)

**Files Modified**:
- `src/utils/helpers.py` - Added `generate_calibration_vectors()` function
- `src/pipeline/triangulations_features_dataset.py` - Updated to use multi-calibration
- `main.py` - Added `--calibration-distributions` argument
- `src/utils/helpers.py` - Updated path generation to include distribution types

**Usage**:
```bash
# Single distribution
python main.py --use-calibration-vector --calibration-distributions gaussian ...

# Multiple distributions (recommended)
python main.py --use-calibration-vector --calibration-distributions gaussian sparse bimodal ...

# All distributions
python main.py --use-calibration-vector --calibration-distributions uniform gaussian sparse bimodal edges ...
```

## Architecture Overview

```
Flow with FiLM + Multi-Calibration:

1. Generate multiple calibration vectors (e.g., gaussian, sparse, bimodal)
2. Encrypt each with current key → 3 encrypted vectors
3. Embed each → 3 x 768d = 2304d calibration fingerprint
4. FiLM Generator: calibration_fingerprint → (γ₁, β₁, γ₂, β₂)
5. Main features → Dense → FiLM(γ₁,β₁) → Dense → FiLM(γ₂,β₂) → Output
```

## Command-Line Arguments

### New Arguments

```bash
--calibration-distributions <dist1> <dist2> ...
  Specify which distribution types to use for calibration vectors.
  Options: uniform, gaussian, sparse, bimodal, edges
  Default: ["gaussian"]
  Example: --calibration-distributions gaussian sparse bimodal
```

### Required for FiLM

```bash
--iim-name film_conditioned
  Use the FiLM-conditioned IIM architecture

--use-calibration-vector
  Enable calibration vector(s)
```

## Complete Example Commands

### Minimal Test
```bash
python main.py \
  --experiment-to-run model_training \
  --iim-name film_conditioned \
  --use-calibration-vector \
  --calibration-distributions gaussian \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30
```

### Recommended Configuration
```bash
python main.py \
  --experiment-to-run model_training \
  --iim-name film_conditioned \
  --use-calibration-vector \
  --calibration-distributions gaussian sparse bimodal \
  --use-cloud-models xception \
  --datasets mushroom \
  --triangulation-samples 3 \
  --triangulation-mode diff \
  --triangulation-embedding-model dino \
  --iim-epochs 30 \
  --encoder-rotating-key
```

### Comparison with Baseline
```bash
# Old approach (concatenation with single gaussian)
python main.py \
  --iim-name dense \
  --use-calibration-vector \
  --calibration-distributions gaussian \
  --use-cloud-models xception \
  --datasets mushroom

# New approach (FiLM with multi-calibration)
python main.py \
  --iim-name film_conditioned \
  --use-calibration-vector \
  --calibration-distributions gaussian sparse bimodal \
  --use-cloud-models xception \
  --datasets mushroom
```

## Testing Strategy

### Phase 1: Verify Implementation
```bash
# Test model compiles
python -c "from src.internal_model import FiLMConditionedIIM; print('OK')"

# Test with minimal data
python main.py --iim-name film_conditioned --use-calibration-vector --calibration-distributions gaussian --datasets mushroom --iim-epochs 5
```

### Phase 2: Ablation Study
Test incrementally:
1. Baseline: No calibration
2. Single: `gaussian`
3. Double: `gaussian sparse`
4. Triple: `gaussian sparse bimodal`
5. All: `uniform gaussian sparse bimodal edges`

### Phase 3: Dataset Evaluation
Run best configuration from Phase 2 on multiple datasets:
- mushroom (easy)
- adult (hard)
- bank_marketing (large)

## Expected Outcomes

### Hypothesis
Multi-calibration with FiLM should improve performance on datasets where:
1. Rotating keys cause high variance in triangulation features
2. The calibration signal contains learnable structure about the key
3. The cloud vector provides complementary information

### Metrics to Track
- Accuracy improvement over baseline (no calibration)
- Consistency across datasets
- Training stability (loss curves)
- Inference time overhead

## Files Created/Modified

### Created
- `src/internal_model/model.py::FiLMConditionedIIM` (lines 332-452)
- `src/utils/helpers.py::generate_calibration_vectors()` (lines 327-400)
- `docs/plans/2026-01-16-film-conditioned-iim-design.md`
- `docs/multi-calibration-usage.md`
- `output/example-calibration-commands.sh`

### Modified
- `src/utils/constansts.py` - Added `FILM_CONDITIONED` enum
- `src/internal_model/__init__.py` - Registered FiLMConditionedIIM
- `src/pipeline/triangulations_features_dataset.py` - Multi-calibration support
- `src/utils/helpers.py` - Updated `get_dataset_path()` to include distributions
- `main.py` - Added `--calibration-distributions` argument

## Next Steps

1. **Test Compilation**: Verify the model compiles and trains
2. **Run Ablation**: Test different distribution combinations
3. **Compare Performance**: Evaluate against baseline
4. **Tune Hyperparameters**: Adjust `film_hidden_dim`, dropout, etc.
5. **Document Results**: Record which configurations work best

## References

- **FiLM Paper**: [Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871)
- **Hypernetworks**: [Ha et al., 2016](https://arxiv.org/abs/1609.09106)
- **Original EPKD**: `papers/Secure_ML_in_the_Cloud___KDD_2023 (2).pdf`

## Questions & Answers

**Q: Why FiLM instead of concatenation?**
A: Concatenation treats calibration as "more data." FiLM treats it as "instructions for how to interpret the data," giving it direct control over feature processing.

**Q: How many calibration distributions should I use?**
A: Start with 2-3 diverse ones (e.g., `gaussian sparse bimodal`). More isn't always better—test what works for your data.

**Q: Does this work without rotating keys?**
A: Yes, but the benefit is smaller. The main use case is when each sample has a different encryption key.

**Q: What if performance doesn't improve?**
A: Try:
1. Different distribution combinations
2. Adjusting FiLM architecture (film_hidden_dim)
3. More training epochs
4. Checking if the problem is actually the key variation (run diagnostics)

**Q: Can I use this with other IIM architectures?**
A: FiLM is specifically designed for calibration. For other architectures, you might try the multi-calibration with concatenation (use `--iim-name dense` or `transformer`).
