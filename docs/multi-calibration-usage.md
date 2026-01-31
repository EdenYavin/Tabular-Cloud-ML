# Multi-Calibration Vectors: Usage Guide

**Created**: 2026-01-16
**Feature**: Multiple calibration vectors with different distributions

## Overview

Instead of using a single calibration vector, you can now use **multiple calibration vectors** with different statistical distributions. Each distribution captures different aspects of the encryption key's behavior, creating a richer "key fingerprint" for the IIM to learn from.

## Why Multiple Distributions?

Different distributions test different aspects of how the encryption key transforms data:

| Distribution | Purpose | Characteristics |
|-------------|---------|-----------------|
| **uniform** | Tests consistent response across entire range | Values uniformly distributed [0, 1] |
| **gaussian** | Tests natural variation around center | Normal distribution centered at 0.5, clipped to [0, 1] |
| **sparse** | Tests sparse pattern sensitivity | 75% zeros, 25% random values [0.5, 1.0] |
| **bimodal** | Tests contrasting signal patterns | Two peaks: 50% at ~0.2, 50% at ~0.8 |
| **edges** | Tests extreme value responses | Binary: 50% at 0, 50% at 1 |

## Command-Line Usage

### Basic Usage (Single Distribution)

```bash
python main.py \
  --use-calibration-vector \
  --calibration-distributions gaussian \
  --iim-name film_conditioned \
  ... other args ...
```

### Multiple Distributions (Recommended)

Use 2-3 distributions for a richer key fingerprint:

```bash
python main.py \
  --use-calibration-vector \
  --calibration-distributions gaussian sparse bimodal \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30
```

### All Distributions (Maximum Information)

```bash
python main.py \
  --use-calibration-vector \
  --calibration-distributions uniform gaussian sparse bimodal edges \
  --iim-name film_conditioned \
  ... other args ...
```

## Examples

### Example 1: Testing Different Combinations

**Hypothesis**: Sparse + Bimodal might capture key structure better than Gaussian alone

```bash
# Baseline: Single Gaussian (current default)
python main.py --use-calibration-vector \
  --calibration-distributions gaussian \
  --iim-name film_conditioned \
  --datasets mushroom

# Test: Sparse + Bimodal
python main.py --use-calibration-vector \
  --calibration-distributions sparse bimodal \
  --iim-name film_conditioned \
  --datasets mushroom

# Test: All three
python main.py --use-calibration-vector \
  --calibration-distributions gaussian sparse bimodal \
  --iim-name film_conditioned \
  --datasets mushroom
```

### Example 2: Systematic Evaluation

```bash
# Create a batch script to test all combinations
for dist in "gaussian" "uniform" "sparse" "bimodal" "edges"; do
  python main.py --use-calibration-vector \
    --calibration-distributions $dist \
    --iim-name film_conditioned \
    --datasets mushroom
done

# Then test pairs
python main.py --use-calibration-vector \
  --calibration-distributions gaussian sparse \
  --iim-name film_conditioned \
  --datasets mushroom

python main.py --use-calibration-vector \
  --calibration-distributions uniform bimodal \
  --iim-name film_conditioned \
  --datasets mushroom
```

## Integration with FiLM-Conditioned IIM

The **FiLMConditionedIIM** automatically adapts to the number of calibration vectors:

- 1 distribution: 768d calibration embedding (for DINO)
- 2 distributions: 1536d calibration embedding
- 3 distributions: 2304d calibration embedding
- etc.

The FiLM generator learns to extract relevant information from this richer signal.

## Output Paths

Results are automatically organized by distribution types:

```
output/
  mushroom/
    cloud/
      xception/
        dino/
          triangulation_and_raw/
            1/
              classes/
                diff/
                  calib_gaussian/              # Single Gaussian
                    3/
                  calib_gaussian_sparse/       # Gaussian + Sparse (sorted)
                    3/
                  calib_bimodal_gaussian_sparse/  # All three (sorted alphabetically)
                    3/
```

## Recommendations

### For Initial Experiments
Start with **2-3 diverse distributions**:
```bash
--calibration-distributions gaussian sparse bimodal
```

### For Ablation Studies
Test incrementally:
1. Single: `gaussian`
2. Pair: `gaussian sparse`
3. Triple: `gaussian sparse bimodal`
4. All: `uniform gaussian sparse bimodal edges`

### For Production
Use the combination that performed best in your experiments.

## Technical Details

### How It Works

1. **Generation**: Each distribution type generates a fixed vector (same across train/test with seed=42)
2. **Encryption**: Each calibration vector is encrypted with the current key
3. **Embedding**: Each encrypted vector is embedded (768d for DINO, 512d for CLIP)
4. **Concatenation**: All calibration embeddings are concatenated into a single fingerprint
5. **FiLM**: The concatenated fingerprint generates scale (γ) and shift (β) parameters

### Dimensionality

```python
# For DINO (768d embeddings):
n_distributions = len(calibration_distributions)
calib_embedding_dim = 768 * n_distributions

# For CLIP (512d embeddings):
calib_embedding_dim = 512 * n_distributions
```

### Computational Cost

- **Training**: Minimal overhead (calibration vectors are generated once, encrypted per sample)
- **Inference**: Same as training
- **Memory**: Increases linearly with number of distributions

## Troubleshooting

### Issue: "calibration_distributions not found in config"

**Solution**: Make sure you're using `--use-calibration-vector` flag:
```bash
python main.py --use-calibration-vector --calibration-distributions gaussian sparse ...
```

### Issue: Results not improving with more distributions

**Possible causes**:
1. The key variation isn't actually helping (try different distribution combinations)
2. The IIM architecture may need tuning (try adjusting film_hidden_dim)
3. Not enough training data to learn the richer representation

### Issue: Different results from same command

**Check**: Distribution order doesn't matter (they're sorted alphabetically in paths), but make sure:
- Same seed (default: 42)
- Same encoder configuration
- Same preprocessing

## Next Steps

After implementing multi-calibration:
1. Run ablation studies on 1-2 datasets
2. Compare results with baseline (no calibration)
3. Document which distribution combinations work best for your datasets
4. Consider dataset-specific calibration strategies

## References

- Design document: `docs/plans/2026-01-16-film-conditioned-iim-design.md`
- Implementation: `src/utils/helpers.py::generate_calibration_vectors()`
- Pipeline: `src/pipeline/triangulations_features_dataset.py`
