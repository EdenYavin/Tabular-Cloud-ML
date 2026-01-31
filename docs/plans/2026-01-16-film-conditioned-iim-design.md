
# FiLM-Conditioned IIM Design Document

**Date**: 2026-01-16
**Author**: Claude (via brainstorming session)
**Status**: Ready for Implementation

## Problem Statement

The current IIM struggles to learn effectively with rotating encryption keys. Each sample is encrypted with a different key, causing the triangulation features to vary wildly between samples even when they are semantically similar.

The calibration vector approach (concatenating the encrypted embedding of a fixed reference vector) provides minimal improvement because:
1. Concatenation treats calibration as "just more features"
2. The network must implicitly learn to use it to interpret other features
3. This is a hard learning problem with limited signal

## Proposed Solution: FiLM Conditioning

**Key Insight**: The calibration embedding shouldn't be treated as "more data" - it should be treated as "instructions for how to interpret the data."

### FiLM (Feature-wise Linear Modulation)

Instead of concatenating the calibration embedding, use it to **generate** scale (γ) and shift (β) parameters that modulate each layer's features.

```
Current Flow:
  [triangulation | calib_emb | cloud_vec] → Dense → Dense → Output

FiLM Flow:
  calib_emb → MLP → (γ, β) parameters
  [triangulation | cloud_vec] → Dense → FiLM(γ,β) → Dense → FiLM(γ,β) → Output
```

### Why FiLM Works for Rotating Keys

1. Each key produces a different calibration embedding
2. FiLM parameters adapt the feature processing to that specific key's "transformation space"
3. The IIM learns key-invariant representations internally while FiLM handles key-specific adaptation

## Architecture

```python
class FiLMConditionedIIM:
    Input: [main_features | calib_embedding | cloud_vector (optional)]

    # FiLM Generator (from calibration embedding)
    film_gen = Dense(128, relu) → Dense(256, relu)
    gamma_1, beta_1 = Dense(256), Dense(256)
    gamma_2, beta_2 = Dense(128), Dense(128)

    # Main pathway with FiLM modulation
    x = Dense(256, leaky_relu)(main_features)
    x = BatchNorm(x)
    x = gamma_1 * x + beta_1  # FiLM Layer 1
    x = Dropout(0.2)(x)

    x = Dense(128, leaky_relu)(x)
    x = BatchNorm(x)
    x = gamma_2 * x + beta_2  # FiLM Layer 2
    x = Dropout(rate)(x)

    # Fuse cloud vector if present
    if cloud_vector: x = concat([x, cloud_vector])

    # Classification head
    x = Dense(64, leaky_relu)(x)
    output = Dense(num_classes, softmax)(x)
```

## Implementation Plan

### Files to Modify

1. `src/utils/constansts.py` - Add `FILM_CONDITIONED` to `IIM_MODELS` enum
2. `src/internal_model/model.py` - Add `FiLMConditionedIIM` class
3. `src/internal_model/base.py` - Register model in factory (if applicable)

### Optional Enhancement: Multi-Calibration Fingerprint

Instead of a single calibration vector, use 3 vectors with different characteristics:
- Uniform random
- Gaussian centered
- Sparse pattern

This creates a richer "key fingerprint" for better FiLM conditioning.

## Expected Outcomes

1. More stable performance across datasets
2. Better utilization of the calibration vector information
3. Improved accuracy when using cloud vectors with rotating keys

## References

- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Hypernetworks](https://arxiv.org/abs/1609.09106)
- [A Brief Review of Hypernetworks in Deep Learning](https://arxiv.org/html/2306.06955v3)