#!/bin/bash
# Example commands for testing multi-calibration vectors
# Created: 2026-01-16

# =============================================================================
# SINGLE DISTRIBUTION TESTS
# =============================================================================

# Test 1: Gaussian only (current baseline behavior)
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions gaussian \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30 \
  --triangulation-samples 3 \
  --triangulation-mode diff \
  --triangulation-embedding-model dino

# Test 2: Uniform distribution
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions uniform \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 3: Sparse distribution
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions sparse \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 4: Bimodal distribution
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions bimodal \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 5: Edges distribution
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions edges \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# =============================================================================
# DOUBLE DISTRIBUTION TESTS (Recommended starting point)
# =============================================================================

# Test 6: Gaussian + Sparse
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions gaussian sparse \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 7: Uniform + Bimodal
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions uniform bimodal \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 8: Sparse + Edges
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions sparse edges \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# =============================================================================
# TRIPLE DISTRIBUTION TESTS (Good balance)
# =============================================================================

# Test 9: Gaussian + Sparse + Bimodal (RECOMMENDED)
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions gaussian sparse bimodal \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# Test 10: Uniform + Sparse + Edges
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions uniform sparse edges \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# =============================================================================
# ALL DISTRIBUTIONS (Maximum information)
# =============================================================================

# Test 11: All five distributions
python main.py \
  --experiment-to-run model_training \
  --use-calibration-vector \
  --calibration-distributions uniform gaussian sparse bimodal edges \
  --iim-name film_conditioned \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# =============================================================================
# COMPARISON WITH BASELINE (No calibration)
# =============================================================================

# Test 12: No calibration (baseline)
python main.py \
  --experiment-to-run model_training \
  --iim-name dense \
  --use-cloud-models xception \
  --datasets mushroom \
  --iim-epochs 30

# =============================================================================
# SYSTEMATIC LOOP (Uncomment to run all)
# =============================================================================

# for dist in "gaussian" "uniform" "sparse" "bimodal" "edges"; do
#   echo "Testing distribution: $dist"
#   python main.py \
#     --experiment-to-run model_training \
#     --use-calibration-vector \
#     --calibration-distributions $dist \
#     --iim-name film_conditioned \
#     --use-cloud-models xception \
#     --datasets mushroom \
#     --iim-epochs 30
# done
