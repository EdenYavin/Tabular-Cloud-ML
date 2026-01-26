"""
Training script for Key Encoder proof-of-concept.

This script:
1. Generates synthetic keyspace data
2. Trains SetTransformer-based Key Encoder with InfoNCE loss
3. Validates that encoder can distinguish different encryption keys
4. Saves trained model and evaluation metrics
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.meta_learning.keyspace_generator import generate_keyspace_dataset
from src.meta_learning.key_encoder import create_key_encoder
from src.encryptor.model import DenseEncryptor
from src.utils.constansts import OUTPUT_DIR_PATH


def train_key_encoder_poc(
    num_keys=500,
    num_calibration_pairs=50,
    embedding_dim=64,
    output_embedding_dim=256,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    validation_split=0.2,
    output_dir=None
):
    """
    Train Key Encoder proof-of-concept.

    Args:
        num_keys: Number of unique encryption keys to generate
        num_calibration_pairs: Calibration pairs per key
        embedding_dim: Dimension of data to encrypt
        output_embedding_dim: Dimension of functional embeddings
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Fraction for validation
        output_dir: Directory to save results

    Returns:
        Dictionary with training results and metrics
    """
    logger.info("=" * 80)
    logger.info("KEY ENCODER PROOF-OF-CONCEPT")
    logger.info("=" * 80)

    # Set up output directory
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR_PATH) / "meta_learning" / "key_encoder_poc"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Generate synthetic keyspace data
    logger.info(f"\n[1/5] Generating synthetic keyspace data for {num_keys} keys...")

    dataset = generate_keyspace_dataset(
        encryptor_class=DenseEncryptor,
        dataset_name="poc",
        embedding_dim=embedding_dim,
        num_keys=num_keys,
        num_calibration_pairs=num_calibration_pairs,
        save_path=str(output_dir / "keyspace_data.npz")
    )

    X_A = dataset['X_A']
    X_B = dataset['X_B']
    key_ids = dataset['key_ids']
    calibration_dim = dataset['calibration_dim']
    encrypted_dim = dataset['encrypted_dim']

    logger.info(f"Generated data shape: X_A={X_A.shape}, X_B={X_B.shape}")
    logger.info(f"Calibration dim: {calibration_dim}, Encrypted dim: {encrypted_dim}")

    # Step 2: Split into train/validation
    logger.info(f"\n[2/5] Splitting into train/validation ({1-validation_split:.0%}/{validation_split:.0%})...")

    indices = np.arange(len(X_A))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_split,
        random_state=42
    )

    X_A_train, X_A_val = X_A[train_idx], X_A[val_idx]
    X_B_train, X_B_val = X_B[train_idx], X_B[val_idx]
    key_ids_train, key_ids_val = key_ids[train_idx], key_ids[val_idx]

    logger.info(f"Train samples: {len(X_A_train)}, Validation samples: {len(X_A_val)}")

    # Step 3: Create and compile Key Encoder
    logger.info(f"\n[3/5] Creating Key Encoder model...")

    encoder = create_key_encoder(
        calibration_dim=calibration_dim,
        encrypted_dim=encrypted_dim,
        embedding_dim=output_embedding_dim
    )

    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # Build the model with the correct input shape
    encoder.build(input_shape=(None, X_A_train.shape[1], X_A_train.shape[2]))

    logger.info("Model architecture:")
    encoder.set_transformer.summary()

    # Step 4: Train the encoder
    logger.info(f"\n[4/5] Training Key Encoder for {epochs} epochs...")

    train_dataset = tf.data.Dataset.from_tensor_slices((X_A_train, X_B_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_A_val, X_B_val))
    val_dataset = val_dataset.batch(batch_size)

    history = encoder.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1
    )

    # Save model
    model_path = output_dir / "key_encoder.keras"
    encoder.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Step 5: Evaluate key discrimination ability
    logger.info(f"\n[5/5] Evaluating key discrimination ability...")

    # Encode all validation samples
    embeddings_val = encoder(X_A_val, training=False).numpy()

    # Test 1: K-NN classification (can we identify keys from embeddings?)
    logger.info("\nTest 1: K-NN Key Classification")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(embeddings_val, key_ids_val)

    predictions = knn.predict(embeddings_val)
    accuracy = accuracy_score(key_ids_val, predictions)

    logger.info(f"  K-NN Accuracy: {accuracy:.4f}")
    logger.info(f"  Baseline (random): {1/len(np.unique(key_ids_val)):.4f}")

    # Test 2: Embedding consistency (same key should have similar embeddings)
    logger.info("\nTest 2: Embedding Consistency")

    embeddings_A_val = encoder(X_A_val, training=False).numpy()
    embeddings_B_val = encoder(X_B_val, training=False).numpy()

    # Normalize
    emb_A_norm = embeddings_A_val / np.linalg.norm(embeddings_A_val, axis=1, keepdims=True)
    emb_B_norm = embeddings_B_val / np.linalg.norm(embeddings_B_val, axis=1, keepdims=True)

    # Positive pair similarity (same key, different context)
    positive_sim = np.sum(emb_A_norm * emb_B_norm, axis=1)
    avg_positive_sim = np.mean(positive_sim)

    # Negative pair similarity (different keys)
    negative_sims = []
    for i in range(min(100, len(emb_A_norm))):
        for j in range(i+1, min(100, len(emb_A_norm))):
            sim = np.dot(emb_A_norm[i], emb_A_norm[j])
            negative_sims.append(sim)

    avg_negative_sim = np.mean(negative_sims)

    logger.info(f"  Average positive pair similarity: {avg_positive_sim:.4f}")
    logger.info(f"  Average negative pair similarity: {avg_negative_sim:.4f}")
    logger.info(f"  Margin (positive - negative): {avg_positive_sim - avg_negative_sim:.4f}")

    # Test 3: Embedding distribution uniformity
    logger.info("\nTest 3: Embedding Distribution")

    embedding_norms = np.linalg.norm(embeddings_val, axis=1)
    logger.info(f"  Embedding norm - mean: {np.mean(embedding_norms):.4f}, std: {np.std(embedding_norms):.4f}")

    # Save results
    results = {
        'history': history.history,
        'knn_accuracy': accuracy,
        'positive_similarity': avg_positive_sim,
        'negative_similarity': avg_negative_sim,
        'margin': avg_positive_sim - avg_negative_sim,
        'embedding_norm_mean': np.mean(embedding_norms),
        'embedding_norm_std': np.std(embedding_norms)
    }

    results_path = output_dir / "training_results.npz"
    np.savez(results_path, **results)
    logger.info(f"\nSaved results to {results_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PROOF-OF-CONCEPT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✓ Generated {num_keys} synthetic keys")
    logger.info(f"✓ Trained Key Encoder for {epochs} epochs")
    logger.info(f"✓ K-NN Accuracy: {accuracy:.4f} (vs random {1/len(np.unique(key_ids_val)):.4f})")
    logger.info(f"✓ Positive/Negative similarity margin: {avg_positive_sim - avg_negative_sim:.4f}")

    if accuracy > 0.8 and (avg_positive_sim - avg_negative_sim) > 0.3:
        logger.info("\n✅ SUCCESS: Encoder can distinguish encryption keys!")
    else:
        logger.warning("\n⚠️  PARTIAL SUCCESS: Encoder shows learning but may need tuning")

    logger.info("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Key Encoder POC")

    parser.add_argument("--num-keys", type=int, default=500,
                       help="Number of unique encryption keys to generate")
    parser.add_argument("--num-calibration-pairs", type=int, default=50,
                       help="Calibration pairs per key")
    parser.add_argument("--embedding-dim", type=int, default=64,
                       help="Dimension of data to encrypt")
    parser.add_argument("--output-embedding-dim", type=int, default=256,
                       help="Dimension of functional embeddings")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results")

    args = parser.parse_args()

    train_key_encoder_poc(
        num_keys=args.num_keys,
        num_calibration_pairs=args.num_calibration_pairs,
        embedding_dim=args.embedding_dim,
        output_embedding_dim=args.output_embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
