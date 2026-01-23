"""
Handler for Key Encoder training experiments.

Orchestrates the training of the Key Encoder model for encryption key analysis.
"""

from loguru import logger
from src.experiments.base import ExperimentHandler
from src.meta_learning.train_key_encoder import train_key_encoder_poc
from src.utils.config import config
from src.utils.constansts import REPORT_PATH


class KeyEncoderTrainingHandler(ExperimentHandler):
    """
    Handler for Key Encoder training experiments.

    Runs the key encoder proof-of-concept training with configuration
    from the main config system and command-line arguments.
    """

    def __init__(self, report_path: str = REPORT_PATH):
        super().__init__(experiment_name="key_encoder_training", report_path=report_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run_experiment(self):
        """
        Run key encoder training with configuration from config object.

        Uses command-line arguments via the config system to customize:
        - Number of keys
        - Number of calibration pairs
        - Embedding dimensions
        - Training epochs, batch size, learning rate
        - Output directory
        """
        logger.info("=" * 60)
        logger.info("Starting Key Encoder Training")
        logger.info("=" * 60)

        # Extract parameters from config or use defaults
        # These can be set via command-line arguments or config file
        num_keys = getattr(config, 'num_keys', 500)
        num_calibration_pairs = getattr(config, 'num_calibration_pairs', 50)
        embedding_dim = getattr(config, 'embedding_dim', 64)
        output_embedding_dim = getattr(config, 'output_embedding_dim', 256)
        epochs = getattr(config, 'num_epochs', 50)
        batch_size = getattr(config, 'batch_size', 32)
        learning_rate = getattr(config, 'learning_rate', 1e-3)
        output_dir = getattr(config, 'output_dir', None)

        logger.info(f"Key Encoder Configuration:")
        logger.info(f"  - Number of keys: {num_keys}")
        logger.info(f"  - Calibration pairs per key: {num_calibration_pairs}")
        logger.info(f"  - Input embedding dimension: {embedding_dim}")
        logger.info(f"  - Output embedding dimension: {output_embedding_dim}")
        logger.info(f"  - Training epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        if output_dir:
            logger.info(f"  - Output directory: {output_dir}")

        try:
            # Run the key encoder training
            train_key_encoder_poc(
                num_keys=num_keys,
                num_calibration_pairs=num_calibration_pairs,
                embedding_dim=embedding_dim,
                output_embedding_dim=output_embedding_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir
            )
            logger.info("Key Encoder Training Completed Successfully")

        except Exception as e:
            logger.error(f"Key Encoder Training Failed: {e}")
            raise