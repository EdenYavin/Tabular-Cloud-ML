"""
Handler for Key Encoder training experiments.

Orchestrates the training of the Key Encoder model for encryption key analysis.
"""

from loguru import logger
from src.experiments.base import ExperimentHandler
from src.meta_learning.train_key_encoder import train_key_encoder_poc
from src.utils.constansts import REPORT_PATH


class KeyEncoderTrainingHandler(ExperimentHandler):
    """
    Handler for Key Encoder training experiments.

    Runs the key encoder proof-of-concept training with configuration
    from the main config system and command-line arguments.
    """

    def __init__(
        self,
        report_path: str = REPORT_PATH,
        num_keys: int = 500,
        num_calibration_pairs: int = 50,
        embedding_dim: int = 64,
        output_embedding_dim: int = 256,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        output_dir: str = None,
    ):
        super().__init__(experiment_name="key_encoder_training", report_path=report_path)
        self.num_keys = num_keys
        self.num_calibration_pairs = num_calibration_pairs
        self.embedding_dim = embedding_dim
        self.output_embedding_dim = output_embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = output_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run_experiment(self):
        """
        Run key encoder training with stored configuration.

        Uses parameters passed during initialization or defaults.
        """
        logger.info("=" * 60)
        logger.info("Starting Key Encoder Training")
        logger.info("=" * 60)

        logger.info(f"Key Encoder Configuration:")
        logger.info(f"  - Number of keys: {self.num_keys}")
        logger.info(f"  - Calibration pairs per key: {self.num_calibration_pairs}")
        logger.info(f"  - Input embedding dimension: {self.embedding_dim}")
        logger.info(f"  - Output embedding dimension: {self.output_embedding_dim}")
        logger.info(f"  - Training epochs: {self.num_epochs}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        if self.output_dir:
            logger.info(f"  - Output directory: {self.output_dir}")

        try:
            # Run the key encoder training
            train_key_encoder_poc(
                num_keys=self.num_keys,
                num_calibration_pairs=self.num_calibration_pairs,
                embedding_dim=self.embedding_dim,
                output_embedding_dim=self.output_embedding_dim,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                output_dir=self.output_dir
            )
            logger.info("Key Encoder Training Completed Successfully")

        except Exception as e:
            logger.error(f"Key Encoder Training Failed: {e}")
            raise