import keras, tensorflow as tf
import gc
from loguru import logger
from src.cloud.base import CloudModel
from src.cloud.vision import (XceptionCloudModel,
      ResNetEmbeddingCloudModel,
      VGG16CloudModel, VGG16Cifer10CloudModel, VGG16Cifar100CloudModel,
      InceptionCloudModel, EfficientNetCloudModel, DenseNetCloudModel,
        MobileNetCloudModel
      )
from src.cloud.tabular import EnsembleCloudModel, TabularCloudModel, NeuralNetCloudModel
from src.cloud.llm import SequenceClassificationLLMCloudModel, BertLLMCloudModel

CLOUD_MODELS = {
    NeuralNetCloudModel.name: NeuralNetCloudModel,
    TabularCloudModel.name: TabularCloudModel,
    EnsembleCloudModel.name: EnsembleCloudModel,
    XceptionCloudModel.name: XceptionCloudModel,
    VGG16Cifer10CloudModel.name: VGG16Cifer10CloudModel,
    VGG16CloudModel.name: VGG16CloudModel,
    VGG16Cifar100CloudModel.name: VGG16Cifar100CloudModel,
    EfficientNetCloudModel.name: EfficientNetCloudModel,
    InceptionCloudModel.name: InceptionCloudModel,
    DenseNetCloudModel.name: DenseNetCloudModel,
    MobileNetCloudModel.name: MobileNetCloudModel,
    SequenceClassificationLLMCloudModel.name: SequenceClassificationLLMCloudModel,
    BertLLMCloudModel.name: BertLLMCloudModel
}

class CloudModelManager:
    def __init__(self):
        self.current_model_name = None
        self.current_model = None
        self._active_models = {}  # Using a dictionary: model_id -> model_object

    def __enter__(self):
        """Called when entering the 'with' block, marks the start of a run."""
        logger.info("[ModelRunManager] Starting a new model run.")
        # We don't clear models here; they are cleared on __exit__
        # This ensures that if the manager instance persists and models were added
        # outside a 'with' block, they are cleared at the end of the *next* run.
        return self  # Makes the manager instance available as the 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'with' block, marks the end of a run."""
        logger.warning(f"[ModelRunManager] Model run finished. Clearing {len(self._active_models)} active model(s).")

        # Get a list of keys to avoid issues if modification happened during iteration (though not here)
        model_ids_to_clear = list(self._active_models.keys())

        for model_id in model_ids_to_clear:
            if model_id in self._active_models:  # Check if still present (good practice)
                model_instance = self._active_models.pop(model_id)
                logger.info(f"  [ModelRunManager] Removed '{model_instance.model_id}' from active list.")
                del model_instance

        collected_count = gc.collect()
        keras.backend.clear_session()
        logger.info(
            f"  [ModelRunManager] gc.collect() ran, {collected_count} objects collected by GC in this cycle (system-wide).")

        # To propagate an exception if one occurred within the 'with' block, return False or None.
        # To suppress an exception, return True.
        return False

    def load_new_model(self, model_id: str):
        """
        Simulates a new model arriving and being loaded into memory.
        If the model (by ID) is already loaded, it returns the existing instance.
        """
        if model_id in self._active_models:
            return self._active_models[model_id]
        else:
            logger.info(f"  [ModelRunManager] New model '{model_id}' arrived. Loading into memory...")
            new_model = CLOUD_MODELS[model_id]()
            self._active_models[model_id] = new_model
            logger.info(f"  [ModelRunManager] Model '{model_id}' loaded. Active models: {len(self._active_models)}.")
            return new_model


    def predict(self, model_name, batch):
        model = self.load_new_model(model_name)
        # Ensure input is a tensor
        batch_tensor = tf.convert_to_tensor(batch)
        return model.predict(batch_tensor)