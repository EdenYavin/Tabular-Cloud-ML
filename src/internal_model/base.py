import pickle

from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from keras.src.models import Model
import tensorflow as tf
from loguru import logger
from src.utils.config import config
from src.utils.helpers import plot_history


class TabularInternalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.name = "xgboost"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred), f1_score(y, pred, average='weighted')


class NeuralNetworkInternalModel(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        self.batch_size = config.iim_config.neural_net_config.batch_size
        self.dropout_rate = config.iim_config.neural_net_config.dropout
        self.epochs = config.iim_config.neural_net_config.epochs
        self.model: Model = None
        self.history = None

    @staticmethod
    def prepare_data_for_training(X,y):
        return X, y

    def fit(self, X, y, validation_data=None):
        tf.debugging.set_log_device_placement(True)
        with tf.device('/GPU:0'):
            logger.info(f'Using GPU: {list(filter(lambda d: "GPU:0" in d.name, tf.config.list_physical_devices()))}')
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True, start_from_epoch=5)
            self.history = self.model.fit(X, y,
                           validation_data=validation_data, epochs=self.epochs,
                           batch_size=config.iim_config.neural_net_config.batch_size,
                           verbose=2,
                           callbacks=[lr_scheduler])

    def save_history(self, filename):
        logger.info(f'saving history to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(self.history.history, f)

    def plot_history(self, filename=None, title=None):
        logger.info(f'plotting history to {filename}')
        plot_history(history=self.history,filename=filename, title=title)

    def predict(self, X):
        prediction = self.model.predict(X)
        return np.argmax(prediction, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, metrics=None):
        if metrics is None:
            metrics = ["accuracy"]

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        pred = self.predict(X)
        metrics_results = {}
        for metric in metrics:
            if metric == "accuracy":
                metrics_results[metric] = accuracy_score(y, pred)
            elif metric == "f1_score":
                metrics_results[metric] = f1_score(y, pred, average='weighted')
            elif metric == "auc":
                metrics_results[metric] = roc_auc_score(y, pred)
        return metrics_results


