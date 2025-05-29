import numpy as np
from keras.src.utils import to_categorical

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from keras.src.layers import Dense, Dropout, Input, BatchNormalization
from keras.src.models import Model
import pathlib
from src.utils.constansts import DATASETS_PATH
from src.utils.config import config

DATASET_DIR = pathlib.Path(DATASETS_PATH)


class DataSplitter:

    def __init__(self):
        self.split_ratio = config.dataset_config.split_ratio


class RawDataset:
    name: str

    def __init__(self, **kwargs):

        self.X, self.y = None, None
        self.sample_split = config.dataset_config.split_ratio
        self.baseline_model = config.iim_config.name
        self.name = None
        self.metadata = {}

    def get_n_classes(self):
        return len(np.unique(self.y))

    def get_number_of_features(self):
        return self.X.shape[1]

    def get_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


    def k_fold_iterator(self, n_splits=10, shuffle=True, random_state=None):
        """Yields train and test splits for K-Fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            _, X_sample, _, y_sample = train_test_split(
                X_train, y_train, test_size=self.sample_split, stratify=y_train, random_state=42
            )

            yield X_train.values, X_test.values, X_sample.values, y_sample.values, y_train.values, y_test.values


    def get_cloud_model_baseline(self, X_train, X_test, y_train, y_test):
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            f1 = f1_score(y_test, y_pred, average='weighted')
        except Exception:
            f1 = -1

        return acc, f1


    def get_baseline(self, X_train, X_test, y_train, y_test):
        """
        The method trains a model on the raw dataset to get a baseline
        on the raw dataset.
        """

        # if self.baseline_model == XGBOOST_BASELINE:
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        # else:
        #     with tf.device(CPU_DEVICE):
        #
        #         clf = InternalInferenceModelFactory.get_model(
        #             num_classes=self.get_n_c
        #             input_shape=self.get_number_of_features(),  # Only give the number of features
        #         )
        #         y_train = to_categorical(y_train)
        #
        #         # Dense networks run faster on CPU
        #         clf.fit(X_train, y_train)
        #
        #         preds = clf.predict(X_test)
        #
        acc = accuracy_score(y_test, preds)
        try:
            f1 = f1_score(y_test, preds, average='weighted')
        except Exception:
            f1 = -1

        return acc, f1


    def _get_model(self, X_train, y_train):
        inputs = Input(shape=(X_train.shape[1],))  # Dynamic input shape

        # Define the hidden layers
        x = BatchNormalization()(inputs)
        x = Dense(units=128, activation='leaky_relu')(x)
        x = Dropout(config.neural_net_config.dropout)(x)

        # x = Dense(units=64, activation='leaky_relu')(x)
        # x = Dropout(0.3)(x)

        # Define the output layer
        outputs = Dense(units=len(np.unique(y_train)), activation='softmax')(x)

        # Create the model
        clf = Model(inputs=inputs, outputs=outputs)

        # Compile the model with F1 Score
        clf.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      )

        return clf