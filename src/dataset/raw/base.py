import numpy as np

from src.utils.helpers import load_data, save_data
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from keras.src.layers import Dense, Dropout, Input, Attention, Concatenate, Lambda, Conv1D, Flatten, MaxPooling1D
from keras.src.models import Model
from keras.src.utils import to_categorical
from src.utils.constansts import DATASETS_PATH, XGBOOST_BASELINE, NEURAL_NET_BASELINE
import pathlib

DATASET_DIR = pathlib.Path(DATASETS_PATH)


class RawDataset:
    def __init__(self, **kwargs):

        self.X, self.y = None, None
        self.sample_split = kwargs.get("ratio")
        self.baseline_model = kwargs.get("baseline")
        self.name = None



    def get_dataset(self):
        return self.X, self.y

    def get_split(self):
        try:
            return load_data(self.name, self.sample_split)

        except FileNotFoundError:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, stratify=self.y,
                                                                random_state=42)

            _, X_sample, _, y_sample = train_test_split(X_train, y_train, test_size=self.sample_split, stratify=y_train,
                                                        random_state=42)

            X_train = X_train.values
            X_test = X_test.values
            X_sample = X_sample.values
            y_train = y_train.values
            y_test = y_test.values
            y_sample = y_sample.values

            save_data(self.name, self.sample_split,
                      [X_train, X_test, X_sample, y_train, y_test,  y_sample])

            return X_train, X_test, X_sample, y_train, y_test, y_sample

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



    def get_baseline(self, X_train, X_test, y_train, y_test):

        if self.baseline_model == XGBOOST_BASELINE:
            clf = XGBClassifier()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
        else:
            clf = self._get_model(X_train, y_train)
            y_train = to_categorical(y_train)
            clf.fit(X_train, y_train, epochs=10, batch_size=8)
            preds = np.argmax(clf.predict(X_test), axis=1)

        return accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted')


    def _get_model(self, X_train, y_train):
        inputs = Input(shape=(X_train.shape[1],))  # Dynamic input shape

        # Define the hidden layers
        x = Dense(units=128, activation='leaky_relu')(inputs)
        x = Dropout(0.3)(x)

        x = Dense(units=64, activation='leaky_relu')(x)
        x = Dropout(0.3)(x)

        # Define the output layer
        outputs = Dense(units=len(np.unique(y_train)), activation='softmax')(x)

        # Create the model
        clf = Model(inputs=inputs, outputs=outputs)

        # Compile the model with F1 Score
        clf.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      )

        return clf