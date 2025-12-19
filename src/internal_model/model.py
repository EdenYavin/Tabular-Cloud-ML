from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.src.models import Model
from keras.src.layers import (
    Dense, Dropout, Input, BatchNormalization, concatenate, LSTM,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Reshape, Lambda
)
from keras.src.metrics import F1Score, AUC
from keras.src import regularizers
import numpy as np
import tensorflow as tf

from src.internal_model.base import NeuralNetworkInternalModel
from src.utils.config import config
from src.utils.constansts import IIM_MODELS

models = {
    IIM_MODELS.XGBOOST.value: XGBClassifier,
}


class TransformerIIM(NeuralNetworkInternalModel):
    """
    Robust Attention-based IIM that automatically adapts to:
    - Different Embeddings (DINO=768 / CLIP=512)
    - Cloud Vectors (Present=1000 / Absent=0)
    - Raw Features (Present / Absent)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "transformer_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # -------------------------------------------------------------------------
        # 1. Determine Embedding Dimension (DINO vs CLIP)
        # -------------------------------------------------------------------------
        # Check kwargs first, then config, then default to DINO (768)
        if "embedding_dim" in kwargs:
            self.embedding_dim = kwargs["embedding_dim"]
        elif hasattr(config.experiment_config, "triangulation_embedding_name"):
            name = config.experiment_config.triangulation_embedding_name.lower()
            if "dino" in name:
                self.embedding_dim = 768
            elif "clip" in name:
                self.embedding_dim = 512
            else:
                self.embedding_dim = 768
        else:
            self.embedding_dim = 768  # Default

        # -------------------------------------------------------------------------
        # 2. Determine Cloud Vector Size
        # -------------------------------------------------------------------------
        # If cloud models are used in config, assume 1000 (ImageNet/Xception), else 0
        if config.cloud_config.names:
            self.cloud_vector_size = kwargs.get("cloud_vector_size", 1000)
        else:
            self.cloud_vector_size = 0

        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(self.input_shape_total,))

        # -------------------------------------------------------------------------
        # 3. Dynamic Parsing of Input Vector
        # -------------------------------------------------------------------------
        # Input Structure: [Raw Features (Optional) | Triangulation Seq | Cloud Vector (Optional)]

        # Step A: Identify Cloud Part
        # We assume Cloud Vector is always at the END if present
        remainder_size = self.input_shape_total - self.cloud_vector_size

        # Safety Check: If subtracting cloud makes size negative, assume NO cloud
        if remainder_size < 0:
            logger.warning(
                f"Input size {self.input_shape_total} is smaller than expected cloud vector {self.cloud_vector_size}. Assuming NO Cloud vector.")
            self.cloud_vector_size = 0
            remainder_size = self.input_shape_total

        # Step B: Identify Raw Features vs Triangulation
        # The Triangulation part MUST be a multiple of embedding_dim (e.g. N * 768)
        # Any 'leftover' bytes are assumed to be Raw Features (e.g., 20 dims for 'ring' dataset)
        raw_feature_size = remainder_size % self.embedding_dim
        triangulation_size = remainder_size - raw_feature_size

        seq_length = triangulation_size // self.embedding_dim

        logger.info(f"TransformerIIM Structure Detected: "
                    f"Total={self.input_shape_total} | "
                    f"Raw={raw_feature_size} | "
                    f"Triangulation={triangulation_size} ({seq_length}x{self.embedding_dim}) | "
                    f"Cloud={self.cloud_vector_size}")

        # -------------------------------------------------------------------------
        # 4. Slicing
        # -------------------------------------------------------------------------
        # Use Lambda layers to slice the flat input tensor

        # Slice 1: Raw Features (Start)
        if raw_feature_size > 0:
            raw_part = Lambda(lambda x: x[:, :raw_feature_size])(inputs)
            start_triang = raw_feature_size
        else:
            raw_part = None
            start_triang = 0

        # Slice 2: Triangulation (Middle)
        end_triang = start_triang + triangulation_size
        triangulation_part = Lambda(lambda x: x[:, start_triang:end_triang])(inputs)

        # Slice 3: Cloud Vector (End)
        if self.cloud_vector_size > 0:
            cloud_part = Lambda(lambda x: x[:, end_triang:])(inputs)
        else:
            cloud_part = None

        # -------------------------------------------------------------------------
        # 5. Transformer Logic (Attention on Triangulation)
        # -------------------------------------------------------------------------
        # Reshape to (Batch, Seq_Len, Emb_Dim)
        x_seq = Reshape((seq_length, self.embedding_dim))(triangulation_part)

        # Self-Attention
        attn_output = MultiHeadAttention(num_heads=8, key_dim=self.embedding_dim)(x_seq, x_seq)
        x_seq = LayerNormalization(epsilon=1e-6)(x_seq + attn_output)

        # FFN
        ffn = Dense(self.embedding_dim, activation="relu")(x_seq)
        ffn = Dropout(0.1)(ffn)
        x_seq = LayerNormalization(epsilon=1e-6)(x_seq + ffn)

        # Pooling to vector
        x_emb = GlobalAveragePooling1D()(x_seq)

        # -------------------------------------------------------------------------
        # 6. Fusion (Concatenate Raw + Attended_Triangulation + Cloud)
        # -------------------------------------------------------------------------
        components_to_concat = [x_emb]

        if raw_part is not None:
            components_to_concat.insert(0, raw_part)  # Prepend Raw
        if cloud_part is not None:
            components_to_concat.append(cloud_part)  # Append Cloud

        if len(components_to_concat) > 1:
            combined = concatenate(components_to_concat)
        else:
            combined = x_emb

        # -------------------------------------------------------------------------
        # 7. Classification Head
        # -------------------------------------------------------------------------
        x = BatchNormalization()(combined)
        x = Dense(256, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='leaky_relu')(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])

        return model

class DenseInternalModel(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "neural_network"
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def get_model(self, num_classes, input_shape):
        # Build the model
        inputs = Input(shape=(input_shape,))  # Dynamic input shape

        # Define the hidden layers
        x = BatchNormalization()(inputs)
        x = Dense(units=128, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Define the output layer
        outputs = Dense(units=num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])
        return model


class BiggerDense(DenseInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "big_neural_network"

    def get_model(self, num_classes, input_shape):
        # Build the model
        inputs = Input(shape=(input_shape,))  # Dynamic input shape

        # Define the hidden layers
        x = BatchNormalization()(inputs)
        x = Dense(units=1024, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Dense(units=512, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Dense(units=256, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        x = BatchNormalization()(x)
        x = Dense(units=128, activation='leaky_relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Define the output layer
        outputs = Dense(units=num_classes, activation='softmax')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])
        return model


class LSTMIIM(NeuralNetworkInternalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "lstm"
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def get_model(self, num_classes, input_shape):

        inputs = Input(shape=(1, input_shape))
        # LSTM layers with dropout and recurrent dropout
        x = LSTM(units=1024, return_sequences=True)(inputs)

        x = LSTM(units=512, return_sequences=True)(x)

        x = LSTM(units=256, return_sequences=True)(x)

        x = LSTM(units=128, return_sequences=True)(x)

        x = LSTM(units=64, return_sequences=False)(x)

        # Dense layers with dropout
        x = Dense(32, activation='leaky_relu')(x)
        x = Dense(16, activation='leaky_relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])
        return model

    @staticmethod
    def prepare_data_for_training(X,y):
        return tf.reshape(X, [-1, 1, X.shape[-1]]), y

    def fit(self, X, y, validation_data=None):
        X = X.reshape(-1, 1, X.shape[1])
        if validation_data:
            X_val, y_val = validation_data
            X_val = X_val.reshape(-1, 1, X_val.shape[1])
            validation_data = (X_val, y_val)

        super().fit(X, y, validation_data=validation_data)

    def evaluate(self, X, y, metrics=None):
        X = X.reshape(-1, 1, X.shape[1])
        return super().evaluate(X, y, metrics=metrics)


class DoubleDenseInternalModel(NeuralNetworkInternalModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "neural_network"
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def get_model(self, num_classes, input_shape):
        inputs_sub_networks = []

        input_shape_a, input_shape_b = input_shape
        input_a = Input(shape=(input_shape_a,))

        x = Dense(input_shape_a // 2, activation="relu", kernel_regularizer=regularizers.L2(0.1), bias_regularizer=regularizers.L2(0.01))(
            input_a)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dropout(0.3)(x)
        # x = Dense(input_shape_a / 2, activation="relu")(x)
        x = Model(inputs=input_a, outputs=x)

        inputs_sub_networks.append(x)

        input_b = Input(shape=(input_shape_b,))
        # the second branch operates on the second input
        y = Dense(input_shape_b // 4, activation="relu", kernel_regularizer=regularizers.L2(0.1),  bias_regularizer=regularizers.L2(0.01))(
            input_b)
        y = BatchNormalization(momentum=0.7)(y)
        y = Dropout(0.3)(y)
        y = Model(inputs=input_b, outputs=y)

        inputs_sub_networks.append(y)

        combined = concatenate([k.output for k in inputs_sub_networks])

        m = Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.L2(0.1),
                  bias_regularizer=regularizers.L2(0.1))(combined)

        model = Model(inputs=[k.input for k in inputs_sub_networks], outputs=m)
        # Compile the model with F1 Score
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', F1Score()]
                      )

        return model

class StackingInternalModel:

    name: str

    def __init__(self, **kwargs):
        self.models = None
        self.final_model = None


    def fit(self, X: list, y):
        assert len(X) == len(self.models), "Number of datasets, targets, and models must be the same"

        # Fit each model on its corresponding dataset
        for i, x in enumerate(X):
            self.models[i].fit(x, y)

        # Collect predictions from each model
        meta_features = []
        for i, x in enumerate(X):
            preds = self.models[i].predict_proba(x)
            meta_features.append(preds)

        # Stack predictions horizontally (axis=1) to form the meta-features
        meta_features = np.hstack(meta_features)

        # Fit the final model on the meta-features
        self.final_model.fit(meta_features, y)  # Assuming y is the target for the final model


    def predict(self, X):
        # Collect predictions from each model
        meta_features = []
        for i, x in enumerate(X):
            preds = self.models[i].predict_proba(x)
            meta_features.append(preds)

            # Stack predictions horizontally (axis=1) to form the meta-features
        meta_features = np.hstack(meta_features)

        # Predict using the final model
        return self.final_model.predict(meta_features)

    def evaluate(self, X, y):

        pred = self.predict(X)
        if len(y.shape) == 2 and len(pred.shape) == 1:
            y = y.argmax(axis=1)
        return accuracy_score(y, pred), f1_score(y, pred, average='weighted')



class StackingDenseInternalModel(StackingInternalModel):

    name = "neural_network_stacking_iim"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_models = len(config.cloud_config.names)
        self.models = [DenseInternalModel(**kwargs) for _ in range(num_models)]

        # For the final model, we need to init it according to the correct number of inputs. The final model will need a
        # different number of inputs which is num_classes * num_models
        input_size = num_models * kwargs.get("num_classes")
        kwargs['input_shape'] = input_size
        self.final_model = DenseInternalModel(**kwargs)


class StackingXGInternalModel(StackingInternalModel):

    name = "xg_stacking_iim"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_models = len(config.cloud_config.names)
        self.models = [XGBClassifier() for _ in range(num_models)]

        # For the final model, we need to init it according to the correct number of inputs. The final model will need a
        # different number of inputs which is num_classes * num_models
        input_size = num_models * kwargs.get("num_classes")
        kwargs['input_shape'] = input_size
        self.final_model = DenseInternalModel(**kwargs)


class StackingMixedInternalModel(StackingInternalModel):
    name = "logistic_xg_boost_stacking_iim"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_cloud_models = len(config.cloud_config.names)
        self.nn_models = [DenseInternalModel(**kwargs) for _ in range(num_cloud_models)]
        self.xg_models = [XGBClassifier() for _ in range(num_cloud_models)]
        self.ll_models = [LogisticRegression() for _ in range(num_cloud_models)]

        num_models = len(self.nn_models) + len(self.xg_models) + len(self.ll_models)
        # For the final model, we need to init it according to the correct number of inputs. The final model will need a
        # different number of inputs which is num_classes * num_models
        input_size = num_models * kwargs.get("num_classes")
        kwargs['input_shape'] = input_size
        self.final_model = DenseInternalModel(**kwargs)

    def fit(self, X: list, y):

        # # Fit each model on its corresponding dataset
        # for i, x in enumerate(X):
        #     self.ll_models[i].fit(x, np.argmax(y, axis=1))

        for i, x in enumerate(X):
            self.xg_models[i].fit(x, y)

        for i, x in enumerate(X):
            self.nn_models[i].fit(x, y)

        # Collect predictions from each model
        meta_features = []
        for i, x in enumerate(X):
            preds = self.ll_models[i].predict_proba(x)
            meta_features.append(preds)

        for i, x in enumerate(X):
            preds = self.xg_models[i].predict_proba(x)
            meta_features.append(preds)

        for i, x in enumerate(X):
            preds = self.nn_models[i].predict_proba(x)
            meta_features.append(preds)

        # Stack predictions horizontally (axis=1) to form the meta-features
        meta_features = np.hstack(meta_features)

        # Fit the final model on the meta-features
        self.final_model.fit(meta_features, y)  # Assuming y is the target for the final model

    def predict(self, X):

        # Collect predictions from each model
        meta_features = []
        # for i, x in enumerate(X):
        #     preds = self.ll_models[i].predict_proba(x)
        #     meta_features.append(preds)

        for i, x in enumerate(X):
            preds = self.xg_models[i].predict_proba(x)
            meta_features.append(preds)

        for i, x in enumerate(X):
            preds = self.nn_models[i].predict_proba(x)
            meta_features.append(preds)

        # Stack predictions horizontally (axis=1) to form the meta-features
        meta_features = np.hstack(meta_features)

        # Predict using the final model
        return self.final_model.predict(meta_features)



