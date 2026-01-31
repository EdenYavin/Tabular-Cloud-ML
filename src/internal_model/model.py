from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.src.layers import TimeDistributed, RepeatVector, Activation
from keras.src.models import Model, Sequential
from keras.src.layers import (
    Dense, Dropout, Input, BatchNormalization, concatenate, LSTM, Flatten,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Reshape, Lambda, Multiply, Add
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


class DeepSetsReconstructionIIM(NeuralNetworkInternalModel):
    """
    Implements Deep Sets + Reconstruction with GAN-style Alternating Training.
    Architecture: Standard Concatenation (No FiLM).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "deep_sets_gan_concat_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # --- Dimensions ---
        self.embedding_dim = 768 if "dino" in config.encoder_config.embedding else 512
        self.n_anchors = config.experiment_config.n_triangulation_samples
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.cloud_vector_size = 1000 * self.num_cloud_models

        # --- Sizing ---
        self.sample_size = self.embedding_dim
        self.encrypted_anchors_size = self.n_anchors * self.embedding_dim

        known_parts_size = self.sample_size + self.encrypted_anchors_size + self.cloud_vector_size
        self.plaintext_anchors_size = self.input_shape_total - known_parts_size

        if self.plaintext_anchors_size <= 0:
            raise ValueError(f"Input shape {self.input_shape_total} is too small.")

        self.target_dim = self.plaintext_anchors_size // self.n_anchors

        # We use a custom model class that overrides train_step
        self.model = self.build_gan_model()

        # --- FIX: COMPILE THE MODEL IMMEDIATELY ---
        # We must pass an optimizer INSTANCE (not string) for the custom train_step
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy'
        )

    def build_gan_model(self):
        # 1. Encoder (Deep Sets): {q_i} -> c
        encoder = Sequential([
            Input(shape=(self.n_anchors, self.embedding_dim)),
            TimeDistributed(Sequential([
                Dense(512), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.1),
                Dense(256), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.1)
            ])),
            GlobalAveragePooling1D(),
            Dense(256), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.1),
            Dense(128), BatchNormalization(), tf.keras.layers.LeakyReLU(alpha=0.1)  # Context c (Size 128)
        ], name="encoder_deepsets")

        # 2. Decoder (T): Concatenate(q_i, c) -> p_i
        # Functional API for concatenation
        decoder_input_anchor = Input(shape=(self.embedding_dim,))
        decoder_input_context = Input(shape=(128,))

        dec_concat = concatenate([decoder_input_anchor, decoder_input_context])

        x = Dense(512)(dec_concat)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        dec_output = Dense(self.target_dim, activation='linear')(x)

        decoder = Model(inputs=[decoder_input_anchor, decoder_input_context], outputs=dec_output, name="decoder_T")

        # 3. Classifier (SIN): Concatenate(q_x, c, cloud) -> Label
        input_qx = Input(shape=(self.embedding_dim,))
        input_c = Input(shape=(128,))
        input_cloud = Input(shape=(self.cloud_vector_size,)) if self.cloud_vector_size > 0 else None

        features = [input_qx, input_c]
        if input_cloud is not None:
            # Optional: project cloud to smaller dim
            cloud_proj = Dense(256, activation='leaky_relu')(input_cloud)
            features.append(cloud_proj)

        fused = concatenate(features)

        x = Dense(512, activation='leaky_relu')(fused)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='leaky_relu')(x)
        class_output = Dense(self.num_classes, activation='softmax')(x)

        inputs_classifier = [input_qx, input_c]
        if input_cloud is not None:
            inputs_classifier.append(input_cloud)

        classifier = Model(inputs=inputs_classifier, outputs=class_output, name="classifier_head")

        # 4. Wrap in GAN Model
        return DeepSetsGANModel(
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
            input_shape_total=self.input_shape_total,
            dims={
                'sample': self.sample_size,
                'enc_anchors': self.encrypted_anchors_size,
                'cloud': self.cloud_vector_size,
                'plain_anchors': self.plaintext_anchors_size,
                'n_anchors': self.n_anchors,
                'emb_dim': self.embedding_dim,
                'target_dim': self.target_dim,
                'context_dim': 128
            }
        )


# --- The Custom Training Logic ---
class DeepSetsGANModel(Model):
    def __init__(self, encoder, decoder, classifier, input_shape_total, dims, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.dims = dims
        self.input_shape_total = input_shape_total

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.auc_tracker = AUC(multi_label=False, name="auc")

    def compile(self, optimizer, loss, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.class_loss_fn = loss
        self.lambda_recon = 50.0

    def call(self, inputs, training=False):
        # 1. Slice
        q_x, q_anchors, cloud, _ = self._slice_inputs(inputs)

        # 2. Context
        c = self.encoder(q_anchors, training=training)

        # 3. Classify
        q_x_vec = tf.reshape(q_x, (-1, self.dims['emb_dim']))

        classifier_inputs = [q_x_vec, c]
        if cloud is not None:
            classifier_inputs.append(cloud)

        return self.classifier(classifier_inputs, training=training)

    def _slice_inputs(self, inputs):
        cursor = 0
        q_x_flat = inputs[:, cursor: cursor + self.dims['sample']]
        cursor += self.dims['sample']

        q_anchors_flat = inputs[:, cursor: cursor + self.dims['enc_anchors']]
        cursor += self.dims['enc_anchors']

        if self.dims['cloud'] > 0:
            cloud = inputs[:, cursor: cursor + self.dims['cloud']]
            cursor += self.dims['cloud']
        else:
            cloud = None

        p_anchors_flat = inputs[:, cursor: cursor + self.dims['plain_anchors']]

        q_anchors = tf.reshape(q_anchors_flat, (-1, self.dims['n_anchors'], self.dims['emb_dim']))
        p_anchors = tf.reshape(p_anchors_flat, (-1, self.dims['n_anchors'], self.dims['target_dim']))

        return q_x_flat, q_anchors, cloud, p_anchors

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # 1. Slice Inputs
        q_x, q_anchors, cloud, p_anchors_target = self._slice_inputs(x)

        # ============================================================
        # PHASE 1: Train Context (Encoder + Decoder) - FREEZE CLASSIFIER
        # ============================================================
        with tf.GradientTape() as tape_recon:
            # Generate Context
            c = self.encoder(q_anchors, training=True)

            # Prepare Decoder Inputs
            c_repeated = tf.repeat(tf.expand_dims(c, 1), self.dims['n_anchors'], axis=1)

            # Flatten for functional model
            q_anchors_flat = tf.reshape(q_anchors, (-1, self.dims['emb_dim']))
            c_repeated_flat = tf.reshape(c_repeated, (-1, self.dims['context_dim']))

            # Reconstruct
            p_anchors_pred_flat = self.decoder([q_anchors_flat, c_repeated_flat], training=True)
            p_anchors_pred = tf.reshape(p_anchors_pred_flat, (-1, self.dims['n_anchors'], self.dims['target_dim']))

            # Loss
            recon_loss = tf.reduce_mean(tf.square(p_anchors_target - p_anchors_pred))
            total_recon_loss = recon_loss * self.lambda_recon

        trainable_vars_recon = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads_recon = tape_recon.gradient(total_recon_loss, trainable_vars_recon)
        self.optimizer.apply_gradients(zip(grads_recon, trainable_vars_recon))

        # ============================================================
        # PHASE 2: Train Classifier - FREEZE CONTEXT
        # ============================================================
        with tf.GradientTape() as tape_class:
            # Get Fixed Context
            c_fixed = self.encoder(q_anchors, training=False)

            # Classifier Inputs
            q_x_vec = tf.reshape(q_x, (-1, self.dims['emb_dim']))
            classifier_inputs = [q_x_vec, c_fixed]
            if cloud is not None:
                classifier_inputs.append(cloud)

            # Predict
            y_pred = self.classifier(classifier_inputs, training=True)

            # Loss
            class_loss = self.class_loss_fn(y, y_pred, sample_weight=sample_weight)

        trainable_vars_class = self.classifier.trainable_variables
        grads_class = tape_class.gradient(class_loss, trainable_vars_class)
        self.optimizer.apply_gradients(zip(grads_class, trainable_vars_class))

        # Update Metrics
        self.loss_tracker.update_state(class_loss + total_recon_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.acc_tracker.update_state(y, y_pred)
        self.auc_tracker.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "auc": self.auc_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_loss_tracker, self.class_loss_tracker, self.acc_tracker, self.auc_tracker]

class oldDeepSetsReconstructionIIM(NeuralNetworkInternalModel):
    """
    Implements the Deep Sets + Reconstruction architecture.
    UPDATED: Significantly larger capacity for Phi, Rho, and T networks to improve context learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "deep_sets_reconstruction_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # --- Dimensions ---
        self.embedding_dim = 768 if "dino" in config.encoder_config.embedding else 512
        self.n_anchors = config.experiment_config.n_triangulation_samples
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.cloud_vector_size = 1000 * self.num_cloud_models

        # --- Calculate Vector Partitions ---
        self.sample_size = self.embedding_dim
        self.encrypted_anchors_size = self.n_anchors * self.embedding_dim

        known_parts_size = self.sample_size + self.encrypted_anchors_size + self.cloud_vector_size
        self.plaintext_anchors_size = self.input_shape_total - known_parts_size

        if self.plaintext_anchors_size <= 0:
            raise ValueError(f"Input shape {self.input_shape_total} is too small.")

        self.target_dim = self.plaintext_anchors_size // self.n_anchors
        logger.info(f"DeepSetsIIM: Detected Target (Raw) Anchor Dim: {self.target_dim}")

        self.lambda_anchor = 50
        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(self.input_shape_total,), name="total_input")

        # === 1. Slicing with SAFE LAMBDAS ===
        cursor = 0

        # A. Encrypted Sample
        start_qx = cursor
        end_qx = cursor + self.sample_size
        q_x_flat = Lambda(lambda x, s=start_qx, e=end_qx: x[:, s:e], name="slice_qx")(inputs)
        cursor += self.sample_size

        # B. Encrypted Anchors
        start_qi = cursor
        end_qi = cursor + self.encrypted_anchors_size
        q_anchors_flat = Lambda(lambda x, s=start_qi, e=end_qi: x[:, s:e], name="slice_qi")(inputs)
        cursor += self.encrypted_anchors_size

        # C. Cloud Vector
        if self.cloud_vector_size > 0:
            start_cloud = cursor
            end_cloud = cursor + self.cloud_vector_size
            cloud_vector = Lambda(lambda x, s=start_cloud, e=end_cloud: x[:, s:e], name="slice_cloud")(inputs)
            cursor += self.cloud_vector_size
        else:
            cloud_vector = None

        # D. Plaintext Anchors
        start_pi = cursor
        end_pi = cursor + self.plaintext_anchors_size
        p_anchors_flat = Lambda(lambda x, s=start_pi, e=end_pi: x[:, s:e], name="slice_pi")(inputs)

        # === 2. Reshaping ===
        q_anchors = Reshape((self.n_anchors, self.embedding_dim), name="reshape_qi")(q_anchors_flat)
        p_anchors = Reshape((self.n_anchors, self.target_dim), name="reshape_pi")(p_anchors_flat)

        # === 3. Deep Sets (E) - Increased Capacity ===
        # Phi Network: Process each anchor to higher dimension features
        phi = Sequential([
            Dense(512, activation='linear'),  # Increased from 256
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(256, activation='linear'),  # Added layer
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(128, activation='linear'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1)
        ], name="phi_network")

        phi_out = TimeDistributed(phi)(q_anchors)

        # Sum Pooling
        sum_pooled = Lambda(lambda x: tf.reduce_sum(x, axis=1), name="sum_pooling")(phi_out)

        # Rho Network: Process sum to context
        rho = Sequential([
            Dense(256, activation='linear'),  # Increased from 128
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(128, activation='linear'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1)
            # Output dim is 128 (Context size increased)
        ], name="rho_network")

        context_c = rho(sum_pooled)

        # === 4. Reconstruction Network (T) - Significantly Larger ===
        # Goal: Reconstruct p_i from (q_i, c)

        # Repeat c for each anchor
        c_repeated = RepeatVector(self.n_anchors)(context_c)
        decoder_input = concatenate([q_anchors, c_repeated], axis=-1)

        # Decoder Network T
        # Much deeper to solve the inverse encryption problem
        decoder_T = Sequential([
            Dense(1024, activation='linear'),  # Very wide first layer
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(512, activation='linear'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(256, activation='linear'),
            BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            Dense(self.target_dim, activation='linear')  # Output raw features
        ], name="T_network")

        p_anchors_pred = TimeDistributed(decoder_T)(decoder_input)

        # === 5. SIN / Classification Head ===
        features_to_fuse = [context_c, q_x_flat]

        if cloud_vector is not None:
            # Increased cloud projection capacity
            cloud_proj = Dense(256, activation='linear')(cloud_vector)  # Increased from 128
            cloud_proj = BatchNormalization()(cloud_proj)
            cloud_proj = tf.keras.layers.LeakyReLU(alpha=0.1)(cloud_proj)
            features_to_fuse.append(cloud_proj)

        fused = concatenate(features_to_fuse)

        x = Dense(512, activation='leaky_relu')(fused)  # Increased from 256
        x = Dropout(0.4)(x)  # Slightly higher dropout for larger model
        x = Dense(256, activation='leaky_relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='leaky_relu')(x)
        outputs = Dense(self.num_classes, activation='softmax', name="class_output")(x)

        # === 6. Model & Loss ===
        model = Model(inputs=inputs, outputs=outputs)

        reconstruction_loss = tf.reduce_mean(tf.square(p_anchors - p_anchors_pred))
        model.add_loss(self.lambda_anchor * reconstruction_loss)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(multi_label=False, name='auc')]
        )

        model.add_metric(reconstruction_loss, name="anchor_recon_loss")

        return model


class GatedCloudIIM(NeuralNetworkInternalModel):
    """
    Robust Gated IIM that supports MULTIPLE cloud models.
    The gate scales automatically to the total size of the cloud vector.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "gated_cloud_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # --- FIX: Support Multiple Cloud Models ---
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.single_cloud_dim = 1000
        self.cloud_vector_size = self.single_cloud_dim * self.num_cloud_models

        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(self.input_shape_total,))

        # 1. Dynamic Slicing
        triangulation_dim = self.input_shape_total - self.cloud_vector_size

        triangulation_part = Lambda(lambda x: x[:, :triangulation_dim])(inputs)

        if self.cloud_vector_size > 0:
            cloud_part = Lambda(lambda x: x[:, triangulation_dim:])(inputs)
        else:
            cloud_part = None

        # 2. Process Triangulation
        x_tri = Dense(256, activation='leaky_relu')(triangulation_part)
        x_tri = BatchNormalization()(x_tri)
        x_tri = Dropout(0.2)(x_tri)

        features_to_fuse = [x_tri]

        # 3. Gating Mechanism
        if cloud_part is not None:
            # The gate layer naturally scales.
            # If we have 2000 cloud features, we learn 2000 gate weights.
            # This is valid because we want to gate specific classes from specific models.
            gate = Dense(self.cloud_vector_size, activation='sigmoid', name="validity_gate")(cloud_part)

            gated_cloud = Multiply(name="gated_cloud_signal")([cloud_part, gate])

            features_to_fuse.append(gated_cloud)

        # 4. Fusion
        if len(features_to_fuse) > 1:
            combined = concatenate(features_to_fuse)
        else:
            combined = features_to_fuse[0]

        # 5. Head
        x = Dense(128, activation='leaky_relu')(combined)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])

        return model


class GatedFiLMConditionedIIM(NeuralNetworkInternalModel):
    """
    State-of-the-Art IIM: Uses FiLM to process local features, then uses
    those 'clean' features to GATE the cloud vector.
    If cloud vector is noise (like in 'magic'), the Gate closes (approaches 0).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "gated_film"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # Config
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.single_cloud_dim = 1000
        self.cloud_vector_size = self.single_cloud_dim * self.num_cloud_models

        # Calibration embedding
        embedding_name = config.encoder_config.embedding.lower() if hasattr(config.encoder_config,
                                                                            'embedding') else 'dino'
        single_emb_dim = 768 if "dino" in embedding_name else 512
        n_calib_vectors = len(config.experiment_config.calibration_distributions) if hasattr(config.experiment_config,
                                                                                             'calibration_distributions') else 1
        self.calib_embedding_dim = single_emb_dim * n_calib_vectors

        self.film_hidden_dim = 512
        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(self.input_shape_total,))

        # === 1. Slicing ===
        cloud_end = self.input_shape_total
        cloud_start = cloud_end - self.cloud_vector_size
        calib_end = cloud_start
        calib_start = calib_end - self.calib_embedding_dim
        main_end = calib_start

        main_features = Lambda(lambda x: x[:, :main_end], name="slice_main")(inputs)
        calib_embedding = Lambda(lambda x: x[:, calib_start:calib_end], name="slice_calib")(inputs)

        if self.cloud_vector_size > 0:
            cloud_vector = Lambda(lambda x: x[:, cloud_start:], name="slice_cloud")(inputs)
        else:
            cloud_vector = None

        # === 2. FiLM Generation (Processing the Key) ===
        # Normalize calibration input
        calib_norm = BatchNormalization(name="calib_norm")(calib_embedding)

        # Generator
        film_gen = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.001))(calib_norm)
        film_gen = BatchNormalization()(film_gen)
        film_gen = Dropout(0.2)(film_gen)

        # Generate FiLM parameters
        gamma = Dense(self.film_hidden_dim, kernel_initializer='ones', name="gamma")(film_gen)
        beta = Dense(self.film_hidden_dim, kernel_initializer='zeros', name="beta")(film_gen)

        # === 3. Local Feature Extraction (The "Expert") ===
        x = Dense(self.film_hidden_dim, activation='linear')(main_features)
        x = BatchNormalization()(x)

        # Apply FiLM
        x = Multiply()([x, gamma])
        x = Add()([x, beta])
        x = tf.keras.layers.Activation('leaky_relu')(x)

        local_expert = Dropout(0.2)(x)  # These are the robust local features

        # === 4. Intelligent Gating ===
        if cloud_vector is not None:
            # Compress cloud vector
            c = Dense(self.film_hidden_dim, activation='leaky_relu', name="cloud_proj")(cloud_vector)
            c = BatchNormalization()(c)

            # THE FIX: Generate Gate based on LOCAL EXPERT + CALIBRATION
            # "Given what I know about the key (calib) and the data (local), is the cloud useful?"
            gate_input = concatenate([local_expert, film_gen])
            gate = Dense(self.film_hidden_dim, activation='sigmoid', name="trust_gate")(gate_input)

            # Apply Gate
            gated_cloud = Multiply(name="gated_cloud")([c, gate])

            # Fuse
            fused = concatenate([local_expert, gated_cloud])
        else:
            fused = local_expert

        # === 5. Head ===
        x = Dense(256, activation='leaky_relu')(fused)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', AUC(multi_label=False, name='auc')])

        return model

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


class FiLMConditionedIIM(NeuralNetworkInternalModel):
    """
    Robust FiLM-Conditioned IIM with stabilized training dynamics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "film_conditioned_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")

        # Configuration
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.single_cloud_dim = 1000
        self.cloud_vector_size = self.single_cloud_dim * self.num_cloud_models

        # Calibration embedding dimension
        embedding_name = config.encoder_config.embedding.lower() if hasattr(config.encoder_config,
                                                                            'embedding') else 'dino'
        single_emb_dim = 768 if "dino" in embedding_name else 512
        n_calib_vectors = len(config.experiment_config.calibration_distributions) if hasattr(config.experiment_config,
                                                                                             'calibration_distributions') else 1
        self.calib_embedding_dim = single_emb_dim * n_calib_vectors

        # Increased hidden dim for better capacity
        self.film_hidden_dim = 512
        self.model = self.get_model()

    def get_model(self):
        inputs = Input(shape=(self.input_shape_total,))

        # === STEP 1: Slicing ===
        # Input: [Main Features | Calibration | Cloud]
        cloud_end = self.input_shape_total
        cloud_start = cloud_end - self.cloud_vector_size
        calib_end = cloud_start
        calib_start = calib_end - self.calib_embedding_dim
        main_features_end = calib_start

        main_features = Lambda(lambda x: x[:, :main_features_end], name="slice_main")(inputs)
        calib_embedding = Lambda(lambda x: x[:, calib_start:calib_end], name="slice_calib")(inputs)

        if self.cloud_vector_size > 0:
            cloud_vector = Lambda(lambda x: x[:, cloud_start:], name="slice_cloud")(inputs)
        else:
            cloud_vector = None

        # === STEP 2: Stabilized FiLM Generator ===
        # Normalize calibration input first - Crucial for stability with rotating keys
        calib_norm = BatchNormalization(name="calib_norm")(calib_embedding)

        # Deeper Generator Network
        film_gen = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.001))(calib_norm)
        film_gen = BatchNormalization()(film_gen)
        film_gen = Dropout(0.2)(film_gen)
        film_gen = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.001))(film_gen)

        # Generate Parameters (Gamma=Scale, Beta=Shift)
        # Initialize Gamma to 1.0 and Beta to 0.0 to start as Identity function
        gamma = Dense(self.film_hidden_dim, kernel_initializer='ones', bias_initializer='zeros', name="gamma")(film_gen)
        beta = Dense(self.film_hidden_dim, kernel_initializer='zeros', bias_initializer='zeros', name="beta")(film_gen)

        # === STEP 3: Modulated Feature Extraction ===
        # Project main features to hidden dim
        x = Dense(self.film_hidden_dim, activation='linear', name="feat_projection")(main_features)
        x = BatchNormalization()(x)

        # --- FiLM Application ---
        # x = gamma * x + beta
        x_modulated = Multiply(name="film_scale")([x, gamma])
        x_modulated = Add(name="film_shift")([x_modulated, beta])
        x_modulated = tf.keras.layers.Activation('leaky_relu')(x_modulated)

        # Residual Connection (Optional but recommended: allows gradient flow if FiLM is bad initially)
        # We project input to match dimension if needed, or just use the modulated output
        x = Dropout(0.2)(x_modulated)

        # Second Dense Block
        x = Dense(256, activation='leaky_relu', kernel_regularizer=regularizers.L2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # === STEP 4: Cloud Fusion (Late Fusion) ===
        if cloud_vector is not None:
            # Compress cloud vector before fusion
            c = Dense(256, activation='leaky_relu', name="cloud_compress")(cloud_vector)
            c = BatchNormalization()(c)
            c = Dropout(0.2)(c)

            # Concatenate
            x = concatenate([x, c])

        # === STEP 5: Head ===
        x = Dense(128, activation='leaky_relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # CRITICAL FIX: Add clipnorm=1.0 to prevent exploding gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(multi_label=False, name='auc')]
        )

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



