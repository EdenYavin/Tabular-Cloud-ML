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


class EpochTrackerCallback(tf.keras.callbacks.Callback):
    """Callback to set training phase at the start of each epoch."""
    def on_epoch_begin(self, epoch, logs=None):
        # If T network is frozen, always train classifier only
        if getattr(self.model, 'freeze_t_network', False):
            self.model.train_reconstruction_phase.assign(False)
            logger.info(f"Epoch {epoch}: Training Classification (T-network frozen)")
        else:
            train_reconstruction = (epoch % 2 == 0)
            self.model.train_reconstruction_phase.assign(train_reconstruction)
            phase = "Reconstruction (Encoder+Decoder)" if train_reconstruction else "Classification"
            logger.info(f"Epoch {epoch}: Training {phase}")

models = {
    IIM_MODELS.XGBOOST.value: XGBClassifier,
}


class DeepSetsReconstructionIIM(NeuralNetworkInternalModel):
    """
    Implements Deep Sets + Reconstruction with GAN-style Alternating Training.

    NEW Architecture (DeepSets on Anchor Pairs):
    - Encoder (T) input: Anchor pairs (p_i, q_i) for each anchor, plus p_x
      - p_i = Sparse autoencoder embeddings (before encryption)
      - q_i = Encrypted → DINO/CLIP embedding
    - Decoder target: Reconstruct q_x (encrypted X embedding)

    Input Vector: [ p_x | p_i | q_i | cloud | q_x_target ]
    - p_x: 64 (sparse autoencoder sample embedding)
    - p_i: n_anchors * 64 (sparse autoencoder anchor embeddings)
    - q_i: n_anchors * emb_dim (encrypted anchor embeddings)
    - q_x_target: emb_dim (encrypted sample embedding - reconstruction target)
    """

    def __init__(self, pretrained_t_network_path=None, freeze_t_network=False, **kwargs):
        super().__init__(**kwargs)
        self.name = "deep_sets_gan_concat_iim"
        self.num_classes = kwargs.get("num_classes")
        self.input_shape_total = kwargs.get("input_shape")
        self.pretrained_t_network_path = pretrained_t_network_path
        self.freeze_t_network = freeze_t_network

        # --- Dimensions ---
        self.embedding_dim = 768 if "dino" in config.encoder_config.embedding else 512
        self.n_anchors = config.experiment_config.n_triangulation_samples
        self.num_cloud_models = len(config.cloud_config.names) if config.cloud_config.names else 0
        self.cloud_vector_size = 1000 * self.num_cloud_models

        # --- Calculate raw_dim from input shape ---
        # Input: [ p_x | p_i | q_i | cloud | q_x_target ]
        # Known sizes: q_i = n_anchors * emb_dim, cloud, q_x_target = emb_dim
        # Unknown: p_x = raw_dim, p_i = n_anchors * raw_dim
        # Total: raw_dim + n_anchors*raw_dim + n_anchors*emb_dim + cloud + emb_dim = input_shape
        # raw_dim * (1 + n_anchors) = input_shape - n_anchors*emb_dim - cloud - emb_dim

        q_i_size = self.n_anchors * self.embedding_dim
        q_x_target_size = self.embedding_dim
        known_size = q_i_size + self.cloud_vector_size + q_x_target_size

        remaining = self.input_shape_total - known_size
        # remaining = raw_dim * (1 + n_anchors)
        self.raw_dim = remaining // (1 + self.n_anchors)

        # --- Final Sizing ---
        self.p_x_size = self.raw_dim                              # Raw sample features
        self.p_i_size = self.n_anchors * self.raw_dim             # Raw anchor features
        self.q_i_size = self.n_anchors * self.embedding_dim       # Encrypted anchor embeddings
        self.q_x_target_size = self.embedding_dim                 # Reconstruction target

        # Verify input shape matches expected structure
        expected_size = self.p_x_size + self.p_i_size + self.q_i_size + self.cloud_vector_size + self.q_x_target_size
        logger.info(f"DeepSetsIIM: raw_dim={self.raw_dim}, emb_dim={self.embedding_dim}, n_anchors={self.n_anchors}")
        logger.info(f"DeepSetsIIM: p_x={self.p_x_size}, p_i={self.p_i_size}, q_i={self.q_i_size}, "
                   f"cloud={self.cloud_vector_size}, q_x_target={self.q_x_target_size}")
        if self.input_shape_total != expected_size:
            logger.warning(f"Input shape {self.input_shape_total} != expected {expected_size}")

        self.context_dim = 128  # Output dimension of encoder

        # We use a custom model class that overrides train_step
        self.model = self.build_gan_model()

        # Use TWO Optimizers for alternating training phases
        self.model.compile(
            optimizer_recon=tf.keras.optimizers.Adam(learning_rate=0.001),
            optimizer_class=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', AUC(multi_label=False, name='auc')]
        )

    def fit(self, X, y, validation_data=None):
        """Override fit to include epoch tracker callback."""
        from keras.src.callbacks import ReduceLROnPlateau

        tf.debugging.set_log_device_placement(True)
        with tf.device('/GPU:0'):
            logger.info(f'Using GPU: {list(filter(lambda d: "GPU:0" in d.name, tf.config.list_physical_devices()))}')

            # Create callbacks
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            epoch_tracker = EpochTrackerCallback()

            self.history = self.model.fit(
                X, y,
                validation_data=validation_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=2,
                callbacks=[lr_scheduler, epoch_tracker]
            )

    def _build_encoder(self):
        """Build the Deep Sets encoder architecture - matches TNetworkOnlyIIM."""
        anchor_pair_dim = self.raw_dim + self.embedding_dim

        # φ network inputs
        encoder_input_pairs = Input(shape=(self.n_anchors, anchor_pair_dim), name="anchor_pairs")
        encoder_input_px = Input(shape=(self.raw_dim,), name="p_x")

        # φ network: Per-anchor transformation (matches TNetworkOnlyIIM)
        x = TimeDistributed(Dense(512, name='phi_dense1'), name='td_phi_dense1')(encoder_input_pairs)
        x = TimeDistributed(BatchNormalization(name='phi_bn1'), name='td_phi_bn1')(x)
        x = TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1, name='phi_act1'), name='td_phi_act1')(x)

        x = TimeDistributed(Dense(256, name='phi_dense2'), name='td_phi_dense2')(x)
        x = TimeDistributed(BatchNormalization(name='phi_bn2'), name='td_phi_bn2')(x)
        x = TimeDistributed(tf.keras.layers.LeakyReLU(alpha=0.1, name='phi_act2'), name='td_phi_act2')(x)

        x = TimeDistributed(Dense(128, name='phi_dense3'), name='td_phi_dense3')(x)

        # Pool across anchors (permutation invariant)
        c_anchors = GlobalAveragePooling1D(name='anchor_pool')(x)  # (batch, 128)

        # Process p_x through dedicated layers (matches TNetworkOnlyIIM)
        p_x_features = Dense(256, activation='relu', name='p_x_dense1')(encoder_input_px)
        p_x_features = Dense(128, activation='relu', name='p_x_dense2')(p_x_features)  # (batch, 128)

        # Combine: c_anchors (128) + p_x_features (128) = 256
        combined = concatenate([c_anchors, p_x_features], name='encoder_concat')

        # ρ network (matches TNetworkOnlyIIM)
        context = Dense(256, name='rho_dense1')(combined)
        context = BatchNormalization(name='rho_bn1')(context)
        context = tf.keras.layers.LeakyReLU(alpha=0.1, name='rho_act1')(context)

        context = Dense(self.context_dim, name='rho_dense2')(context)
        context = BatchNormalization(name='rho_bn2')(context)
        context = tf.keras.layers.LeakyReLU(alpha=0.1, name='rho_act2')(context)

        encoder = Model(
            inputs=[encoder_input_pairs, encoder_input_px],
            outputs=context,
            name="encoder_deepsets"
        )
        return encoder

    def _build_decoder(self):
        """Build the T network decoder architecture - matches TNetworkOnlyIIM."""
        decoder_input_context = Input(shape=(self.context_dim,), name='decoder_input')

        x = Dense(256, name='decoder_dense1')(decoder_input_context)
        x = BatchNormalization(name='decoder_bn1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='decoder_act1')(x)

        x = Dense(512, name='decoder_dense2')(x)
        x = BatchNormalization(name='decoder_bn2')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name='decoder_act2')(x)

        dec_output = Dense(self.embedding_dim, activation='linear', name='decoder_output')(x)

        decoder = Model(inputs=decoder_input_context, outputs=dec_output, name="decoder_T")
        return decoder

    def _build_classifier(self):
        """Build the classifier (SIN) architecture."""
        input_c = Input(shape=(self.context_dim,))
        input_cloud = Input(shape=(self.cloud_vector_size,)) if self.cloud_vector_size > 0 else None

        features = [input_c]
        if input_cloud is not None:
            cloud_proj = Dense(256, activation='leaky_relu')(input_cloud)
            features.append(cloud_proj)

        if len(features) > 1:
            fused = concatenate(features)
        else:
            fused = features[0]

        x = Dense(512, activation='leaky_relu')(fused)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='leaky_relu')(x)
        class_output = Dense(self.num_classes, activation='softmax')(x)

        inputs_classifier = [input_c]
        if input_cloud is not None:
            inputs_classifier.append(input_cloud)

        classifier = Model(inputs=inputs_classifier, outputs=class_output, name="classifier_head")
        return classifier

    def _load_pretrained_t_network(self):
        """Load pretrained T network and extract encoder/decoder."""
        from pathlib import Path
        import json

        model_path = Path(self.pretrained_t_network_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pretrained T network not found: {model_path}")

        logger.info(f"Loading pretrained T network from: {model_path}")

        # Verify compatibility with metadata
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
            for key, expected in [('n_anchors', self.n_anchors), ('raw_dim', self.raw_dim), ('emb_dim', self.embedding_dim)]:
                saved = saved_metadata.get(key)
                if saved != expected:
                    logger.warning(f"{key} mismatch: expected={expected}, saved={saved}")

        # Load full T network
        t_network = tf.keras.models.load_model(model_path)

        # Rebuild encoder and decoder with matching architecture
        encoder = self._build_encoder()
        decoder = self._build_decoder()

        # Weight mapping for encoder: TNetworkOnlyIIM layer name -> DeepSetsIIM layer location
        encoder_layer_mapping = {
            # φ network layers (inside TimeDistributed)
            'phi_dense1': 'td_phi_dense1',
            'phi_bn1': 'td_phi_bn1',
            'phi_act1': 'td_phi_act1',
            'phi_dense2': 'td_phi_dense2',
            'phi_bn2': 'td_phi_bn2',
            'phi_act2': 'td_phi_act2',
            'phi_dense3': 'td_phi_dense3',
            # p_x processing layers (direct match)
            'p_x_dense1': 'p_x_dense1',
            'p_x_dense2': 'p_x_dense2',
            # ρ network layers (direct match)
            'rho_dense1': 'rho_dense1',
            'rho_bn1': 'rho_bn1',
            'rho_act1': 'rho_act1',
            'rho_dense2': 'rho_dense2',
            'rho_bn2': 'rho_bn2',
            'rho_act2': 'rho_act2',
        }

        # Transfer encoder weights
        loaded_count = 0
        for source_name, target_name in encoder_layer_mapping.items():
            try:
                source_layer = t_network.get_layer(source_name)
                target_layer = encoder.get_layer(target_name)

                # Handle TimeDistributed: extract inner layer
                if hasattr(target_layer, 'layer'):
                    target_layer = target_layer.layer

                target_layer.set_weights(source_layer.get_weights())
                loaded_count += 1
                logger.debug(f"Loaded encoder weight: {source_name} -> {target_name}")
            except ValueError as e:
                logger.warning(f"Failed to load encoder layer {source_name}: {e}")

        # Weight mapping for decoder (direct match)
        decoder_layer_mapping = {
            'decoder_dense1': 'decoder_dense1',
            'decoder_bn1': 'decoder_bn1',
            'decoder_act1': 'decoder_act1',
            'decoder_dense2': 'decoder_dense2',
            'decoder_bn2': 'decoder_bn2',
            'decoder_act2': 'decoder_act2',
            'decoder_output': 'decoder_output',
        }

        # Transfer decoder weights
        for source_name, target_name in decoder_layer_mapping.items():
            try:
                source_layer = t_network.get_layer(source_name)
                target_layer = decoder.get_layer(target_name)
                target_layer.set_weights(source_layer.get_weights())
                loaded_count += 1
                logger.debug(f"Loaded decoder weight: {source_name} -> {target_name}")
            except ValueError as e:
                logger.warning(f"Failed to load decoder layer {source_name}: {e}")

        total_expected = len(encoder_layer_mapping) + len(decoder_layer_mapping)
        logger.info(f"Loaded {loaded_count}/{total_expected} T network layers")

        if loaded_count < total_expected:
            logger.warning(f"Some layers failed to load - T network may not be fully initialized")
        elif loaded_count == total_expected:
            logger.info("Successfully loaded all pretrained T network weights")

        return encoder, decoder

    def _freeze_t_network(self, encoder, decoder):
        """Freeze encoder and decoder layers."""
        logger.info("Freezing T network (encoder + decoder) layers")

        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

        decoder.trainable = False
        for layer in decoder.layers:
            layer.trainable = False

        logger.info(f"Frozen encoder: {len(encoder.layers)} layers")
        logger.info(f"Frozen decoder: {len(decoder.layers)} layers")

    def build_gan_model(self):
        """Build the GAN model, optionally loading pretrained T network."""

        if self.pretrained_t_network_path:
            encoder, decoder = self._load_pretrained_t_network()
        else:
            encoder = self._build_encoder()
            decoder = self._build_decoder()

        classifier = self._build_classifier()

        # Freeze T-network if requested
        if self.freeze_t_network:
            if not self.pretrained_t_network_path:
                raise ValueError(
                    "freeze_t_network=True requires a pretrained T-Network model. "
                    "Either train a T-Network first with --experiment-to-run t_network_training "
                    "or provide --pretrained-t-network <path>."
                )
            self._freeze_t_network(encoder, decoder)

        # Wrap everything in the GAN Model
        return DeepSetsGANModel(
            encoder=encoder,
            decoder=decoder,
            classifier=classifier,
            input_shape_total=self.input_shape_total,
            freeze_t_network=self.freeze_t_network,
            dims={
                'p_x': self.p_x_size,
                'p_i': self.p_i_size,
                'q_i': self.q_i_size,
                'cloud': self.cloud_vector_size,
                'q_x_target': self.q_x_target_size,
                'n_anchors': self.n_anchors,
                'raw_dim': self.raw_dim,
                'emb_dim': self.embedding_dim,
                'context_dim': self.context_dim
            }
        )


# --- The Custom Training Logic ---
class DeepSetsGANModel(Model):
    """
    Custom Model with alternating training for DeepSets + Reconstruction.

    NEW Architecture:
    - Encoder: Takes anchor pairs (p_i, q_i) + p_x → context
    - Decoder: Takes context → reconstructs q_x
    - Classifier: Takes context + cloud → class prediction
    """

    def __init__(self, encoder, decoder, classifier, input_shape_total, dims, freeze_t_network=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.dims = dims
        self.input_shape_total = input_shape_total
        self.freeze_t_network = freeze_t_network

        # Phase tracking for alternating training (True = reconstruction, False = classification)
        self.train_reconstruction_phase = tf.Variable(True, trainable=False, dtype=tf.bool)

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.t_loss_tracker = tf.keras.metrics.Mean(name="t_loss")  # Encoder (T) loss
        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.auc_tracker = AUC(multi_label=False, name="auc")

    def compile(self, optimizer_recon, optimizer_class, loss, **kwargs):
        super().compile(loss=loss, **kwargs)
        self.optimizer_recon = optimizer_recon
        self.optimizer_class = optimizer_class
        self.class_loss_fn = loss
        self.lambda_recon = 50.0

    def call(self, inputs, training=False):
        # 1. Slice inputs
        p_x, anchor_pairs, cloud, _ = self._slice_inputs(inputs)

        # 2. Get context from encoder
        c = self.encoder([anchor_pairs, p_x], training=training)

        # 3. Classify using context + cloud
        classifier_inputs = [c]
        if cloud is not None:
            classifier_inputs.append(cloud)

        return self.classifier(classifier_inputs, training=training)

    def _slice_inputs(self, inputs):
        """
        Parse NEW input vector structure:
        [ p_x | p_i | q_i | cloud | q_x_target ]

        Where:
        - p_x, p_i are RAW tabular data (raw_dim)
        - q_i, q_x_target are EMBEDDED vectors (emb_dim)

        Returns: p_x, anchor_pairs (p_i concat q_i), cloud, q_x_target
        """
        cursor = 0

        # p_x: Sparse autoencoder sample embedding (64)
        p_x = inputs[:, cursor:cursor + self.dims['p_x']]
        cursor += self.dims['p_x']

        # p_i: Sparse autoencoder anchor embeddings (n_anchors * 64)
        p_i_flat = inputs[:, cursor:cursor + self.dims['p_i']]
        cursor += self.dims['p_i']

        # q_i: Encrypted anchor embeddings (n_anchors * emb_dim)
        q_i_flat = inputs[:, cursor:cursor + self.dims['q_i']]
        cursor += self.dims['q_i']

        # Cloud predictions (if present)
        if self.dims['cloud'] > 0:
            cloud = inputs[:, cursor:cursor + self.dims['cloud']]
            cursor += self.dims['cloud']
        else:
            cloud = None

        # q_x_target: Encrypted X embedding (emb_dim) - reconstruction target
        q_x_target = inputs[:, cursor:cursor + self.dims['q_x_target']]

        # Reshape anchors with DIFFERENT dimensions
        n_anchors = self.dims['n_anchors']
        raw_dim = self.dims['raw_dim']
        emb_dim = self.dims['emb_dim']

        p_i = tf.reshape(p_i_flat, (-1, n_anchors, raw_dim))   # (batch, n_anchors, raw_dim)
        q_i = tf.reshape(q_i_flat, (-1, n_anchors, emb_dim))   # (batch, n_anchors, emb_dim)

        # Concatenate anchor pairs: (p_i, q_i) → (batch, n_anchors, raw_dim + emb_dim)
        anchor_pairs = tf.concat([p_i, q_i], axis=-1)

        return p_x, anchor_pairs, cloud, q_x_target

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        p_x, anchor_pairs, cloud, q_x_target = self._slice_inputs(x)

        # If T network is frozen, only train classifier
        if self.freeze_t_network:
            return self._train_step_classifier_only(p_x, anchor_pairs, cloud, y, sample_weight)
        else:
            return self._train_step_alternating(p_x, anchor_pairs, cloud, q_x_target, y, sample_weight)

    def _train_step_classifier_only(self, p_x, anchor_pairs, cloud, y, sample_weight):
        """Training step with frozen T network - only train classifier."""
        class_vars = self.classifier.trainable_variables

        with tf.GradientTape() as tape:
            # Frozen encoder (no gradient tracking)
            c = self.encoder([anchor_pairs, p_x], training=False)

            # Classifier forward pass
            classifier_inputs = [c]
            if cloud is not None:
                classifier_inputs.append(cloud)

            y_pred = self.classifier(classifier_inputs, training=True)
            class_loss = self.class_loss_fn(y, y_pred, sample_weight=sample_weight)

        # Compute and apply gradients only for classifier
        grads_class = tape.gradient(class_loss, class_vars)
        self.optimizer_class.apply_gradients(zip(grads_class, class_vars))

        # Update metrics
        self.loss_tracker.update_state(class_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.acc_tracker.update_state(y, y_pred)
        self.auc_tracker.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "auc": self.auc_tracker.result(),
        }

    def _train_step_alternating(self, p_x, anchor_pairs, cloud, q_x_target, y, sample_weight):
        """Original alternating training step for reconstruction and classification."""
        # Get all trainable variables upfront (needed for tf.cond)
        recon_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        class_vars = self.classifier.trainable_variables

        # --- Compute both forward passes and gradients ---
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction forward pass
            # Encoder: (anchor_pairs, p_x) -> context
            c = self.encoder([anchor_pairs, p_x], training=True)

            # Decoder: context -> q_x prediction
            q_x_pred = self.decoder(c, training=True)

            # Reconstruction loss: MSE between predicted and actual encrypted X embedding
            recon_loss = tf.reduce_mean(tf.square(q_x_target - q_x_pred))
            total_recon_loss = recon_loss * self.lambda_recon

            # Classification forward pass (use stop_gradient on context)
            c_for_class = tf.stop_gradient(c)
            classifier_inputs = [c_for_class]
            if cloud is not None:
                classifier_inputs.append(cloud)

            y_pred = self.classifier(classifier_inputs, training=True)
            class_loss = self.class_loss_fn(y, y_pred, sample_weight=sample_weight)

        # Compute gradients for both phases
        grads_recon = tape.gradient(total_recon_loss, recon_vars)
        grads_class = tape.gradient(class_loss, class_vars)
        del tape  # Release persistent tape

        # Conditionally apply gradients based on current phase
        def apply_recon_grads():
            self.optimizer_recon.apply_gradients(zip(grads_recon, recon_vars))
            return tf.constant(0.0)

        def apply_class_grads():
            self.optimizer_class.apply_gradients(zip(grads_class, class_vars))
            return tf.constant(0.0)

        # tf.cond to apply the correct gradients based on phase
        tf.cond(
            self.train_reconstruction_phase,
            apply_recon_grads,
            apply_class_grads
        )

        # Update Metrics
        self.loss_tracker.update_state(class_loss + total_recon_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.t_loss_tracker.update_state(total_recon_loss)  # T (encoder) loss with lambda scaling
        self.class_loss_tracker.update_state(class_loss)
        self.acc_tracker.update_state(y, y_pred)
        self.auc_tracker.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "t_loss": self.t_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "auc": self.auc_tracker.result(),
        }

    def test_step(self, data):
        """
        Custom validation logic.
        Calculates both reconstruction and classification losses without updating weights.
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # 1. Slice Inputs (NEW structure)
        p_x, anchor_pairs, cloud, q_x_target = self._slice_inputs(x)

        # 2. Forward Pass (Reconstruction)
        # Encoder: (anchor_pairs, p_x) -> context
        c = self.encoder([anchor_pairs, p_x], training=False)

        # Decoder: context -> q_x prediction
        q_x_pred = self.decoder(c, training=False)

        # Calculate Recon Loss: MSE between predicted and actual encrypted X embedding
        recon_loss = tf.reduce_mean(tf.square(q_x_target - q_x_pred))
        total_recon_loss = recon_loss * self.lambda_recon

        # 3. Forward Pass (Classification)
        classifier_inputs = [c]
        if cloud is not None:
            classifier_inputs.append(cloud)

        y_pred = self.classifier(classifier_inputs, training=False)

        # Calculate Class Loss
        class_loss = self.class_loss_fn(y, y_pred, sample_weight=sample_weight)

        # 4. Update Metrics
        self.loss_tracker.update_state(class_loss + total_recon_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.t_loss_tracker.update_state(total_recon_loss)
        self.class_loss_tracker.update_state(class_loss)
        self.acc_tracker.update_state(y, y_pred)
        self.auc_tracker.update_state(y, y_pred)

        return {
            "loss": self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "t_loss": self.t_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "auc": self.auc_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_loss_tracker, self.t_loss_tracker, self.class_loss_tracker, self.acc_tracker, self.auc_tracker]


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


class FlexibleSINClassifier(NeuralNetworkInternalModel):
    """
    Flexible Single Inference Network (SIN) classifier for feature ablation studies.

    This classifier accepts variable input_shape parameter and dynamically builds
    a Dense neural network architecture at runtime. Designed for frozen T network
    ablation experiments where different feature combinations have different dimensions.

    Supported feature combinations:
    - combo1: [p_x, q_x, T_context] → 960 dims
    - combo2: [q_x, T_context] → 896 dims
    - combo3: [p_x, q_x, T_context, cloud] → 960+cloud dims
    - combo4: [q_x, T_context, cloud] → 896+cloud dims

    Architecture:
    - Input layer: Dynamic shape based on input_shape parameter
    - BatchNormalization
    - Dense(128, activation='leaky_relu')
    - Dropout(dropout_rate)
    - Output layer: Dense(num_classes, activation='softmax')

    The architecture adapts to any input dimension without code changes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "flexible_sin_classifier"
        num_classes = kwargs.get("num_classes")
        input_shape = kwargs.get("input_shape")

        # Log initialization for ablation study context
        logger.info(f"FlexibleSINClassifier initialized: input_shape={input_shape}, num_classes={num_classes}")

        # Build the model with dynamic input shape
        self.model = self.get_model(num_classes=num_classes, input_shape=input_shape)

    def get_model(self, num_classes, input_shape):
        """
        Build Dense architecture with dynamic input shape.

        Args:
            num_classes: Number of output classes
            input_shape: Input feature dimension (varies by combination)

        Returns:
            Compiled Keras model
        """
        # Dynamic input layer - accepts any dimension
        inputs = Input(shape=(input_shape,), name='feature_input')

        # Normalize input features
        x = BatchNormalization(name='input_bn')(inputs)

        # Hidden layer - fixed 128 units regardless of input size
        x = Dense(units=128, activation='leaky_relu', name='hidden_dense')(x)
        x = Dropout(self.dropout_rate, name='hidden_dropout')(x)

        # Output layer
        outputs = Dense(units=num_classes, activation='softmax', name='output')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs, name='flexible_sin')

        # Compile with optimizer and metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(multi_label=False, name='auc')]
        )

        logger.info(f"FlexibleSINClassifier model built:")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Hidden units: 128")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  Total parameters: {model.count_params()}")

        return model


class TNetworkOnlyIIM(tf.keras.Model):
    """
    T Network-Only model for reconstruction experiments.

    This model contains only the Encoder and Decoder (T Network) without
    the classification head. Used to verify the T network can converge
    and properly reconstruct encrypted embeddings.

    Input vector structure: [p_x | p_i | q_i | cloud | q_x_target]
    - p_x: Sparse autoencoder sample embedding (64,)
    - p_i: Sparse autoencoder anchor embeddings (n_anchors * 64,)
    - q_i: Encrypted anchor embeddings (n_anchors * emb_dim,)
    - cloud: Cloud model predictions (optional)
    - q_x_target: Target encrypted embedding to reconstruct (emb_dim,)

    Output: Reconstructed q_x prediction (emb_dim,)
    """

    def __init__(
        self,
        n_anchors: int,
        raw_dim: int,
        emb_dim: int,
        cloud_vector_size: int = 0,
        context_dim: int = 128,
        name: str = "t_network_only",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.n_anchors = n_anchors
        self.raw_dim = raw_dim
        self.emb_dim = emb_dim
        self.cloud_vector_size = cloud_vector_size
        self.context_dim = context_dim

        # Calculate input slice indices
        # Structure: [p_x | p_i | q_i | cloud | q_x_target]
        self.p_x_end = raw_dim
        self.p_i_end = raw_dim + (n_anchors * raw_dim)
        self.q_i_end = self.p_i_end + (n_anchors * emb_dim)
        self.cloud_end = self.q_i_end + cloud_vector_size
        self.q_x_target_end = self.cloud_end + emb_dim

        # Encoder: processes (p_i, q_i, p_x) to produce context
        self._build_encoder()

        # Decoder (T Network): reconstructs q_x from context
        self._build_decoder()

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae")
        self.cosine_tracker = tf.keras.metrics.Mean(name="cosine_sim")

    def _build_encoder(self):
        """Build the Deep Sets encoder for anchor pairs + sample."""
        anchor_pair_dim = self.raw_dim + self.emb_dim

        # Per-anchor transformation (φ)
        self.phi_dense1 = Dense(512, name='phi_dense1')
        self.phi_bn1 = BatchNormalization(name='phi_bn1')
        self.phi_act1 = tf.keras.layers.LeakyReLU(alpha=0.1, name='phi_act1')

        self.phi_dense2 = Dense(256, name='phi_dense2')
        self.phi_bn2 = BatchNormalization(name='phi_bn2')
        self.phi_act2 = tf.keras.layers.LeakyReLU(alpha=0.1, name='phi_act2')

        self.phi_dense3 = Dense(128, name='phi_dense3')

        # Process p_x (raw sample)
        self.p_x_dense1 = Dense(256, activation='relu', name='p_x_dense1')
        self.p_x_dense2 = Dense(128, activation='relu', name='p_x_dense2')

        # Combine all features (ρ network)
        self.rho_dense1 = Dense(256, name='rho_dense1')
        self.rho_bn1 = BatchNormalization(name='rho_bn1')
        self.rho_act1 = tf.keras.layers.LeakyReLU(alpha=0.1, name='rho_act1')

        self.rho_dense2 = Dense(self.context_dim, name='rho_dense2')
        self.rho_bn2 = BatchNormalization(name='rho_bn2')
        self.rho_act2 = tf.keras.layers.LeakyReLU(alpha=0.1, name='rho_act2')

        # Build encoder as a Keras functional model
        # This allows the encoder to be called independently for context extraction
        anchor_pairs_input = tf.keras.layers.Input(
            shape=(self.n_anchors, anchor_pair_dim),
            name='anchor_pairs'
        )
        p_x_input = tf.keras.layers.Input(
            shape=(self.raw_dim,),
            name='p_x_input'
        )

        # Process anchors through φ network
        x = self.phi_dense1(anchor_pairs_input)
        x = self.phi_bn1(x)
        x = self.phi_act1(x)
        x = self.phi_dense2(x)
        x = self.phi_bn2(x)
        x = self.phi_act2(x)
        x = self.phi_dense3(x)

        # Pool across anchors (permutation invariant)
        c_anchors = tf.reduce_mean(x, axis=1)  # (batch, 128)

        # Process p_x sample
        p_x_features = self.p_x_dense1(p_x_input)
        p_x_features = self.p_x_dense2(p_x_features)  # (batch, 128)

        # Combine all features to create context (ρ network)
        combined = tf.concat([c_anchors, p_x_features], axis=-1)  # (batch, 256)
        context = self.rho_dense1(combined)
        context = self.rho_bn1(context)
        context = self.rho_act1(context)
        context = self.rho_dense2(context)
        context = self.rho_bn2(context)
        context = self.rho_act2(context)  # (batch, context_dim)

        # Create encoder submodel
        self.encoder = tf.keras.Model(
            inputs=[anchor_pairs_input, p_x_input],
            outputs=context,
            name='encoder_deepsets'
        )

    def _build_decoder(self):
        """Build the T network decoder for q_x reconstruction."""
        self.decoder_dense1 = Dense(256, name='decoder_dense1')
        self.decoder_bn1 = BatchNormalization(name='decoder_bn1')
        self.decoder_act1 = tf.keras.layers.LeakyReLU(alpha=0.1, name='decoder_act1')

        self.decoder_dense2 = Dense(512, name='decoder_dense2')
        self.decoder_bn2 = BatchNormalization(name='decoder_bn2')
        self.decoder_act2 = tf.keras.layers.LeakyReLU(alpha=0.1, name='decoder_act2')

        self.decoder_output = Dense(self.emb_dim, activation='linear', name='decoder_output')

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, p_x + p_i + q_i + cloud + q_x_target)
            training: Whether in training mode

        Returns:
            q_x_pred: Reconstructed embedding (batch, emb_dim)
        """
        # Slice input components
        p_x = inputs[:, :self.p_x_end]
        p_i = inputs[:, self.p_x_end:self.p_i_end]
        q_i = inputs[:, self.p_i_end:self.q_i_end]
        # cloud and q_x_target not used in forward pass

        # Reshape p_i and q_i for anchor processing
        p_i_reshaped = tf.reshape(p_i, (-1, self.n_anchors, self.raw_dim))
        q_i_reshaped = tf.reshape(q_i, (-1, self.n_anchors, self.emb_dim))

        # Concatenate anchor pairs: (p_i, q_i) -> (batch, n_anchors, raw_dim + emb_dim)
        anchor_pairs = tf.concat([p_i_reshaped, q_i_reshaped], axis=-1)

        # Use encoder submodel to get context
        # Note: BatchNormalization layers inside encoder will use training mode
        context = self.encoder([anchor_pairs, p_x], training=training)  # (batch, context_dim)

        # Decode context to reconstruct q_x
        x = self.decoder_dense1(context)
        x = self.decoder_bn1(x, training=training)
        x = self.decoder_act1(x)

        x = self.decoder_dense2(x)
        x = self.decoder_bn2(x, training=training)
        x = self.decoder_act2(x)

        q_x_pred = self.decoder_output(x)  # (batch, emb_dim)

        return q_x_pred

    def train_step(self, data):
        """Custom training step with MSE loss on q_x reconstruction."""
        x, _ = data  # y is ignored, we use q_x_target from x

        # Extract target (q_x_target is embedded in x)
        q_x_target = x[:, self.cloud_end:self.q_x_target_end]

        with tf.GradientTape() as tape:
            q_x_pred = self(x, training=True)
            loss = tf.reduce_mean(tf.square(q_x_target - q_x_pred))

        # Compute gradients and update
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute metrics
        mae = tf.reduce_mean(tf.abs(q_x_target - q_x_pred))

        # Cosine similarity
        q_x_target_norm = tf.nn.l2_normalize(q_x_target, axis=1)
        q_x_pred_norm = tf.nn.l2_normalize(q_x_pred, axis=1)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(q_x_target_norm * q_x_pred_norm, axis=1))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        self.cosine_tracker.update_state(cosine_sim)

        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "cosine_sim": self.cosine_tracker.result(),
        }

    def test_step(self, data):
        """Custom test step."""
        x, _ = data

        q_x_target = x[:, self.cloud_end:self.q_x_target_end]
        q_x_pred = self(x, training=False)

        loss = tf.reduce_mean(tf.square(q_x_target - q_x_pred))
        mae = tf.reduce_mean(tf.abs(q_x_target - q_x_pred))

        q_x_target_norm = tf.nn.l2_normalize(q_x_target, axis=1)
        q_x_pred_norm = tf.nn.l2_normalize(q_x_pred, axis=1)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(q_x_target_norm * q_x_pred_norm, axis=1))

        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(mae)
        self.cosine_tracker.update_state(cosine_sim)

        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "cosine_sim": self.cosine_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_tracker, self.cosine_tracker]

    def get_config(self):
        return {
            "n_anchors": self.n_anchors,
            "raw_dim": self.raw_dim,
            "emb_dim": self.emb_dim,
            "cloud_vector_size": self.cloud_vector_size,
            "context_dim": self.context_dim,
        }

    def extract_target(self, inputs):
        """Extract q_x_target from input tensor for loss computation."""
        if isinstance(inputs, np.ndarray):
            return inputs[:, self.cloud_end:self.q_x_target_end]
        return inputs[:, self.cloud_end:self.q_x_target_end]



