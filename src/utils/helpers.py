import pathlib
import pickle
import os
from typing import Generator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from loguru import logger
from src.utils.constansts import MODELS_PATH, DATASETS_PATH, DATA_CACHE_PATH, OUTPUT_DIR_PATH, EMBEDDING_TYPES
from src.utils.config import config


def plot_history(history, filename=None, title=None):
    """Plot and optionally save training curves"""
    plt.figure(figsize=(12, 6))
    plt.title(title or "Training Curve")
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot accuracy if available
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

    if filename:
        plt.savefig(filename)
    plt.show()


def get_dataset_path(dataset_name: str, n_pred_vectors, use_cloud=True, feature_combination=None) -> pathlib.Path:
    rotate_dir = "rotate" if config.encoder_config.rotating_key else ""
    use_cloud_features = "cloud" if (config.cloud_config.names and use_cloud) else "no_cloud"
    cloud_models = "_".join(config.cloud_config.names) if (config.cloud_config.names and use_cloud) else ""

    use_raw_features = ""
    if config.experiment_config.use_embedding and config.experiment_config.n_triangulation_samples > 0:
        use_raw_features = "triangulation_and_raw"
    elif config.experiment_config.n_triangulation_samples <= 0 or config.experiment_config.use_raw:
        use_raw_features = "raw"

    triang_type = (config.experiment_config.triangulation_choosing
                   if type(config.experiment_config.n_triangulation_samples) is str
                   else "_".join(config.experiment_config.triangulation_choosing).replace(",", "")
                   )
    triang_num = str(config.experiment_config.n_triangulation_samples)
    embedding_model = config.encoder_config.embedding
    triang_features = config.experiment_config.triangulation_mode

    # Include calibration distribution types in path
    if config.experiment_config.use_calibration_vector:
        if config.experiment_config.use_key_encoder:
            use_calib_vector = "key_encoder"
        else:
            # Use getattr to be safe in case config isn't fully updated
            dists = getattr(config.experiment_config, 'calibration_distributions', ['gaussian'])
            calib_dists = "_".join(sorted(dists))
            use_calib_vector = f"calib_{calib_dists}"
    else:
        use_calib_vector = ""

    # Include feature combination identifier if present
    feature_combo_dir = ""
    if feature_combination:
        feature_combo_dir = f"ablation_{feature_combination}"

    if config.experiment_config.use_deepset:
        path = (pathlib.Path(OUTPUT_DIR_PATH) / dataset_name / rotate_dir / use_cloud_features / cloud_models /
                embedding_model / use_raw_features / str(n_pred_vectors) / triang_type  / triang_num / "deepset" / feature_combo_dir)
    else:
        path = (pathlib.Path(
            OUTPUT_DIR_PATH) / dataset_name / rotate_dir / use_cloud_features / cloud_models / embedding_model
                / use_raw_features / str(n_pred_vectors) / triang_type / triang_features / use_calib_vector /
                    triang_num / feature_combo_dir)
    os.makedirs(path, exist_ok=True)
    return path


def get_experiment_name() -> str:
    use_embed = "emb" if config.experiment_config.use_embedding else "no_emb"
    use_cloud = "cloud_vec" if config.cloud_config.names else "no_cloud_vec"
    use_rotate_key = "rotate_key" if config.encoder_config.rotating_key else "no_rotate_key"
    use_raw_features = "_raw" if config.experiment_config.use_raw else ""

    if config.experiment_config.use_deepset:
        return f"deepset_{use_cloud}"
    # NEW: Include calibration info in experiment name for logging
    calib_str = ""
    if config.experiment_config.use_calibration_vector:
        dists = getattr(config.experiment_config, 'calibration_distributions', ['gaussian'])
        calib_str = f"_calib_{'_'.join(sorted(dists))}"

    return f"{use_rotate_key}_{use_embed}_{use_cloud}{use_raw_features}{calib_str}"


def load_pretrained_t_network(
    model_path: str | pathlib.Path,
    freeze_weights: bool = True
) -> tf.keras.Model:
    """
    Load a pretrained T network model from disk.

    Args:
        model_path: Path to saved T network .keras file
        freeze_weights: If True, set all layers to trainable=False

    Returns:
        Loaded TNetworkOnlyIIM model with frozen weights

    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If loaded model is not a TNetworkOnlyIIM instance
    """
    # Convert to Path object for easier handling
    model_path = pathlib.Path(model_path)

    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"T network model file not found: {model_path}\n"
            f"Please ensure the model has been trained and saved at this location."
        )

    # Load the model
    try:
        logger.info(f"Loading T network model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load T network model from {model_path}: {e}"
        ) from e

    # Verify the model has an encoder attribute (confirms it's a TNetworkOnlyIIM)
    if not hasattr(model, 'encoder'):
        raise ValueError(
            f"Loaded model does not have an 'encoder' attribute. "
            f"Expected a TNetworkOnlyIIM model, got {type(model).__name__}"
        )

    # Freeze weights if requested
    if freeze_weights:
        for layer in model.layers:
            layer.trainable = False
        logger.info(f"Froze all {len(model.layers)} layers in T network")
    else:
        logger.info(f"Loaded T network with trainable weights")

    # Log model information
    try:
        encoder_output_shape = model.encoder.output_shape
        logger.info(f"T network encoder output shape: {encoder_output_shape}")
    except Exception:
        logger.warning("Could not determine encoder output shape")

    logger.info(f"Successfully loaded T network from {model_path}")
    return model


def get_num_classes(y: np.ndarray) -> int:
    return len(np.unique(y))


def batch(list_: list, size: int) -> Generator[list, None, None]:
    list_ = np.vstack(list_)
    yield from (list_[i: i + size] for i in range(0, len(list_), size))


def pad_image(image, max_shape):
    # Check if any dimension of the image matches the max_shape
    if any(s == max_shape for s in image.shape):
        # If the image is already of the desired shape, no need to pad it
        return image

    # Calculate the padding needed for height and width
    height_padding = (max_shape - image.shape[-3]) // 2
    width_padding = (max_shape - image.shape[-2]) // 2

    # Determine the rank of the input tensor
    rank = len(image.shape)

    if rank == 3:
        # For 3D tensors (height, width, channels)
        paddings = [[height_padding, height_padding], [width_padding, width_padding], [0, 0]]
    elif rank == 4:
        # For 4D tensors (batch_size, height, width, channels)
        paddings = [[0, 0], [height_padding, height_padding], [width_padding, width_padding], [0, 0]]
    else:
        raise ValueError("Unsupported tensor rank: {}".format(rank))

    # Pad the image to match the required shape
    padded_image = tf.pad(image, paddings, mode='CONSTANT')

    return padded_image


def create_image_from_number(number, image_size=(224, 224), font_size=80):
    img = Image.new('RGB', image_size, color='white')  # White background
    draw = ImageDraw.Draw(img)

    # Set the font size and draw the number in the center of the image
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Adjust font path if necessary
    except IOError:
        font = ImageFont.load_default()

    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)  # Use textbbox to get bounding box of the text
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    draw.text(position, text, fill="black", font=font)

    return img


def create_image_from_numbers(numbers, image_size=(224, 224), font_size=80, numbers_per_row=4):
    img = Image.new('RGB', image_size, color='white')  # White background
    draw = ImageDraw.Draw(img)

    # Set the font size and draw the numbers in the center of the image
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Adjust font path if necessary
    except IOError:
        font = ImageFont.load_default()

    rows = [numbers[i:i + numbers_per_row] for i in range(0, len(numbers), numbers_per_row)]
    y_offset = 0
    for row in rows:
        text = ' '.join(map(str, row))
        text += "\n\n"
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Use textbbox to get bounding box of the text
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((image_size[0] - text_width) // 2, y_offset)
        draw.text(position, text, fill="black", font=font)
        y_offset += text_height

    return img


def expand_matrix_to_img_size(matrix, target_shape):
    """
    Expand a given matrix to the target shape by adding zeros around it.

    Parameters:
    matrix (np.ndarray): The input matrix to be expanded.
    target_shape (tuple): The desired shape of the output matrix (rows, cols).

    Returns:
    np.ndarray: The expanded matrix with the target shape.
    """

    original_shape = matrix.shape
    if len(original_shape) != 2 or len(target_shape) != 2:
        raise ValueError("Both input matrix and target shape must be 2-dimensional")

    if original_shape[0] > target_shape[0] or original_shape[1] > target_shape[1]:
        raise ValueError("Target shape must be larger than or equal to the original shape in both dimensions")

    # Calculate the padding for each dimension
    pad_height = target_shape[0] - original_shape[0]
    pad_width = target_shape[1] - original_shape[1]

    # Calculate padding values for top, bottom, left, and right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    padded_matrix = np.pad(matrix, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Stack the matrix 3 times to create 3 channels
    expanded_matrix = np.stack([padded_matrix] * 3, axis=-1)

    if len(expanded_matrix.shape) == 3:
        expanded_matrix = expanded_matrix[np.newaxis, ...]

    return expanded_matrix


def preprocess(X: pd.DataFrame, cloud_dataset=False):
    """
    The function will preprocess the data:
    1. Categorical features will be label encoded (Boy->1, Girl ->2)
    2. Numerical features will be scaled if the data is intended to be used for baseline. For cloud data set, no scaling will be preformed.

    Return pd.Dataframe
    """
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Initialize lists to store processed columns
    processed_columns = []

    # If there are categorical columns, apply one-hot encoding
    if categorical_cols:
        logger.info("#### PREPROCESSING: ##### Encoding categorical columns")
        X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)
        X_categorical = X_categorical.astype(int)  # Tensorflow can't process boolean
        # label_encoder = LabelEncoder()
        # X_categorical = pd.DataFrame()
        # for col in categorical_cols:
        #     # X_categorical[col] = label_encoder.fit_transform(X[col])
        processed_columns.append(X_categorical)

    # Apply standard scaling to the numeric columns
    if numeric_cols:
        logger.info("#### PREPROCESSING: ##### Scaling numerical columns")
        scaler = MinMaxScaler()
        # X_numeric = X[numeric_cols]
        # if cloud_dataset:
        X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
        # else:
        #     X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)

        processed_columns.append(X_numeric)

    # Combine the processed columns
    if processed_columns:
        X_processed = pd.concat(processed_columns, axis=1)
    else:
        X_processed = X.copy()  # If there are no categorical or numeric columns, keep the original dataframe

    return X_processed


def one_hot_labels(num_classes: int, labels: np.ndarray) -> np.ndarray:
    if np.any(labels >= num_classes) or np.any(labels < 0):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}]")

    # Initialize a 2D array of zeros
    one_hot_matrix = np.zeros((labels.size, num_classes))

    # Set the appropriate elements to 1
    one_hot_matrix[np.arange(labels.size), labels] = 1

    return one_hot_matrix


def sample_noise(row: pd.Series, X: pd.DataFrame, y: pd.Series, sample_n=9):
    if sample_n <= 0:
        return pd.DataFrame(row).T, np.array([])

    # Drop the row with the specified index
    df_dropped = X.drop(index=row.name)

    # Sample N rows from the remaining DataFrame
    sampled_rows = df_dropped.sample(n=sample_n)

    # Concatenate the row with the sampled rows
    concatenated_df = pd.concat([pd.DataFrame(row).T, sampled_rows])

    # Shuffle the concatenated DataFrame
    shuffled_df = concatenated_df.sample(frac=1)

    # Get the labels for the sampled rows including the original row
    sampled_labels = y[shuffled_df.index.tolist()]

    # Replace the label for the original row with -1
    sampled_labels.loc[row.name] = -1

    return shuffled_df, sampled_labels.values.reshape(1, -1)


def load_tabular_models(file: str):
    path = os.path.join(MODELS_PATH, file)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(dataset_name: str, split_ratio: float):
    path = os.path.join(DATASETS_PATH, dataset_name, f"dataset_{split_ratio}.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)


def save_data(dataset_name: str, split_ratio: float, data):
    path = os.path.join(DATASETS_PATH, dataset_name, f"dataset_{split_ratio}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_cache_file(dataset_name: str, split_ratio: float):
    path = os.path.join(DATA_CACHE_PATH, f"{dataset_name}_{split_ratio}.pkl")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache_file(dataset_name: str, split_ratio: float, data):
    path = os.path.join(DATA_CACHE_PATH, f"{dataset_name}_{split_ratio}.pkl")
    print(f"Saving cached data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_prompt(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def batching(list_: list, size: int) -> Generator[list, None, None]:
    yield from (list_[i: i + size] for i in range(0, len(list_), size))


def generate_calibration_vectors(embedding_dim: int, distributions: list[str], seed: int = 42) -> list[np.ndarray]:
    """
    Generate multiple calibration vectors with different statistical distributions.
    """
    rng = np.random.default_rng(seed=seed)
    calibration_vectors = []

    for dist_type in distributions:
        dist_type = dist_type.lower().strip()

        if dist_type == "uniform":
            v = rng.uniform(0, 1, size=(1, embedding_dim))

        elif dist_type == "gaussian" or dist_type == "normal":
            v = rng.normal(loc=0.5, scale=0.2, size=(1, embedding_dim))
            v = np.clip(v, 0, 1)

        elif dist_type == "sparse":
            v = np.zeros((1, embedding_dim))
            n_nonzero = embedding_dim // 4
            indices = rng.choice(embedding_dim, size=n_nonzero, replace=False)
            v[0, indices] = rng.uniform(0.5, 1.0, size=n_nonzero)

        elif dist_type == "bimodal":
            v = np.zeros((1, embedding_dim))
            half = embedding_dim // 2
            v[0, :half] = rng.normal(0.2, 0.05, size=half)
            v[0, half:] = rng.normal(0.8, 0.05, size=embedding_dim - half)
            v = np.clip(v, 0, 1)

        elif dist_type == "edges":
            v = rng.choice([0.0, 1.0], size=(1, embedding_dim))

        else:
            logger.warning(f"Unknown calibration distribution '{dist_type}', using uniform")
            v = rng.uniform(0, 1, size=(1, embedding_dim))

        calibration_vectors.append(v)

    logger.info(f"Generated {len(calibration_vectors)} calibration vectors with distributions: {distributions}")
    return calibration_vectors


def get_t_network_model_path(
    dataset_name: str,
    n_anchors: int = None,
    embedding: str = None,
    rotating_key: bool = None,
    ensure_exists: bool = False
) -> pathlib.Path:
    """
    Generate the standardized path for T-Network model files.

    Uses config defaults if parameters not provided.

    Args:
        dataset_name: Name of the dataset the T-Network was trained on
        n_anchors: Number of triangulation anchors (defaults to config)
        embedding: Embedding model name (defaults to config)
        rotating_key: Whether rotating key was used (defaults to config)
        ensure_exists: If True, raises FileNotFoundError if path doesn't exist

    Returns:
        Path object pointing to the T-Network .keras file

    Raises:
        FileNotFoundError: If ensure_exists=True and file doesn't exist
    """
    # Use config defaults
    if n_anchors is None:
        n_anchors = config.experiment_config.n_triangulation_samples
    if embedding is None:
        embedding = config.encoder_config.embedding
    if rotating_key is None:
        rotating_key = config.encoder_config.rotating_key

    # Construct path (SAME logic as _save_model)
    rotate_key = "rotate" if rotating_key else "no_rotate"
    model_filename = f"t_network_{dataset_name}_{n_anchors}anchors_{embedding}_{rotate_key}.keras"
    model_path = pathlib.Path(OUTPUT_DIR_PATH) / "t_network_models" / model_filename

    # Optional existence check with helpful error
    if ensure_exists and not model_path.exists():
        raise FileNotFoundError(
            f"T-Network model not found at: {model_path}\n"
            f"Expected parameters:\n"
            f"  - Dataset: {dataset_name}\n"
            f"  - Anchors: {n_anchors}\n"
            f"  - Embedding: {embedding}\n"
            f"  - Rotating Key: {rotating_key}\n"
            f"Train a T-Network first with:\n"
            f"  python main.py --experiment-to-run t_network_training --datasets {dataset_name}"
        )

    return model_path