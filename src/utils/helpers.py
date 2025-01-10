
import pickle
import os
from typing import Generator

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from PIL import Image, ImageDraw, ImageFont
from src.utils.constansts import MODELS_PATH, DATASETS_PATH, DATA_CACHE_PATH
import tensorflow as tf



def batch(list_: list, size: int) -> Generator[list, None, None]:
    list_ = np.vstack(list_)
    yield from (list_[i : i + size] for i in range(0, len(list_), size))

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
        print("\nEncoding categorical columns...")
        X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)
        # label_encoder = LabelEncoder()
        # X_categorical = pd.DataFrame()
        # for col in categorical_cols:
        #     # X_categorical[col] = label_encoder.fit_transform(X[col])
        processed_columns.append(X_categorical)

    # Apply standard scaling to the numeric columns
    if numeric_cols:
        print("\nScaling numerical columns...")
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
