import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pathlib
from sklearn.model_selection import train_test_split


from src.utils.constansts import MODELS_PATH, DATASETS_PATH, DATA_CACHE_PATH


def preprocess(X: pd.DataFrame):
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Initialize lists to store processed columns
    processed_columns = []

    # If there are categorical columns, apply one-hot encoding
    if categorical_cols:
        print("Encoding categorical columns...")
        onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
        X_categorical = pd.DataFrame(onehot_encoder.fit_transform(X[categorical_cols]),
                                     columns=onehot_encoder.get_feature_names_out(categorical_cols))
        processed_columns.append(X_categorical)

    # Apply standard scaling to the numeric columns
    if numeric_cols:
        print("Scaling numerical columns...")
        scaler = StandardScaler()
        X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols)
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
        return row, np.array([])

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


def load_cache_file(dataset_name: str, split_ratio: float):
    path = os.path.join(DATA_CACHE_PATH, f"{dataset_name}_{split_ratio}.pkl")
    print(f"Loading cache for {dataset_name}_{split_ratio}...")
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache_file(dataset_name: str, split_ratio: float, data):
    path = os.path.join(DATA_CACHE_PATH, f"{dataset_name}_{split_ratio}.pkl")
    print(f"Saving cached data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)
