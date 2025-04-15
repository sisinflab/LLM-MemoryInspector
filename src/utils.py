import pandas as pd
import os
from rapidfuzz import fuzz

def compute_similarity(str_a, str_b):
    """
    Compute similarity (fuzzy match ratio) between two strings using rapidfuzz.

    Parameters:
        str_a (str): First string.
        str_b (str): Second string.

    Returns:
        float: Similarity ratio between 0 and 100.
    """
    return round(fuzz.ratio(str_a, str_b), 2)

def load_dataset(csv_path):
    """
    Load dataset from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return pd.read_csv(csv_path)
