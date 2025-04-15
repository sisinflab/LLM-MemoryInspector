import os
import pickle
import pandas as pd
import numpy as np
import math
import re
import logging
from collections import defaultdict
from tqdm import tqdm  # For progress tracking
from rapidfuzz import fuzz  # Use rapidfuzz for fuzzy matching

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Precompile regex patterns.
_CLEAN_REGEX = re.compile(r'\s*\(.*?\)')
_PUNCT_REGEX = re.compile(r'[^\w\s]')

def clean_title(title):
    """
    Remove anything in parentheses from the title and strip extra whitespace.
    For example:
        clean_title("Harry Potter (2008)") returns "Harry Potter"
    """
    return _CLEAN_REGEX.sub('', title).strip()

def normalize_title(title):
    """
    Normalize the movie title for matching by cleaning the title (removing text in parentheses),
    lowercasing, stripping whitespace, and removing punctuation.
    """
    title = clean_title(title)
    title = title.lower().strip()
    return _PUNCT_REGEX.sub('', title)

def is_similar(title1, title2, threshold=80):
    """
    Return True if the fuzzy similarity ratio between title1 and title2 is at least the threshold.
    The threshold is given as an integer (e.g., 80 means 80% similarity).
    Uses rapidfuzz.fuzz.ratio.
    """
    return fuzz.ratio(title1, title2) >= threshold

def compute_recommendation_statistics(recs_by_user):
    """
    Computes overall statistics on the recommended items across all users.

    Parameters:
         recs_by_user (dict): Dictionary mapping each user_id to a list of tuples (rank, title)
                             representing that user's recommendations.

    Returns:
         stats (dict): Dictionary containing:
             - 'total_recommendations': total number of recommendations.
             - 'unique_items': total number of unique recommended items.
             - 'item_counts': dictionary mapping each item (title) to the number of times it was recommended.
             - 'position_counts': dictionary mapping each rank (position) to a dictionary of item counts at that position.
             - 'top_items': list of tuples (item, count) sorted in descending order of frequency.
    """
    item_counts = {}
    position_counts = {}

    for user, recs in recs_by_user.items():
        for rank, title in recs:
            item_counts[title] = item_counts.get(title, 0) + 1
            if rank not in position_counts:
                position_counts[rank] = {}
            position_counts[rank][title] = position_counts[rank].get(title, 0) + 1

    total_recommendations = sum(item_counts.values())
    unique_items = len(item_counts)
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    stats = {
        "total_recommendations": total_recommendations,
        "unique_items": unique_items,
        "item_counts": item_counts,
        "position_counts": position_counts,
        "top_items": top_items
    }
    return stats

def evaluate_checkpoint(checkpoint_file, test_file, item_data_path, similarity_threshold=0.85,
                        cutoffs=[1, 3, 5, 10, 20, 50]):
    """
    Loads a checkpoint (.pkl file) containing saved recommendations, then loads the test data,
    and computes evaluation metrics (HR@K, nDCG@K, and MRR@K) for each user based on a specified similarity threshold.
    Also computes qualitative statistics of the recommendation lists.

    Parameters:
        checkpoint_file (str): Path to the .pkl checkpoint file.
        test_file (str): Path to the TSV file with test interactions.
        item_data_path (str): Path to the CSV file with item metadata (e.g., MovieID, Title).
        similarity_threshold (float): Threshold for considering two titles as matching (default is 0.85).
                                      This value is multiplied by 100 to map to a 0–100 scale.
        cutoffs (list): List of cutoff values (K) at which to compute the metrics.

    Returns:
        metrics (dict): A dictionary where each key is a cutoff value and each value is a dict with
                        the average HR@K, nDCG@K, and MRR@K over the users.
        rec_stats (dict): A dictionary containing qualitative statistics of the recommendations.
    """
    # --- 1. Load the checkpoint file. ---
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist.")
    with open(checkpoint_file, "rb") as f:
        checkpoint = pickle.load(f)
    recommendations = checkpoint.get("recommendations", [])
    if not recommendations:
        logger.warning("No recommendations found in the checkpoint file.")
        return {}, {}

    # --- 2. Load and merge the test data with the items metadata. ---
    items_df = pd.read_csv(item_data_path)
    test_df = pd.read_csv(
        test_file,
        sep="\t",
        header=None,
        names=["UserID", "MovieID", "Interact"],
        usecols=["UserID", "MovieID"]
    )
    test_df = pd.merge(test_df, items_df[['MovieID', 'Title']], on='MovieID', how='inner')
    test_titles_by_user = test_df.groupby("UserID")["Title"].apply(list).to_dict()

    # Precompute normalized test titles for each user.
    normalized_test_titles_by_user = {
        user: [normalize_title(t) for t in titles]
        for user, titles in test_titles_by_user.items()
    }

    # --- 3. Group recommendations by user. ---
    recs_by_user = defaultdict(list)
    for rec in recommendations:
        recs_by_user[rec["UserID"]].append((rec["Rank"], rec["Title"]))
    rec_stats = compute_recommendation_statistics(recs_by_user)

    # Precompute discount factors for ranks up to the maximum cutoff.
    max_cutoff = max(cutoffs)
    discount_cache = {r: 1 / math.log2(r + 1) for r in range(1, max_cutoff + 1)}

    # Prepare similarity threshold as integer on 0-100 scale.
    threshold_int = int(similarity_threshold * 100)

    # Initialize metric containers.
    hr_scores = {k: [] for k in cutoffs}
    ndcg_scores = {k: [] for k in cutoffs}
    mrr_scores = {k: [] for k in cutoffs}

    # --- 4. Recompute the metrics per user. ---
    for user_id, recs in tqdm(recs_by_user.items(), desc="Evaluating Users"):
        if user_id not in normalized_test_titles_by_user:
            continue
        norm_test_titles = normalized_test_titles_by_user[user_id]
        norm_recs = sorted(
            [(rank, normalize_title(title)) for rank, title in recs],
            key=lambda x: x[0]
        )
        for k in cutoffs:
            top_k = [(rank, rec_title) for rank, rec_title in norm_recs if rank <= k]
            hit_flag = 0
            first_relevant_rank = None
            DCG = 0.0
            for rank, rec_title in top_k:
                if any(is_similar(rec_title, norm_test, threshold=threshold_int)
                       for norm_test in norm_test_titles):
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    DCG += discount_cache.get(rank, 1 / math.log2(rank + 1))
                    hit_flag = 1
                    # One hit is enough for HR@K, so break out of checking this recommendation.
                    # (nDCG still accumulates discount only once per rec.)
            hr_scores[k].append(hit_flag)
            ideal_count = min(len(norm_test_titles), k)
            ideal_DCG = sum(discount_cache[i] for i in range(1, ideal_count + 1))
            ndcg_scores[k].append(DCG / ideal_DCG if ideal_DCG > 0 else 0)
            mrr_scores[k].append(1 / first_relevant_rank if first_relevant_rank is not None else 0)

    # --- 5. Compute average metrics across users. ---
    metrics = {}
    for k in cutoffs:
        avg_hr = np.mean(hr_scores[k]) if hr_scores[k] else 0.0
        avg_ndcg = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
        avg_mrr = np.mean(mrr_scores[k]) if mrr_scores[k] else 0.0
        metrics[k] = {f"HR@{k}": avg_hr, f"nDCG@{k}": avg_ndcg, f"MRR@{k}": avg_mrr}
        logger.info(f"Cutoff {k} -> HR@{k}: {avg_hr:.4f}, nDCG@{k}: {avg_ndcg:.4f}, MRR@{k}: {avg_mrr:.4f}")

    return metrics, rec_stats

def check_absent_items(checkpoint_path, item_data_path):
    """
    Checks which items recommended in the checkpoint are completely absent from the dataset.
    Counts and prints the missing items along with their recommendation frequencies.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    recommendations = checkpoint.get("recommendations", [])
    if not recommendations:
        logger.warning("No recommendations found in the checkpoint file.")
        return

    # Count frequency of normalized recommended titles.
    rec_freq = {}
    for rec in recommendations:
        title = rec.get("Title", "")
        norm_title = normalize_title(title)
        rec_freq[norm_title] = rec_freq.get(norm_title, 0) + 1

    # Load dataset items.
    items_df = pd.read_csv(item_data_path)
    dataset_titles = set(normalize_title(t) for t in items_df['Title'].dropna().unique())

    # Find recommended items missing in the dataset.
    missing_items = {title: count for title, count in rec_freq.items() if title not in dataset_titles}
    missing_count = len(missing_items)
    total_missing_recs = sum(missing_items.values())

    logger.info(f"Distinct recommended items missing from dataset: {missing_count}")
    logger.info(f"Total missing recommendations (with duplicates): {total_missing_recs}")
    if missing_items:
        logger.info("Missing items and their counts:")
        for title, count in missing_items.items():
            logger.info(f"  {title}: {count} times")
    else:
        logger.info("All recommended items are present in the dataset.")

# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    checkpoint_path = "meta-llama_Llama-3.2-1B-Instruct_checkpoint.pkl"
    test_file_path = "../data/movielens_1M/elliot/test.tsv"
    item_data_path = "../data/movielens_1M/movies.csv"
    threshold = 1  # 0.8 similarity threshold - 1 is equal to exact matching

    # Derive log file name from the checkpoint file.
    checkpoint_basename = os.path.basename(checkpoint_path)  # e.g., "gpt-4o-mini_checkpoint.pkl"
    evaluation_log = checkpoint_basename.replace("_checkpoint.pkl", "_evaluation.out")

    # Add file handler to logger.
    file_handler = logging.FileHandler(evaluation_log)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Evaluate checkpoint.
    metrics, rec_stats = evaluate_checkpoint(
        checkpoint_file=checkpoint_path,
        test_file=test_file_path,
        item_data_path=item_data_path,
        similarity_threshold=threshold
    )

    logger.info(f"Evaluation Metrics with {threshold*100}% threshold:")
    for k, metric in metrics.items():
        logger.info(f"Cutoff {k}: {metric}")

    logger.info("\nOverall Recommendation Statistics:")
    logger.info(f"Total recommendations: {rec_stats['total_recommendations']}")
    logger.info(f"Unique items recommended: {rec_stats['unique_items']}")
    logger.info("Top recommended items:")
    for title, count in rec_stats['top_items'][:10]:
        logger.info(f"  {title}: {count} times")

    logger.info("\nRecommendation Frequency by Position:")
    for rank in sorted(rec_stats["position_counts"].keys()):
        logger.info(f"Rank {rank}:")
        for title, count in sorted(rec_stats["position_counts"][rank].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"    {title}: {count} times")

    # --- Check for absent items in the dataset ---
    check_absent_items(checkpoint_path, item_data_path)