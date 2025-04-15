"""
This module implements a pipeline for:
    1. Loading configuration and datasets.
    2. Performing LLM-based data augmentation and analysis on movie and user data.
    3. Analyzing user interactions.
    4. Generating recommendations using LLMs.

The pipeline utilizes various helper functions from local modules:
    - utils.py: For data loading.
    - llm_requests.py: For interfacing with LLM-based APIs.
    - analysis.py: For analyzing and reporting the results.
"""

import logging
import os
from typing import Tuple, List, Dict, Any

import pandas as pd
import yaml

from utils import load_dataset
from llm_requests import (
    fetch_movie_name_with_LLM,
    fetch_next_interaction_with_LLM,
    fetch_user_attribute_with_LLM,
    fetch_next_user_interaction_with_LLM,
    leave_n_out_recommendation_with_LLM,
    create_train_test_files
)
from analysis import (
    analyze_results,
    save_analysis_report,
    analyze_results_popularity,
    compute_item_popularity,
    analyze_results_aggregate_popularity,
    analyze_results_by_popularity_buckets,
    compute_recall,
    compute_mean_hit_ratio
)

# ------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def sample_users(ratings_df: pd.DataFrame,
                 n_per_group: int = 50,
                 random_state: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Sample users from top, bottom, and middle interaction groups.

    The function first calculates the number of unique item interactions per user,
    sorts the users by their interaction counts, and then selects samples from the
    top 20%, middle 60%, and bottom 20% of the sorted list.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing user interactions (must include 'UserID' and 'MovieID' columns).
        n_per_group (int): Maximum number of users to sample from each group.
        random_state (int): Seed for reproducibility of the sampling process.

    Returns:
        Tuple[List, List, List]: A tuple containing lists of UserIDs for:
            - Top users (highest interaction counts)
            - Bottom users (lowest interaction counts)
            - Middle users
    """
    # Count the number of unique movies each user has interacted with.
    user_counts = ratings_df.groupby('UserID')['MovieID'].nunique().reset_index(name='InteractionCount')
    # Sort users in descending order by their interaction counts.
    user_counts = user_counts.sort_values(by='InteractionCount', ascending=False).reset_index(drop=True)

    total_users = len(user_counts)
    # Define indices to split the data into three groups:
    # top 20%, middle 60%, and bottom 20%.
    top_idx = int(0.2 * total_users)
    bottom_idx = int(0.8 * total_users)

    groups: Dict[str, pd.DataFrame] = {
        'top': user_counts.iloc[:top_idx],
        'middle': user_counts.iloc[top_idx:bottom_idx],
        'bottom': user_counts.iloc[bottom_idx:]
    }

    sampled_users: Dict[str, List[Any]] = {}
    sampled = set()  # Keep track of already sampled users to avoid duplicates.

    # Sample users from each group
    for group_name, group_df in groups.items():
        # Exclude users that have already been sampled in previous groups.
        candidates = group_df[~group_df['UserID'].isin(sampled)]
        sample_size = min(n_per_group, len(candidates))
        # Randomly sample users from the group.
        sampled_ids = set(candidates.sample(n=sample_size, random_state=random_state)['UserID'])
        sampled_users[group_name] = list(sampled_ids)
        sampled.update(sampled_ids)

    return sampled_users['top'], sampled_users['bottom'], sampled_users['middle']


def get_model_name(config: Dict[str, Any]) -> str:
    """
    Generate a standardized model name based on the configuration dictionary.

    Depending on the specified model type in the configuration, the function
    cleans up and returns the appropriate model name.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model settings.

    Returns:
        str: Standardized model name.
    """
    model_type = config.get('model_type', 'unknown')
    if model_type in ['hf', 'sglang']:
        # For Hugging Face or SGLang models, replace any '/' with '_' in the model name.
        return config.get('model_name', 'unknown').replace('/', '_')
    elif model_type == 'foundry':
        return config.get('foundry_model_name', 'unknown').replace('/', '_')
    # Default fallback.
    return config.get('deployment_name', 'unknown')


def log_and_save_report(report: List[str], model_name: str, prefix: str = '') -> None:
    """
    Log each line of the report and save it to a file.

    The function prepends an optional prefix to the output filename and logs
    the report content using the module logger.

    Args:
        report (List[str]): List of report lines.
        model_name (str): Model name used to generate the filename.
        prefix (str): Optional prefix for the filename.
    """
    # Log each line in the report.
    for line in report:
        logger.info(line)

    # Build the filename using the model name and prefix.
    filename = f"{prefix}analysis_summary_{model_name}.txt" if prefix else f"analysis_summary_{model_name}.txt"
    # Save the report to the specified file.
    save_analysis_report(report, output_file=filename)


def main(config_path: str = "../config.yaml") -> None:
    """
    Main pipeline execution.

    The pipeline comprises four major steps:
        1. Load configuration and datasets.
        2. Fetch movie and user data using LLMs.
        3. Fetch user interactions
        4. Generate recommendations using an LLM-based leave-n-out strategy.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # ------------------------------------------------------------
    # 1. Load Configuration and Datasets
    # ------------------------------------------------------------
    try:
        # Open and parse the YAML configuration file.
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        return

    # Extract a standardized model name from the configuration.
    model_name = get_model_name(config)

    # Load the datasets:
    # - items_df: Contains item (e.g., movie) metadata.
    items_df = load_dataset(config["item_data_path"])

    # - interaction_df: Contains user-item interactions; only specific columns are loaded.
    try:
        interaction_df = pd.read_csv(config["interaction_data_path"], usecols=['UserID', 'MovieID'])
    except Exception as e:
        logger.error(f"Error loading interaction data: {e}")
        return

    # - users_df: Contains user profile information.
    try:
        users_df = pd.read_csv(config['user_data_path'])
    except Exception as e:
        logger.error(f"Error loading user data: {e}")
        return

    # ------------------------------------------------------------
    # 2. LLM-based Movie and User Data Augmentation and Analysis
    # ------------------------------------------------------------

    # ----- Movie Title Analysis -----
    # Fetch movie titles (or related metadata) using an LLM.
    movie_results = fetch_movie_name_with_LLM(items_df, config)

    # Analyze the fetched movie results using various percentiles.
    coverage_report = analyze_results(
        movie_results,
        percentiles=[1, 10, 20, 25, 50, 75, 90, 100]
    )
    # Log and save the analysis report.
    log_and_save_report(coverage_report, model_name)

    # ----- Item Popularity Analysis -----
    # Compute the popularity of each item based on the number of user interactions.
    popularity_df = compute_item_popularity(interaction_df)

    # Analyze the LLM movie results as a function of item popularity.
    popularity_report = analyze_results_popularity(
        movie_results,
        popularity_df,
        percentiles=[1, 10, 20, 25, 50, 75, 90, 100]
    )
    # Save the popularity-based analysis report with a specific prefix.
    log_and_save_report(popularity_report, model_name, prefix="Popularity_")

    # Analyze aggregate popularity.
    aggregate_pop_report = analyze_results_aggregate_popularity(movie_results, popularity_df)
    log_and_save_report(aggregate_pop_report, model_name, prefix="Aggregate_Popularity_")

    # ----- Bucketed Popularity Coverage Analysis -----
    # Group items into buckets based on popularity (e.g., lower 20% and upper 80%).
    bucket_report = analyze_results_by_popularity_buckets(
        movie_results,
        popularity_df,
        percentile_buckets=[0.2, 0.8]
    )

    # Define a filename for the bucket analysis summary.
    bucket_filename = f"Aggregate_Popularity_Bucket_analysis_summary_{model_name}.txt"
    # Log each line of the bucket report.
    for line in bucket_report:
        logger.info(line)
    # Save the bucket analysis report.
    save_analysis_report(bucket_report, output_file=bucket_filename)

    # ----- User Attribute Analysis -----
    # Fetch user attributes (e.g., demographics) using an LLM.
    user_results = fetch_user_attribute_with_LLM(users_df, config)
    # Analyze the fetched user attributes.
    user_coverage_report = analyze_results(
        user_results,
        percentiles=[1, 10, 20, 25, 50, 75, 90, 100]
    )
    # Save the user analysis report with a 'user_' prefix.
    log_and_save_report(user_coverage_report, model_name, prefix="user_")

    # ------------------------------------------------------------
    # 3. Analysis of User Interactions
    # ------------------------------------------------------------
    # Sample users from three groups (top, bottom, and middle based on activity levels).
    sampled_top, sampled_bottom, sampled_middle = sample_users(interaction_df, n_per_group=50)
    sampled_groups: Dict[str, List[Any]] = {
        'top_users': sampled_top,
        'bottom_users': sampled_bottom,
        'middle_users': sampled_middle
    }

    # Process and analyze each group separately.
    for group_name, user_group in sampled_groups.items():
        # Append the group name to the model name for unique identification.
        current_model_name = f"{model_name}_{group_name}"
        logger.info(f"Starting analysis for {current_model_name}")

        # Filter interactions to include only those for users in the current group.
        filtered_interactions = interaction_df[interaction_df['UserID'].isin(user_group)]

        # Use the LLM to predict or fetch the next user interaction.
        results, results_file_name = fetch_next_user_interaction_with_LLM(filtered_interactions, config)

        # Rename the results file to include the current model and group name.
        new_results_filename = f"{current_model_name}_interaction_results.csv"
        try:
            os.rename(results_file_name, new_results_filename)
            logger.info(f"File renamed successfully to {new_results_filename}")
        except Exception as e:
            logger.error(f"Error renaming file {results_file_name}: {e}")

        # Analyze the fetched interaction results (here using 100% percentile analysis).
        interaction_report = analyze_results(results, percentiles=[100])
        log_and_save_report(interaction_report, current_model_name)

    # ------------------------------------------------------------
    # 4. LLMs as Recommender Systems
    # ------------------------------------------------------------
    # (Optional) You can merge interaction data with item titles for enriched recommendations.
    # Example: df_merged = pd.merge(interaction_df, items_df[['MovieID', 'Title']], on='MovieID', how='inner')

    # Generate recommendations using a leave-n-out evaluation strategy with an LLM.
    recommendations_df, metrics, outfile = leave_n_out_recommendation_with_LLM(
        '../data/movielens_1M/training.tsv',
        '../data/movielens_1M/test.tsv',
        config
    )

    logger.info("Recommendations generated. Results saved to: " + outfile)


if __name__ == "__main__":
    main()