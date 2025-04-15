import logging
import pandas as pd
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_results(results_df, percentiles=[25, 50, 75, 100]):
    """
    Analyze model coverage (accuracy) at various percentiles and overall.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ["MovieID", "GeneratedTitle", "ErrorFlag"].
        percentiles (list): List of percentiles to evaluate coverage.

    Returns:
        list: Analysis report lines.
    """
    total = len(results_df)
    if total == 0:
        return ["No results to analyze."]

    cumsum_errors = results_df["ErrorFlag"].cumsum()
    total_errors = int(cumsum_errors.iloc[-1])
    analysis_report = []

    for percentile in percentiles:
        if percentile < 0 or percentile > 100:
            analysis_report.append(f"Invalid percentile value: {percentile}. Must be between 0 and 100.")
            continue

        # Determine the row index for the given percentile
        cp = int((percentile / 100) * total)
        cp = max(1, cp)  # Ensure at least one row is considered

        err_cp = cumsum_errors.iloc[cp - 1]
        correct_cp = cp - err_cp  # Number of correct predictions
        coverage_cp = correct_cp / cp
        coverage_cp_percentage = round(coverage_cp * 100, 2)
        total_size = cp  # Total number of entries in this percentile

        analysis_report.append(
            f"Coverage at {percentile}th percentile: {coverage_cp_percentage}% | "
            f"Errors: {err_cp} | Correct: {correct_cp} | Total: {total_size}"
        )

    total_coverage = (total - total_errors) / total if total > 0 else 0
    total_coverage_percentage = round(total_coverage * 100, 2)
    analysis_report.append(f"Total Coverage: {total_coverage_percentage}% | Total Errors: {total_errors}")

    return analysis_report

def save_analysis_report(analysis_report, output_file="results/analysis_summary.txt"):
    """
    Save the analysis report lines into a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in analysis_report:
            f.write(line + "\n")
    logger.info(f"Analysis saved to {output_file}")


def compute_item_popularity(ratings_df):
    """
    Compute the popularity of each item based on the number of unique user interactions.

    Parameters:
        ratings_df (pd.DataFrame): DataFrame with columns ["UserID", "MovieID", "Rating", "Timestamp"].

    Returns:
        pd.DataFrame: DataFrame with columns ["MovieID", "PopularityScore"], ordered by popularity descending.
    """
    # Group by MovieID and count unique UserIDs
    popularity_df = ratings_df.groupby('MovieID')['UserID'].nunique().reset_index()
    popularity_df.rename(columns={'UserID': 'PopularityScore'}, inplace=True)

    # Sort by PopularityScore in descending order
    popularity_df = popularity_df.sort_values(by='PopularityScore', ascending=False).reset_index(drop=True)

    return popularity_df


def analyze_results_popularity(results_df, popularity_df, percentiles=[25, 50, 75, 100]):
    """
    Analyze model coverage (accuracy) at various popularity percentiles and overall.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ["MovieID", "GeneratedTitle", "ErrorFlag"].
        popularity_df (pd.DataFrame): DataFrame with ["MovieID", "PopularityScore"] ordered by popularity descending.
        percentiles (list): List of percentiles to evaluate coverage.

    Returns:
        list: Analysis report lines.
    """
    total_items = len(popularity_df)
    if total_items == 0:
        return ["No items to analyze."]

    # Merge popularity with results to align ErrorFlags
    merged_df = popularity_df.merge(results_df[['MovieID', 'ErrorFlag']], on='MovieID', how='left')

    # Assume ErrorFlag=1 (error) for items not present in results_df
    merged_df['ErrorFlag'] = merged_df['ErrorFlag'].fillna(1).astype(int)

    # Cumulative sum of errors up to each item in popularity order
    cumsum_errors = merged_df["ErrorFlag"].cumsum()
    total_errors = int(cumsum_errors.iloc[-1])

    analysis_report = []

    for percentile in percentiles:
        if not (0 <= percentile <= 100):
            analysis_report.append(f"Invalid percentile value: {percentile}. Must be between 0 and 100.")
            continue

        # Determine the cutoff index for the given percentile
        cutoff_index = int((percentile / 100) * total_items)
        cutoff_index = max(1, cutoff_index)  # Ensure at least one item is considered

        # Errors and correct predictions up to the cutoff
        errors_up_to_cutoff = cumsum_errors.iloc[cutoff_index - 1]
        correct_up_to_cutoff = cutoff_index - errors_up_to_cutoff
        coverage = (correct_up_to_cutoff / cutoff_index) * 100
        coverage = round(coverage, 2)

        analysis_report.append(
            f"Coverage at {percentile}th popularity percentile: {coverage}% | "
            f"Errors: {errors_up_to_cutoff} | Correct: {correct_up_to_cutoff} | Total: {cutoff_index}"
        )

    # Overall coverage
    overall_coverage = ((total_items - total_errors) / total_items) * 100 if total_items > 0 else 0
    overall_coverage = round(overall_coverage, 2)
    analysis_report.append(
        f"Total Coverage: {overall_coverage}% | Total Errors: {total_errors}"
    )

    return analysis_report


def analyze_results_aggregate_popularity(results_df, popularity_df, item_counts=None):
    """
    Analyze model coverage (accuracy) based on increasing numbers of popular items.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ["MovieID", "GeneratedTitle", "ErrorFlag"].
        popularity_df (pd.DataFrame): DataFrame with ["MovieID", "PopularityScore"] ordered by popularity descending.
        item_counts (list, optional): List of item counts to evaluate coverage.
                                     If None, defaults to all counts from 1 to total_items.

    Returns:
        list: Analysis report lines.
    """
    total_items = len(popularity_df)
    if total_items == 0:
        return ["No items to analyze."]

    # Merge popularity with results to align ErrorFlags
    merged_df = popularity_df.merge(results_df[['MovieID', 'ErrorFlag']], on='MovieID', how='left')

    # Assume ErrorFlag=1 (error) for items not present in results_df
    merged_df['ErrorFlag'] = merged_df['ErrorFlag'].fillna(1).astype(int)

    # Cumulative sum of errors up to each item in popularity order
    cumsum_errors = merged_df["ErrorFlag"].cumsum()
    total_errors = int(cumsum_errors.iloc[-1])

    analysis_report = []

    # Generate item_counts from 1 to total_items if not provided
    if item_counts is None:
        item_counts = list(range(1, total_items + 1))

    for count in item_counts:
        if count < 1 or count > total_items:
            analysis_report.append(f"Invalid item count: {count}. Must be between 1 and {total_items}.")
            continue

        # Errors and correct predictions up to the current count
        errors_up_to_count = cumsum_errors.iloc[count - 1]
        correct_up_to_count = count - errors_up_to_count
        coverage = (correct_up_to_count / count) * 100
        coverage = round(coverage, 2)

        analysis_report.append(
            f"Coverage at {count} items: {coverage}% | "
            f"Errors: {errors_up_to_count} | Correct: {correct_up_to_count} | Total: {count}"
        )

    # Overall coverage
    overall_coverage = ((total_items - total_errors) / total_items) * 100 if total_items > 0 else 0
    overall_coverage = round(overall_coverage, 2)
    analysis_report.append(
        f"Total Coverage: {overall_coverage}% | Total Errors: {total_errors} | Total Items: {total_items}"
    )

    return analysis_report

def analyze_results_by_buckets(
        results_df,
        item_counts=None,
        bucket_size=5,
        percentile_buckets=None
):
    """
    Analyze model coverage (accuracy) based on buckets of popular items.

    Depending on which parameters are provided, buckets can be created in three ways:

    1. **Item Counts** (explicit):
       - If you supply `item_counts=[5,10,15,...]`, the function treats these as
         the end indices of buckets. For example, [5,10] means:
           Bucket1 -> items 1..5
           Bucket2 -> items 6..10

    2. **Bucket Size** (default):
       - If `item_counts` is not provided and `percentile_buckets` is None,
         we create buckets of size `bucket_size` until we exhaust all items.
         For example, with `bucket_size=5` and `total_items=13`, we get:
           Bucket1 -> items 1..5
           Bucket2 -> items 6..10
           Bucket3 -> items 11..13

    3. **Percentile Buckets**:
       - If you supply a list of percentiles via `percentile_buckets` (e.g. [0.2, 0.4, 0.8, 1.0]),
         then we convert each percentile to an item index = `ceil(total_items * percentile)`.
         For example, if total_items=10 and `percentile_buckets=[0.2, 0.6, 1.0]`:
           -> item indices = [2, 6, 10]  (i.e. Bucket1 -> items 1..2, Bucket2 -> items 3..6, Bucket3 -> items 7..10)
         *Note:* If the last bucket index is not exactly `total_items`, we ensure it gets extended to `total_items`.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns: ["MovieID", "GeneratedTitle", "ErrorFlag"].

    popularity_df : pd.DataFrame
        Must have columns: ["MovieID", "PopularityScore"],
        ordered by popularity descending (index 0 is most popular).

    item_counts : list, optional
        Explicit list of cumulative item indices defining the end of each bucket.
        e.g., [5,10,15] means 3 buckets for items [1..5], [6..10], [11..15].

    bucket_size : int, optional
        Used only if neither `item_counts` nor `percentile_buckets` is provided.
        Defines how many items per bucket. Defaults to 5.

    percentile_buckets : list of floats, optional
        Each float must be in the range (0, 1]. These define the cumulative
        percent of items. For instance, [0.25, 0.5, 1.0] means:
          - 1st bucket: 0..25% of items
          - 2nd bucket: 25..50% of items
          - 3rd bucket: 50..100% of items

    Returns
    -------
    analysis_report : list of str
        A list of human-readable coverage summaries per bucket plus an overall coverage line.
    """
    total_items = len(results_df)

    # 1) If percentile_buckets is provided, convert them to item_counts
    if percentile_buckets is not None and len(percentile_buckets) > 0:
        # Validate all percentiles
        for p in percentile_buckets:
            if p <= 0 or p > 1:
                raise ValueError(
                    f"Percentile {p} is out of valid range (0,1]."
                )

        item_counts = []
        for pct in percentile_buckets:
            idx = int(math.ceil(total_items * pct))
            # Avoid duplicates or out-of-range indexes
            if idx < 1:
                idx = 1
            elif idx > total_items:
                idx = total_items
            item_counts.append(idx)

        # Ensure the last bucket ends exactly at total_items
        if item_counts[-1] != total_items:
            item_counts[-1] = total_items

    # 2) If still no item_counts, build them by `bucket_size`
    if item_counts is None:
        item_counts = list(range(bucket_size, total_items + 1, bucket_size))
        # Make sure the last bucket ends at total_items
        if item_counts[-1] != total_items:
            item_counts.append(total_items)

    # 3) Now we have item_counts (either user-provided, derived from percentiles, or from bucket_size)
    analysis_report = []
    last_index = 0

    for count in item_counts:
        if count < 1 or count > total_items:
            analysis_report.append(
                f"Invalid item count: {count}. Must be between 1 and {total_items}."
            )
            continue

        # Slice the DataFrame for the current group/bucket
        subset_df = results_df.iloc[last_index:count]

        errors_in_subset = subset_df["ErrorFlag"].sum()
        group_size = len(subset_df)
        correct_in_subset = group_size - errors_in_subset

        coverage = 0.0
        if group_size > 0:
            coverage = (correct_in_subset / group_size) * 100
        coverage = round(coverage, 2)

        # Human-readable coverage line
        analysis_report.append(
            f"Coverage for items [{last_index + 1}..{count}]: "
            f"{coverage}% | Errors: {errors_in_subset} | "
            f"Correct: {correct_in_subset} | Group Size: {group_size}"
        )

        last_index = count

    # Overall coverage across all items
    total_errors = results_df["ErrorFlag"].sum()
    total_correct = total_items - total_errors
    overall_coverage = (total_correct / total_items) * 100 if total_items else 0
    overall_coverage = round(overall_coverage, 2)

    analysis_report.append(
        f"Total Coverage (all {total_items} items): {overall_coverage}% | "
        f"Total Errors: {total_errors}"
    )

    return analysis_report

def analyze_results_by_popularity_buckets(
        results_df,
        popularity_df,
        item_counts=None,
        bucket_size=5,
        percentile_buckets=None
):
    """
    Analyze model coverage (accuracy) based on buckets of popular items.

    Depending on which parameters are provided, buckets can be created in three ways:

    1. **Item Counts** (explicit):
       - If you supply `item_counts=[5,10,15,...]`, the function treats these as
         the end indices of buckets. For example, [5,10] means:
           Bucket1 -> items 1..5
           Bucket2 -> items 6..10

    2. **Bucket Size** (default):
       - If `item_counts` is not provided and `percentile_buckets` is None,
         we create buckets of size `bucket_size` until we exhaust all items.
         For example, with `bucket_size=5` and `total_items=13`, we get:
           Bucket1 -> items 1..5
           Bucket2 -> items 6..10
           Bucket3 -> items 11..13

    3. **Percentile Buckets**:
       - If you supply a list of percentiles via `percentile_buckets` (e.g. [0.2, 0.4, 0.8, 1.0]),
         then we convert each percentile to an item index = `ceil(total_items * percentile)`.
         For example, if total_items=10 and `percentile_buckets=[0.2, 0.6, 1.0]`:
           -> item indices = [2, 6, 10]  (i.e. Bucket1 -> items 1..2, Bucket2 -> items 3..6, Bucket3 -> items 7..10)
         *Note:* If the last bucket index is not exactly `total_items`, we ensure it gets extended to `total_items`.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns: ["MovieID", "GeneratedTitle", "ErrorFlag"].

    popularity_df : pd.DataFrame
        Must have columns: ["MovieID", "PopularityScore"],
        ordered by popularity descending (index 0 is most popular).

    item_counts : list, optional
        Explicit list of cumulative item indices defining the end of each bucket.
        e.g., [5,10,15] means 3 buckets for items [1..5], [6..10], [11..15].

    bucket_size : int, optional
        Used only if neither `item_counts` nor `percentile_buckets` is provided.
        Defines how many items per bucket. Defaults to 5.

    percentile_buckets : list of floats, optional
        Each float must be in the range (0, 1]. These define the cumulative
        percent of items. For instance, [0.25, 0.5, 1.0] means:
          - 1st bucket: 0..25% of items
          - 2nd bucket: 25..50% of items
          - 3rd bucket: 50..100% of items

    Returns
    -------
    analysis_report : list of str
        A list of human-readable coverage summaries per bucket plus an overall coverage line.
    """
    total_items = len(popularity_df)
    if total_items == 0:
        return ["No items to analyze."]

    # Merge popularity with results to align ErrorFlags
    merged_df = popularity_df.merge(
        results_df[['MovieID', 'ErrorFlag']], on='MovieID', how='left'
    )

    # Assume ErrorFlag=1 (error) for items not present in results_df
    merged_df['ErrorFlag'] = merged_df['ErrorFlag'].fillna(1).astype(int)

    # 1) If percentile_buckets is provided, convert them to item_counts
    if percentile_buckets is not None and len(percentile_buckets) > 0:
        # Validate all percentiles
        for p in percentile_buckets:
            if p <= 0 or p > 1:
                raise ValueError(
                    f"Percentile {p} is out of valid range (0,1]."
                )

        item_counts = []
        for pct in percentile_buckets:
            idx = int(math.ceil(total_items * pct))
            # Avoid duplicates or out-of-range indexes
            if idx < 1:
                idx = 1
            elif idx > total_items:
                idx = total_items
            item_counts.append(idx)

        # Ensure the last bucket ends exactly at total_items
        if item_counts[-1] != total_items:
            item_counts[-1] = total_items

    # 2) If still no item_counts, build them by `bucket_size`
    if item_counts is None:
        item_counts = list(range(bucket_size, total_items + 1, bucket_size))
        # Make sure the last bucket ends at total_items
        if item_counts[-1] != total_items:
            item_counts.append(total_items)

    # 3) Now we have item_counts (either user-provided, derived from percentiles, or from bucket_size)
    analysis_report = []
    last_index = 0

    for count in item_counts:
        if count < 1 or count > total_items:
            analysis_report.append(
                f"Invalid item count: {count}. Must be between 1 and {total_items}."
            )
            continue

        # Slice the DataFrame for the current group/bucket
        subset_df = merged_df.iloc[last_index:count]

        errors_in_subset = subset_df["ErrorFlag"].sum()
        group_size = len(subset_df)
        correct_in_subset = group_size - errors_in_subset

        coverage = 0.0
        if group_size > 0:
            coverage = (correct_in_subset / group_size) * 100
        coverage = round(coverage, 2)

        # Human-readable coverage line
        analysis_report.append(
            f"Coverage for items [{last_index + 1}..{count}]: "
            f"{coverage}% | Errors: {errors_in_subset} | "
            f"Correct: {correct_in_subset} | Group Size: {group_size}"
        )

        last_index = count

    # Overall coverage across all items
    total_errors = merged_df["ErrorFlag"].sum()
    total_correct = total_items - total_errors
    overall_coverage = (total_correct / total_items) * 100 if total_items else 0
    overall_coverage = round(overall_coverage, 2)

    analysis_report.append(
        f"Total Coverage (all {total_items} items): {overall_coverage}% | "
        f"Total Errors: {total_errors}"
    )

    return analysis_report

def compute_recall(results_df, n_users=50):
    """
    Computes the recall

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns:
            - "RowIndex" (optional, not used in calculations)
            - "GeneratedInteraction" (not used in calculations)
            - "RealInteraction" (in the format "UserID::ItemID")
            - "ErrorFlag" (0 if correct, 1 if error)
        n_users (int): Number of unique users (default=50).

    Returns:
        float: The mean hit ratio (in [0, 1]) across all users.
    """
    # 1. Parse UserID from the "RealInteraction" column.
    #    "RealInteraction" is in the format "338::2987"
    results_df["UserID"] = results_df["RealInteraction"].apply(lambda x: x.split("::")[0])

    # 2. Group by UserID
    grouped = results_df.groupby("UserID")

    # 3. For each user, compute:
    #    - total predictions
    #    - number of correct predictions (ErrorFlag == 0)
    #    - hit ratio = (# correct) / (# total)
    user_stats = grouped.agg(
        total_predictions=("ErrorFlag", "count"),
        correct_predictions=("ErrorFlag", lambda x: (x == 0).sum())
    )

    user_stats["recall"] = user_stats["correct_predictions"] / user_stats["total_predictions"]

    # 4. Compute the recall across all users
    recall = user_stats["recall"].mean()

    # If you want to explicitly check if you have the expected number of users:
    if len(user_stats) != n_users:
        print(f"Warning: Expected {n_users} users, but found {len(user_stats)}.")

    return recall

def compute_mean_hit_ratio(results_df, n_users=50, top_k=10):
    """
    Computes the mean hit ratio (HR@K) across users and counts users with binary hits.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns:
            - "GeneratedInteraction": List of recommended item IDs (e.g., ["2987", "1234", ...])
            - "RealInteraction": The true interaction in the format "UserID::ItemID"
            - "ErrorFlag" (optional, not used here)
        n_users (int): Expected number of unique users (default=50).
        top_k (int): The number of top recommendations to consider for HR@K (default=10).

    Returns:
        tuple: (mean_hit_ratio, binary_hit_count)
            - mean_hit_ratio (float): The mean hit ratio (HR@K) across all users.
            - binary_hit_count (int): The number of users with at least one hit.
    """
    # 1. Parse UserID and ItemID from the "RealInteraction" column.
    results_df["UserID"] = results_df["RealInteraction"].apply(lambda x: x.split("::")[0])
    results_df["TrueItemID"] = results_df["RealInteraction"].apply(lambda x: x.split("::")[1])

    # 2. Initialize a column to indicate if the true item is in the top-K recommendations
    results_df["Hit"] = results_df.apply(
        lambda row: 1 if row["TrueItemID"] in row["GeneratedInteraction"][:top_k] else 0,
        axis=1
    )

    # 3. Group by UserID and compute whether each user has at least one hit
    grouped = results_df.groupby("UserID")
    user_stats = grouped.agg(total_hits=("Hit", "sum"))

    # Replace total_hits with binary indicator: 1 if any hits, 0 otherwise
    user_stats["binary_hit"] = user_stats["total_hits"].apply(lambda x: 1 if x > 0 else 0)

    # Compute the overall mean hit ratio
    mean_hit_ratio = user_stats["binary_hit"].mean()

    # Count users with at least one hit
    binary_hit_count = user_stats["binary_hit"].sum()

    # Check if the number of users matches the expected value
    if len(user_stats) != n_users:
        print(f"Warning: Expected {n_users} users, but found {len(user_stats)}.")

    return mean_hit_ratio, binary_hit_count