import os
import logging
import pandas as pd
import random
import numpy as np
import torch
import openai
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from huggingface_hub import login, snapshot_download
from utils import compute_similarity
from tqdm import tqdm
import pickle
import re
import math
from rapidfuzz import fuzz


logging.basicConfig(level=logging.INFO, format='\n%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_hf_pipeline(model_name, hf_token, model_dir=None):
    """
    Initialize a Hugging Face text-generation pipeline.

    Parameters:
        model_name (str): Name of the HuggingFace model.

    Returns:
        pipeline: A text-generation pipeline object.
    """
    logger.info(f"Loading Hugging Face model: {model_name}")

    if model_dir is None:
        # Download and save the model if not already done
        model_dir = os.path.join('../models', model_name.replace('/', '_'))

        if not os.path.exists(model_dir):
            # Attempt to log in to Hugging Face
            try:
                login(token=hf_token)
                logger.info("Logged into Hugging Face successfully.")
            except Exception as e:
                logger.warning("Failed to log into Hugging Face. Attempting to load from cache. Error: %s", e)
            os.makedirs(model_dir, exist_ok=True)

            # Downloads the model from Hugging Face Hub using snapshot_download.
            logging.info(f"Starting download of model '{model_name}'...")
            try:
                snapshot_download(repo_id=model_name, local_dir=model_dir,
                                  ignore_patterns=[".gitattributes", "LICENSE.txt", "README.md",
                                                   "USE_POLICY.md", ".cache/*", "original/*"])
                logging.info(f"Model '{model_name}' successfully downloaded to '{model_dir}'.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model '{model_name}': {e}")

    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Pad token was not set. Using EOS token as pad token.")
        else:
            logger.info("Tokenizer loaded successfully with pad token.")
    except Exception as e:
        logger.error("Error loading tokenizer: %s", e)
        raise

    # Determine device IDs
    device_ids = list(range(torch.cuda.device_count()))
    logger.info(f"Using device IDs: {device_ids}")

    # Calculate max memory per GPU
    max_memory = {}
    max_memory_ratio = 0.8
    for device_id in device_ids:
        try:
            total_mem = torch.cuda.get_device_properties(device_id).total_memory
            allocated_mem = total_mem * max_memory_ratio
            max_memory[device_id] = f"{int(allocated_mem // (1024 ** 3))}GB"
            logger.info(f"Device {device_id}: Allocating up to {max_memory[device_id]} memory.")
        except Exception as e:
            logger.error("Error retrieving memory for device %d: %s", device_id, e)
            raise

    # Load model with optimized multi-GPU support
    try:
        if len(max_memory) > 0:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,  # Use mixed precision for efficiency
                # load_in_8bit=True,  # Use 8-bit precision if supported
                offload_folder="offload",  # Folder to offload layers if necessary
                offload_state_dict=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map="auto",
            )
        logger.info("Model loaded successfully with multi-GPU support.")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise

    # Create the text-generation pipeline
    try:
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Ensure consistency with model dtype
            do_sample=True, # For deterministic behaviour
            max_new_tokens=600,
            temperature=0.8,  # Use a low temperature to prevent randomness from becoming too high
            # Additional pipeline configuration can be added here
            # e.g., generation parameters like max_length, temperature, etc.
        )
        logger.info("Text-generation pipeline created successfully.")
    except Exception as e:
        logger.error("Error creating pipeline: %s", e)
        raise

    return hf_pipeline

def query_openai(messages, azure_pipeline, deployment):
    """
    Query the OpenAI model with a conversation history.

    Parameters:
        messages (list): A list of messages as dict (role: user/assistant/system, content: str).
        model_name (str): OpenAI model name (e.g. "gpt-3.5-turbo").
        api_key (str): OpenAI API key.

    Returns:
        str: Generated content from the OpenAI model.
    """
    # Generate the completion
    completion = azure_pipeline.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        seed=42
    )

    if completion.choices[0].finish_reason == 'content_filter':
        return 'content_filter_high'

    return completion.choices[0].message.content.strip()

def query_sglang(sglang_pipeline, messages, model):
    """
    Query the SGlang with OpenAI compatible API

    Parameters:
        sglang_pipeline (object): The pipeline used to interact with the SGlang compatible OpenAI API.
        messages (list): A list of messages as dictionaries (e.g., {"role": "user", "content": "Your message"}).
        model (str): The model name (e.g., "meta-llama/Llama-3.1-405B-Instruct-FP8").

    Returns:
        str: Generated content from the model.
    """
    # Generate the completion
    completion = sglang_pipeline.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Lower temperature for more focused responses
        top_p=1,  # Slightly higher for better fluency
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        n=1,  # Single response is usually more stable
        seed=42,  # Keep for reproducibility
    )
    print(completion)

    return completion.choices[0].message.content.strip()

def query_azure_ai(foundry_pipeline, messages, model):
    """
    Query the SGlang with OpenAI compatible API

    Parameters:
        sglang_pipeline (object): The pipeline used to interact with the SGlang compatible OpenAI API.
        messages (list): A list of messages as dictionaries (e.g., {"role": "user", "content": "Your message"}).
        model (str): The model name (e.g., "meta-llama/Llama-3.1-405B-Instruct-FP8").

    Returns:
        str: Generated content from the model.
    """
    payload = {
        'messages': messages,
        #"max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "seed": 42
    }

    # Generate the completion
    completion = foundry_pipeline.complete(payload)

    return completion.choices[0].message.content.strip()

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(50),
    reraise=True
)
def fetch_with_tenacity(messages, azure_pipeline, deployment):
    return query_openai(messages, azure_pipeline, deployment)


def query_hf(messages, hf_pipeline):
    """
    Query the Hugging Face model pipeline with a single prompt.

    Parameters:
        prompt (str): Prompt text.
        hf_pipeline: Hugging Face text-generation pipeline.

    Returns:
        str: Generated text stripped of the original prompt.
    """
    output = hf_pipeline(messages)
    generated = output[0]['generated_text']
    return generated

def fetch_movie_name_with_LLM(items_df, config):
    """
    Given a DataFrame of movies with columns ["MovieID","Title"],
    use a language model to attempt to regenerate the movie title.

    Parameters:
        items_df (pd.DataFrame): DataFrame with "MovieID","Title" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["MovieID", "GeneratedTitle", "RealTitle", "ErrorFlag"].
    """
    # Validate configuration
    required_keys = ["dataset_name", "model_type", "model_name", "batch_size"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif  config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be 'openai' or 'hf'.")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # Prepare a lookup dict for faster title retrieval
    id_to_title = dict(zip(items_df['MovieID'], items_df['Title']))

    # Initialize model/pipeline based on model_type
    if model_type == "hf":
        model_name = config["model_name"]
        hf_pipeline = get_hf_pipeline(model_name, config["hf_key"], config['model_dir'])
        results_file = f"{model_name.replace('/', '_')}_results.csv"
    elif model_type == "openai":
        azure_pipeline = AzureOpenAI(
            azure_endpoint=config['azure_endpoint'],
            api_key=config['azure_openai_key'],
            api_version=config['api_version'],
        )
        results_file = f"{config['deployment_name']}_results.csv"
    elif model_type == "sglang":
        sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        results_file = f"{config['model_name'].replace('/', '_')}_results.csv"
    elif model_type == "foundry":
        foundry_pipeline = ChatCompletionsClient(
            endpoint= config['foundry_endpoint'],
            credential=AzureKeyCredential(config['foundry_api_key'])
        )
        results_file = f"{config['foundry_model_name'].replace('/', '_')}_results.csv"
    else:
        raise ValueError("Invalid model_type. Must be 'openai' or 'hf'.")

    # Initialize or load existing results
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_ids = set(existing_results['MovieID'].astype(str))
        # Extract the last three examples for prompt
        last_three = existing_results.tail(3).to_dict('records')
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['MovieID', 'GeneratedTitle', 'RealTitle', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_ids = set()
        last_three = []  # No previous examples

    # Define initial examples for the first batch if no previous examples
    initial_examples = [
        {"MovieID": str(items_df.iloc[0]['MovieID']), "RealTitle": items_df.iloc[0]['Title']},
        {"MovieID": str(items_df.iloc[1]['MovieID']), "RealTitle": items_df.iloc[1]['Title']},
        {"MovieID": str(items_df.iloc[2]['MovieID']), "RealTitle": items_df.iloc[2]['Title']},
    ]

    # Prepare all movie IDs as strings
    movie_ids = items_df["MovieID"].astype(str).tolist()
    total_ids = len(movie_ids)

    # Progress bar for better tracking
    with tqdm(total=total_ids, desc="Processing Movies") as pbar:
        for batch_start in range(0, total_ids, batch_size):
            batch_end = min(batch_start + batch_size, total_ids)
            current_batch = movie_ids[batch_start:batch_end]

            # Filter out already processed MovieIDs
            current_batch = [mid for mid in current_batch if mid not in processed_ids]
            if not current_batch:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} movies)...")

            # Select examples
            if last_three:
                examples = last_three
            else:
                # First batch: use initial_examples
                examples = initial_examples

            # Construct messages with selected examples
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are the {dataset_name} dataset. "
                        "When given a lookup key (e.g., a MovieID), you will respond with the exact corresponding value from the dataset. "
                        "Only respond with the value itself. If the key is unknown, respond with 'Unknown'. "
                        "Below are examples of queries and their correct responses:\n\n"
                        "Follow this pattern strictly. Let's think step by step."
                    )
                },
            ]

            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Input: {example['MovieID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['MovieID']}::{example['RealTitle']}\n"
                    },
                ])

            for movie_id_str in current_batch:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Input: {movie_id_str}::"
                    }
                )

                try:
                    if model_type == "openai":
                        output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                        if output == 'content_filter_high':
                            generated_title = 'Azure Content Filter Error'
                        else:
                            generated_title = output.replace(f"{movie_id_str}::", '').strip()
                    elif model_type == "hf":
                        output = query_hf(messages, hf_pipeline=hf_pipeline)
                        generated_title = output[-1]['content'].replace(f"{movie_id_str}::", '').strip()
                    elif model_type == "sglang":
                        output = query_sglang(sglang_pipeline, messages, config['model_name'])
                        generated_title = output[-1]['content'].replace(f"{movie_id_str}::", '').strip()
                    elif model_type == "foundry":
                        output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                        generated_title = output.replace(f"{movie_id_str}::", '').strip()
                except Exception as e:
                    logger.error(f"Error processing MovieID {movie_id_str}: {e}")
                    generated_title = ""

                real_title = id_to_title.get(int(movie_id_str), "Unknown")
                similarity = compute_similarity(generated_title, real_title)

                similarity_threshold = 80
                error_flag = 0 if similarity > similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{generated_title}' == Real: '{real_title}'")
                else:
                    logger.info(f"Error - Generated: '{generated_title}' <> Real: '{real_title}'")

                record = {
                    "MovieID": movie_id_str,
                    "GeneratedTitle": generated_title,
                    "RealTitle": real_title,
                    "ErrorFlag": error_flag
                }

                # Write the record immediately to the CSV file
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for MovieID {movie_id_str}: {e}")

                processed_ids.add(movie_id_str)

                # Update last_three examples
                if error_flag == 0:
                    last_three.append(record)
                    if len(last_three) > 3:
                        last_three.pop(0)
                else:
                    # Optionally, handle errors (e.g., retry, skip)
                    pass

                # Append the assistant's response to messages to maintain context
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"{movie_id_str}::{real_title}\n"
                    }
                )

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")
            pbar.update(batch_end - batch_start)

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results

def fetch_next_interaction_with_LLM(interactions_df, config):
    """
    Given a DataFrame of interactions with columns ["UserID","MovieID"],
    call a language model to guess (or “continue”) the next user–item interaction,
    using a row-based approach:
      1. Create a dictionary `interactions_row` keyed by row index, with
         values in the format: 'UserID::MovieID'.
      2. Batch-process rows (instead of unique user IDs).

    Parameters:
        interactions_df (pd.DataFrame): DataFrame with "UserID","MovieID" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["RowIndex", "InteractionString",
                                             "GeneratedOutput", "ErrorFlag"].
    """

    # -------------------------------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------------------------------
    required_keys = ["dataset_name", "model_type", "model_name", "batch_size"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be one of ['openai','hf','sglang','foundry'].")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # -------------------------------------------------------------------------
    # 2. Create interactions_row dict
    #    Key: row index (int)
    #    Value: "UserID::MovieID"
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in interactions_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['MovieID']}"

    # -------------------------------------------------------------------------
    # 3. Initialize model/pipeline based on model_type
    #    (Below lines are placeholders; replace with your actual code)
    # -------------------------------------------------------------------------
    if model_type == "hf":
        model_name = config["model_name"]
        hf_pipeline = get_hf_pipeline(model_name, config["hf_key"], config['model_dir'])
        results_file = f"{model_name.replace('/', '_')}_interaction_results.csv"
    elif model_type == "openai":
        azure_pipeline = AzureOpenAI(
            azure_endpoint=config['azure_endpoint'],
            api_key=config['azure_openai_key'],
            api_version=config['api_version'],
        )
        results_file = f"{config['deployment_name']}_interaction_results.csv"
    elif model_type == "sglang":
        sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        results_file = f"{config['model_name'].replace('/', '_')}_interaction_results.csv"
    elif model_type == "foundry":
        foundry_pipeline = ChatCompletionsClient(
            endpoint=config['foundry_endpoint'],
            credential=AzureKeyCredential(config['foundry_api_key'])
        )
        results_file = f"{config['foundry_model_name'].replace('/', '_')}_interaction_results.csv"
    else:
        raise ValueError("Invalid model_type. Must be 'openai' or 'hf'.")

    # -------------------------------------------------------------------------
    # 4. Initialize or load existing results
    # -------------------------------------------------------------------------
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_rows = set(existing_results['RowIndex'].astype(int))
        # Read last few lines for example context
        last_three_record = existing_results.tail(3).to_dict('records')

        # Transform each record into a dictionary with UserID and MovieID
        last_three = [
            dict(zip(['UserID', 'MovieID'], record['RealInteraction'].split('::')))
            for record in last_three_record
        ]
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['RowIndex', 'GeneratedInteraction', 'RealInteraction', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_rows = set()
        last_three = []  # No previous examples

    # -------------------------------------------------------------------------
    # 5. Define initial examples (few-shot) if no previous examples
    # -------------------------------------------------------------------------
    # We can create up to 3 “initial examples” from the first rows:
    initial_examples = []
    max_rows_for_examples = min(3, len(interactions_df))
    for i in range(max_rows_for_examples):
        row = interactions_df.iloc[i]
        init_string = f"{row['UserID']}::{row['MovieID']}"
        example = {"RowIndex": i, "UserID":row['UserID'], "MovieID":row['MovieID'], "InteractionString": init_string}
        initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Prepare row-based batching
    #    Instead of unique user IDs, we simply iterate over the row indices.
    # -------------------------------------------------------------------------
    total_rows = len(interactions_df)

    # -------------------------------------------------------------------------
    # 7. Main loop over row-based batches
    # -------------------------------------------------------------------------
    with tqdm(total=total_rows, desc="Processing Rows") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            # current_batch is a slice of interactions_df by row index
            current_batch = interactions_df.iloc[batch_start:batch_end]

            # Filter out already processed interactions
            current_batch = current_batch[~current_batch.index.isin(processed_rows)]

            if current_batch.empty:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} rows)...")

            # 7a. Select examples
            if last_three:
                examples = last_three
            else:
                examples = initial_examples

            # 7b. Construct the system + example messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Recommender Systems. "
                        "Continue user-item interactions list providing the next interaction "
                        f"based on the {dataset_name} dataset. "
                        "When given 'UserID, CurrentInteraction', respond with 'UserID, NextInteraction'. "
                        "If the next interaction is unknown, respond with 'Unknown'. "
                        #"If there's no next interaction for that user, respond with 'No next interaction'. "
                        "\nBelow are examples of queries and their correct responses:\n\n"
                        "Follow this pattern strictly. Let's think step by step."
                    )
                }
            ]
            # Add the few-shot examples from either `last_three` or `initial_examples`
            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"{example['UserID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['UserID']}::{example['MovieID']}"
                    },
                ])

            # 7c. Iterate through the rows in the current batch
            for i, row in current_batch.iterrows():
                if i in processed_rows:
                    # If already processed in a previous run, skip
                    pbar.update(1)
                    continue

                # Look up the "UserID" string
                interaction = interactions_df.iloc[i]

                # Append the new user query
                messages.append(
                    {
                        "role": "user",
                        "content": f"{interaction['UserID']}::"
                    }
                )

                # LLM call
                try:
                    if model_type == "openai":
                        output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                        if output == 'content_filter_high':
                            generated_interaction = 'Azure Content Filter Error'
                        else:
                            generated_interaction = output.strip()

                    elif model_type == "hf":
                        output = query_hf(messages, hf_pipeline=hf_pipeline)
                        generated_interaction = output[-1]['content'].strip()

                    elif model_type == "sglang":
                        output = query_sglang(sglang_pipeline, messages, config['model_name'])
                        generated_interaction = output[-1]['content'].strip()

                    elif model_type == "foundry":
                        output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                        generated_interaction = output.strip()

                    else:
                        raise ValueError(f"Unsupported model_type: {model_type}")

                except Exception as e:
                    logger.error(f"Error processing row {i}: {e}")
                    generated_interaction = "Error"

                # ------------------------------------------------------------------
                # (Optional) If you want to compare to some "real next interaction"
                # or do an error check, define it here. We'll do a dummy check:
                # ------------------------------------------------------------------
                error_flag = 0  # or 1 if some condition fails

                real_interaction = interactions_row[i]
                similarity = compute_similarity(generated_interaction, real_interaction)

                similarity_threshold = 80
                error_flag = 0 if similarity > similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{generated_interaction}' == Real: '{real_interaction}'")
                else:
                    logger.info(f"Error - Generated: '{generated_interaction}' <> Real: '{real_interaction}'")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "RowIndex": i,
                    "GeneratedInteraction": generated_interaction,
                    "RealInteraction": real_interaction,
                    "ErrorFlag": error_flag
                }
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for row {i}: {e}")

                processed_rows.add(i)

                # ------------------------------------------------------------------
                # 7d. Update last_three examples (few-shot) if you want the new row
                #     to become an example.
                # ------------------------------------------------------------------
                # For instance, if there's no error:
                if error_flag == 0:
                    # We'll keep 'InteractionString' as the same.
                    # Or you could parse the model’s response if you want the "next item" specifically.
                    last_three.append(dict(zip(['UserID', 'MovieID'], record['RealInteraction'].split('::'))))
                    if len(last_three) > 3:
                        last_three.pop(0)

                # Add the assistant's response to messages for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": real_interaction
                    }
                )

                pbar.update(1)

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results

def fetch_next_user_interaction_with_LLM(interactions_df, config):
    """
    Given a DataFrame of interactions with columns ["UserID","MovieID"],
    call a language model to guess (or “continue”) the next user–item interaction,
    defining the batch based on the current user:
      1. Create a dictionary `interactions_row` keyed by row index, with
         values in the format: 'UserID::MovieID'.
      2. Process the interactions in batches *per user*, e.g. for each user,
         gather all their interactions and move on to the next user.

    Parameters:
        interactions_df (pd.DataFrame): DataFrame with "UserID","MovieID" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["RowIndex", "InteractionString",
                                             "GeneratedOutput", "ErrorFlag"].
    """

    # -------------------------------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------------------------------
    required_keys = ["dataset_name", "model_type", "model_name", "batch_size"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be one of ['openai','hf','sglang','foundry'].")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # -------------------------------------------------------------------------
    # 2. Create interactions_row dict
    #    Key: row index (int)
    #    Value: "UserID::MovieID"
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in interactions_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['MovieID']}"

    # -------------------------------------------------------------------------
    # 3. Initialize model/pipeline based on model_type
    #    (Below lines are placeholders; replace with your actual code)
    # -------------------------------------------------------------------------
    if model_type == "hf":
        model_name = config["model_name"]
        hf_pipeline = get_hf_pipeline(model_name, config["hf_key"], config['model_dir'])
        results_file = f"{model_name.replace('/', '_')}_interaction_results.csv"
    elif model_type == "openai":
        azure_pipeline = AzureOpenAI(
            azure_endpoint=config['azure_endpoint'],
            api_key=config['azure_openai_key'],
            api_version=config['api_version'],
        )
        results_file = f"{config['deployment_name']}_interaction_results.csv"
    elif model_type == "sglang":
        sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        results_file = f"{config['model_name'].replace('/', '_')}_interaction_results.csv"
    elif model_type == "foundry":
        foundry_pipeline = ChatCompletionsClient(
            endpoint=config['foundry_endpoint'],
            credential=AzureKeyCredential(config['foundry_api_key'])
        )
        results_file = f"{config['foundry_model_name'].replace('/', '_')}_interaction_results.csv"
    else:
        raise ValueError("Invalid model_type. Must be 'openai' or 'hf'.")

    # -------------------------------------------------------------------------
    # 4. Initialize or load existing results
    # -------------------------------------------------------------------------
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_rows = set(existing_results['RowIndex'].astype(int))

        # Example: We won't keep a global last_three, because we will do it per user
        # in the user-based loop. If you still want global examples, adapt as needed.
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['RowIndex', 'GeneratedInteraction', 'RealInteraction', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_rows = set()

    # -------------------------------------------------------------------------
    # 5. Prepare initial examples (few-shot) if you'd like them to be global
    #    E.g. from the first 3 rows overall. (Optional)
    # -------------------------------------------------------------------------
    global_initial_examples = []
    max_rows_for_examples = min(3, len(interactions_df))
    for idx in range(max_rows_for_examples):
        row = interactions_df.iloc[idx]
        init_string = f"{row['UserID']}::{row['MovieID']}"
        example = {
            "RowIndex": idx,
            "UserID": row['UserID'],
            "MovieID": row['MovieID'],
            "InteractionString": init_string
        }
        global_initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Unique users and user-based iteration
    # -------------------------------------------------------------------------
    unique_users = interactions_df['UserID'].unique()
    total_users = len(unique_users)

    with tqdm(total=total_users, desc="Processing Users") as user_pbar:
        for user_id in unique_users:
            user_interactions = interactions_df[interactions_df['UserID'] == user_id].copy()

            # We will maintain a user-specific "last_three" so that each user has independent context
            # If you need a global context across users, you can move this outside the user loop
            last_three = []

            # If you want to reuse previously processed examples for *this* user, you could load them here:
            # e.g. rows_of_this_user_in_existing_results, parse them into last_three, etc.

            # Now we batch the user's interactions
            num_user_rows = len(user_interactions)
            user_pbar.set_description_str(f"User {user_id} ({num_user_rows} rows)")

            # For large numbers of interactions per user, we still use batch_size
            for batch_start in range(0, num_user_rows, batch_size):
                batch_end = min(batch_start + batch_size, num_user_rows)
                current_batch = user_interactions.iloc[batch_start:batch_end]

                # Filter out already processed interactions
                # (keep them from re-processing if you re-run the script)
                current_batch = current_batch[~current_batch.index.isin(processed_rows)]

                if current_batch.empty:
                    continue

                tqdm.write(f"\nUser {user_id}: processing batch {batch_start + 1}-{batch_end} of {num_user_rows} ...")

                # 6a. Select few-shot examples for this user.
                #     We can mix user-specific 'last_three' with global examples, if desired.
                if last_three:
                    examples = last_three
                else:
                    # Start with global examples or an empty list
                    examples = global_initial_examples

                # 6b. Construct the system + example messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a Recommender Systems. "
                            "Continue user-item interactions list providing the next interaction "
                            f"based on the {dataset_name} dataset. "
                            "When given 'UserID::CurrentInteraction', respond with 'UserID::NextInteraction'. "
                            "\nBelow are examples of queries and their correct responses:\n\n"
                            "Follow this pattern strictly. Let's think step by step."
                        )
                    }
                ]
                # Add the few-shot examples
                for example in examples:
                    messages.extend([
                        {
                            "role": "user",
                            "content": f"{example['UserID']}::"
                        },
                        {
                            "role": "assistant",
                            "content": f"{example['UserID']}::{example['MovieID']}"
                        },
                    ])

                # 6c. Iterate through this batch of rows for the current user
                for i, row in current_batch.iterrows():
                    # Safety check in case we re-run:
                    if i in processed_rows:
                        continue

                    # Build user prompt from the row
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{row['UserID']}::"
                        }
                    )

                    # LLM call
                    try:
                        if model_type == "openai":
                            output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                            if output == 'content_filter_high':
                                generated_interaction = 'Azure Content Filter Error'
                            else:
                                generated_interaction = output.strip()

                        elif model_type == "hf":
                            output = query_hf(messages, hf_pipeline=hf_pipeline)
                            generated_interaction = output[-1]['content'].strip()

                        elif model_type == "sglang":
                            output = query_sglang(sglang_pipeline, messages, config['model_name'])
                            generated_interaction = output[-1]['content'].strip()

                        elif model_type == "foundry":
                            output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                            generated_interaction = output.strip()

                        else:
                            raise ValueError(f"Unsupported model_type: {model_type}")

                    except Exception as e:
                        logger.error(f"Error processing row {i}: {e}")
                        generated_interaction = "Error"

                    # Optional: some error-check or similarity check
                    real_interaction = interactions_row[i]
                    similarity = compute_similarity(generated_interaction, real_interaction)
                    similarity_threshold = 90
                    error_flag = 0 if similarity > similarity_threshold else 1

                    if error_flag == 0:
                        logger.info(f"Correct - Generated: '{generated_interaction}' == Real: '{real_interaction}'")
                    else:
                        logger.info(f"Error   - Generated: '{generated_interaction}' <> Real: '{real_interaction}'")

                    # ------------------------------------------------------------------
                    # Save to CSV
                    # ------------------------------------------------------------------
                    record = {
                        "RowIndex": i,
                        "GeneratedInteraction": generated_interaction,
                        "RealInteraction": real_interaction,
                        "ErrorFlag": error_flag
                    }
                    try:
                        pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                    except Exception as e:
                        logger.error(f"Failed to write record for row {i}: {e}")

                    processed_rows.add(i)

                    # ------------------------------------------------------------------
                    # 6d. Update user-specific 'last_three' examples if no error
                    # ------------------------------------------------------------------
                    if error_flag == 0:
                        last_three.append(dict(zip(['UserID','MovieID'], record['RealInteraction'].split('::'))))
                        if len(last_three) > 3:
                            last_three.pop(0)

                    # Add the assistant's response for context (optional)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": real_interaction
                        }
                    )

            user_pbar.update(1)

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results, results_file

def fetch_user_attribute_with_LLM(user_df, config):
    """
    Given a DataFrame of interactions with columns ["UserID","MovieID"],
    call a language model to guess (or “continue”) the next user–item interaction,
    using a row-based approach:
      1. Create a dictionary `interactions_row` keyed by row index, with
         values in the format: 'UserID::MovieID'.
      2. Batch-process rows (instead of unique user IDs).

    The system prompt is set up for a Recommender System:
      "You are a Recommender Systems. Continue user-item interactions list
       providing the next interaction based on the MovieLens1M dataset.
       When given 'UserID, CurrentInteraction', respond with 'UserID, NextInteraction'.
       If there's no next interaction for that user, respond with 'No next interaction'."

    Parameters:
        user_df (pd.DataFrame): DataFrame with "UserID","MovieID" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["RowIndex", "InteractionString",
                                             "GeneratedOutput", "ErrorFlag"].
    """

    # -------------------------------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------------------------------
    required_keys = ["dataset_name", "model_type", "model_name", "batch_size"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be one of ['openai','hf','sglang','foundry'].")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # -------------------------------------------------------------------------
    # 2. Create interactions_row dict
    #    Key: row index (int)
    #    Value: "UserID::MovieID"
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in user_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"

    # -------------------------------------------------------------------------
    # 3. Initialize model/pipeline based on model_type
    #    (Below lines are placeholders; replace with your actual code)
    # -------------------------------------------------------------------------
    if model_type == "hf":
        model_name = config["model_name"]
        hf_pipeline = get_hf_pipeline(model_name, config["hf_key"], config['model_dir'])
        results_file = f"{model_name.replace('/', '_')}_users_results.csv"
    elif model_type == "openai":
        azure_pipeline = AzureOpenAI(
            azure_endpoint=config['azure_endpoint'],
            api_key=config['azure_openai_key'],
            api_version=config['api_version'],
        )
        results_file = f"{config['deployment_name']}_users_results.csv"
    elif model_type == "sglang":
        sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        results_file = f"{config['model_name'].replace('/', '_')}_users_results.csv"
    elif model_type == "foundry":
        foundry_pipeline = ChatCompletionsClient(
            endpoint=config['foundry_endpoint'],
            credential=AzureKeyCredential(config['foundry_api_key'])
        )
        results_file = f"{config['foundry_model_name'].replace('/', '_')}_users_results.csv"
    else:
        raise ValueError("Invalid model_type. Must be 'openai' or 'hf'.")

    # -------------------------------------------------------------------------
    # 4. Initialize or load existing results
    # -------------------------------------------------------------------------
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_rows = set(existing_results['RowIndex'].astype(int))
        # Read last few lines for example context
        last_three_record = existing_results.tail(3).to_dict('records')

        # Transform each record into a dictionary with UserID and MovieID
        last_three = [
            dict(zip(['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], record['RealUser'].split('::')))
            for record in last_three_record
        ]
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['RowIndex', 'GeneratedUser', 'RealUser', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_rows = set()
        last_three = []  # No previous examples

    # -------------------------------------------------------------------------
    # 5. Define initial examples (few-shot) if no previous examples
    # -------------------------------------------------------------------------
    # We can create up to 3 “initial examples” from the first rows:
    initial_examples = []
    max_rows_for_examples = min(3, len(user_df))
    for i in range(max_rows_for_examples):
        row = user_df.iloc[i]
        init_string = f"{row['UserID']}::{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"
        example = {"RowIndex": i, "UserID":row['UserID'],
                   "Gender":row['Gender'], "Age":row['Age'],
                   "Occupation":row['Occupation'], "Zip-code":row['Zip-code'],
                   "InteractionString": init_string}
        initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Prepare row-based batching
    #    Instead of unique user IDs, we simply iterate over the row indices.
    # -------------------------------------------------------------------------
    total_rows = len(user_df)

    # -------------------------------------------------------------------------
    # 7. Main loop over row-based batches
    # -------------------------------------------------------------------------
    with tqdm(total=total_rows, desc="Processing Rows") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            # current_batch is a slice of interactions_df by row index
            current_batch = user_df.iloc[batch_start:batch_end]

            # Filter out already processed interactions
            current_batch = current_batch[~current_batch.index.isin(processed_rows)]

            if current_batch.empty:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} rows)...")

            # 7a. Select examples
            if last_three:
                examples = last_three
            else:
                examples = initial_examples

            # 7b. Construct the system + example messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are the {dataset_name} dataset."
                        "When given a lookup key (e.g., a UserID), you will respond with the exact corresponding Gender,Age,Occupation,Zip-code value from the dataset."
                        "Only respond with the values itself. If the key is unknown, respond with 'Unknown'."
                        "Below are examples of queries and their correct responses:"
                        "Follow this pattern strictly. Let's think step by step.\n\n"
                    )
                }
            ]
            # Add the few-shot examples from either `last_three` or `initial_examples`
            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"{example['UserID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['UserID']}::{example['Gender']}::{example['Age']}::{example['Occupation']}::{example['Zip-code']}"
                    },
                ])

            # 7c. Iterate through the rows in the current batch
            for i, row in current_batch.iterrows():
                if i in processed_rows:
                    # If already processed in a previous run, skip
                    pbar.update(1)
                    continue

                # Look up the "UserID" string
                interaction = user_df.iloc[i]

                # Append the new user query
                messages.append(
                    {
                        "role": "user",
                        "content": f"{interaction['UserID']}::"
                    }
                )

                # LLM call
                try:
                    if model_type == "openai":
                        output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                        if output == 'content_filter_high':
                            generated_user = 'Azure Content Filter Error'
                        else:
                            generated_user = output.strip()

                    elif model_type == "hf":
                        output = query_hf(messages, hf_pipeline=hf_pipeline)
                        generated_user = output[-1]['content'].strip()

                    elif model_type == "sglang":
                        output = query_sglang(sglang_pipeline, messages, config['model_name'])
                        generated_user = output[-1]['content'].strip()

                    elif model_type == "foundry":
                        output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                        generated_user = output.strip()

                    else:
                        raise ValueError(f"Unsupported model_type: {model_type}")

                except Exception as e:
                    logger.error(f"Error processing row {i}: {e}")
                    generated_user = "Error"

                # ------------------------------------------------------------------
                # (Optional) If you want to compare to some "real next interaction"
                # or do an error check, define it here. We'll do a dummy check:
                # ------------------------------------------------------------------
                error_flag = 0  # or 1 if some condition fails

                real_user = interactions_row[i]
                similarity = compute_similarity(generated_user, real_user)

                similarity_threshold = 90
                error_flag = 0 if similarity >= similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{generated_user}' == Real: '{real_user}'")
                else:
                    logger.info(f"Error - Generated: '{generated_user}' <> Real: '{real_user}'")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "RowIndex": i,
                    "GeneratedInteraction": generated_user,
                    "RealUser": real_user,
                    "ErrorFlag": error_flag
                }
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for row {i}: {e}")

                processed_rows.add(i)

                # ------------------------------------------------------------------
                # 7d. Update last_three examples (few-shot) if you want the new row
                #     to become an example.
                # ------------------------------------------------------------------
                # For instance, if there's no error:
                if error_flag == 0:
                    # We'll keep 'InteractionString' as the same.
                    # Or you could parse the model’s response if you want the "next item" specifically.
                    last_three.append(dict(zip(['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], record['RealUser'].split('::'))))
                    if len(last_three) > 3:
                        last_three.pop(0)

                # Add the assistant's response to messages for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": real_user
                    }
                )

                pbar.update(1)

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results


# -----------------------------------------------------------------------------
# Title normalization and fuzzy matching helpers.
# -----------------------------------------------------------------------------
def normalize_title(title):
    """
    Normalize the movie title for matching by lowercasing,
    stripping whitespace, and removing punctuation.
    """
    title = title.lower().strip()
    # Remove punctuation (non-alphanumeric characters except whitespace)
    title = re.sub(r'[^\w\s]', '', title)
    return title


def is_similar(title1, title2, threshold=80):
    """
    Return True if the fuzzy similarity ratio between title1 and title2 is at least the threshold.
    """
    return fuzz.ratio(title1, title2) >= threshold

def create_train_test_files(interactions_df, train_file="training.tsv", test_file="test.tsv", train_ratio=0.8):
    """
    Splits the interactions DataFrame into training and test sets using the hold-out method.
    For each user, the first (train_ratio*100)% of interactions are saved to the training file
    and the remaining interactions are saved to the test file.

    Parameters:
        interactions_df (pd.DataFrame): DataFrame containing at least the columns ["UserID", "Title"].
                                        (Rows must be in the desired order, e.g., chronological.)
        train_file (str): Path to the training TSV file to be created.
        test_file (str): Path to the test TSV file to be created.
        train_ratio (float): The fraction of interactions per user to include in training (default 0.8).

    Returns:
        None. The function writes the split data to `train_file` and `test_file`.
    """
    train_rows = []
    test_rows = []

    # Process each user individually.
    for user_id, user_df in interactions_df.groupby("UserID"):
        # Ensure that the interactions are in order (if not, sort by a timestamp column if available).
        user_df = user_df.reset_index(drop=True)
        n_interactions = len(user_df)

        # If a user has less than 2 interactions, put everything in training.
        if n_interactions < 2:
            train_rows.append(user_df)
        else:
            train_cutoff = int(math.floor(train_ratio * n_interactions))
            # Ensure that at least one interaction is in training.
            if train_cutoff < 1:
                train_cutoff = 1
            # Split the data.
            train_rows.append(user_df.iloc[:train_cutoff])
            test_rows.append(user_df.iloc[train_cutoff:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    # In case no user qualifies for testing, create an empty DataFrame with the same columns.
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(
        columns=interactions_df.columns)

    # Save to TSV files.
    train_df.to_csv(train_file, sep="\t", index=False)
    test_df.to_csv(test_file, sep="\t", index=False)
    logger.info(f"Training data saved to {train_file} ({len(train_df)} rows).")
    logger.info(f"Test data saved to {test_file} ({len(test_df)} rows).")


def leave_n_out_recommendation_with_LLM(training_file, test_file, config):
    """
    Reads pre-split training and test TSV files and, for each user in the training set,
    uses their training history to prompt an LLM to recommend a ranked list of 50 movies.
    The withheld test set (from the test TSV) is then used to compute HR@K, nDCG@K, and MRR@K
    (with K = 1, 3, 5, 10, 20, 50) across users.

    The recommendations are saved in a TSV file with columns: UserID, Title, Rank.
    Checkpoints are saved along the way to avoid losing progress.

    Parameters:
        training_file (str): Path to the TSV file with training interactions.
        test_file (str): Path to the TSV file with test interactions.
        config (dict): Configuration dictionary. Required keys include:
            - dataset_name
            - model_type (one of 'openai', 'hf', 'sglang', 'foundry')
            - model_name
            plus model-specific keys (e.g., hf_key, azure_endpoint, etc.)

    Returns:
        recommendations_df (pd.DataFrame): DataFrame with columns ["UserID", "Title", "Rank"].
        metrics (dict): A dictionary where keys are cutoff values (K) and the values are dicts with
                        HR@K, nDCG@K, and MRR@K.
        output_file (str): Path to the TSV file with recommendations.
    """
    # -------------------------------------------------------------------------
    # 1. Validate configuration parameters.
    # -------------------------------------------------------------------------
    required_keys = ["dataset_name", "model_type", "model_name"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be one of ['openai','hf','sglang','foundry'].")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]

    # -------------------------------------------------------------------------
    # 2. Initialize the appropriate LLM pipeline and setup checkpoints.
    # -------------------------------------------------------------------------
    if model_type == "hf":
        hf_pipeline = get_hf_pipeline(config["model_name"], config["hf_key"], config.get('model_dir', None))
        checkpoint_file = f"{config['model_name'].replace('/', '_')}_checkpoint.pkl"
        output_file = f"{config['model_name'].replace('/', '_')}_recommendations.tsv"
    elif model_type == "openai":
        azure_pipeline = AzureOpenAI(
            azure_endpoint=config['azure_endpoint'],
            api_key=config['azure_openai_key'],
            api_version=config['api_version'],
        )
        checkpoint_file = f"{config['deployment_name'].replace('/', '_')}_checkpoint.pkl"
        output_file = f"{config['deployment_name'].replace('/', '_')}_recommendations.tsv"
    elif model_type == "sglang":
        sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
        checkpoint_file = f"{config['model_name'].replace('/', '_')}_checkpoint.pkl"
        output_file = f"{config['model_name'].replace('/', '_')}_recommendations.tsv"
    elif model_type == "foundry":
        foundry_pipeline = ChatCompletionsClient(
            endpoint=config['foundry_endpoint'],
            credential=AzureKeyCredential(config['foundry_api_key'])
        )
        checkpoint_file = f"{config['foundry_model_name'].replace('/', '_')}_checkpoint.pkl"
        output_file = f"{config['foundry_model_name'].replace('/', '_')}_recommendations.tsv"
    else:
        raise ValueError("Invalid model_type.")

    items_df = pd.read_csv(config["item_data_path"])

    # Read the pre-split training and test data.
    train_df = pd.read_csv(training_file, sep="\t")
    train_df = pd.merge(train_df, items_df[['MovieID', 'Title']], on='MovieID', how='inner')
    test_df = pd.read_csv(test_file, sep="\t")
    test_df = pd.merge(test_df, items_df[['MovieID', 'Title']], on='MovieID', how='inner')
    # Cutoff values for evaluation metrics.
    cutoffs = [1, 3, 5, 10, 20, 50]
    if os.path.exists(checkpoint_file):
        logger.info(f"Checkpoint file {checkpoint_file} found. Resuming from checkpoint...")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        processed_users = checkpoint.get("processed_users", set())
        recommendations = checkpoint.get("recommendations", [])
        hr_scores = checkpoint.get("hr_scores", {k: [] for k in cutoffs})
        ndcg_scores = checkpoint.get("ndcg_scores", {k: [] for k in cutoffs})
        mrr_scores = checkpoint.get("mrr_scores", {k: [] for k in cutoffs})
    else:
        logger.info("No checkpoint file found. Starting a new experiment.")
        processed_users = set()
        recommendations = []
        hr_scores = {k: [] for k in cutoffs}
        ndcg_scores = {k: [] for k in cutoffs}
        mrr_scores = {k: [] for k in cutoffs}

    def save_checkpoint():
        cp = {
            "processed_users": processed_users,
            "recommendations": recommendations,
            "hr_scores": hr_scores,
            "ndcg_scores": ndcg_scores,
            "mrr_scores": mrr_scores
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(cp, f)
        logger.info(f"Checkpoint saved: {len(processed_users)} users processed so far.")

    # -------------------------------------------------------------------------
    # 3. Process each user: generate recommendations and compute metrics.
    # -------------------------------------------------------------------------
    unique_users = train_df['UserID'].unique()
    for user_id in tqdm(unique_users, desc="Processing Users"):
        if user_id in processed_users:
            continue  # Skip already processed users

        # Get the pre-split interactions for the user.
        training_history = train_df[train_df['UserID'] == user_id]
        test_set = test_df[test_df['UserID'] == user_id]

        # Skip users that do not have both training and test interactions.
        if len(training_history) < 1 or len(test_set) < 1:
            processed_users.add(user_id)
            save_checkpoint()
            continue

        # Prepare the list of movie titles.
        training_titles = training_history['Title'].astype(str).tolist()
        test_titles = test_set['Title'].astype(str).tolist()
        training_history_str = ", ".join(training_titles)

        # Construct the prompt to instruct the LLM.
        prompt = (
            f"User {user_id} has interacted with the following movies: {training_history_str}. "
            "Based solely on these interactions, please generate a ranked list of exactly 50 movie recommendations. "
            "Output only the list with no additional commentary or explanation. "
            "Each recommendation must be on a new line in the exact format: 'Rank. Title' (for example: '1. Harry Potter')."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a movie recommendation system for the {dataset_name} dataset. "
                    "Based on the user's past interactions, generate a ranked list of exactly 50 new movie recommendations. "
                    "Your output must contain only the list in the following format: one line per recommendation in the exact format 'Rank. Title' (e.g., '1. Harry Potter'). "
                    "Do not include any additional text, commentary, or explanation."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # ---------------------------------------------------------------------
        # 4a. Query the selected LLM for recommendations.
        # ---------------------------------------------------------------------
        try:
            if model_type == "openai":
                output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                llm_response = output.strip()
            elif model_type == "hf":
                output = query_hf(messages, hf_pipeline=hf_pipeline)
                llm_response = output[-1]['content'].strip()
            elif model_type == "sglang":
                output = query_sglang(sglang_pipeline, messages, config['model_name'])
                llm_response = output[-1]['content'].strip()
            elif model_type == "foundry":
                output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                llm_response = output.strip()
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")
        except Exception as e:
            logger.error(f"Error processing recommendations for user {user_id}: {e}")
            # Do not mark this user as processed so that it can be retried later.
            continue

        # ---------------------------------------------------------------------
        # 4b. Parse the LLM response.
        # Expected format per line: "Rank. Title" (e.g., "1. Harry Potter")
        # ---------------------------------------------------------------------
        recs = []
        for line in llm_response.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\d+)\.\s*(.+)", line)
            if match:
                rank = int(match.group(1))
                rec_title = match.group(2).strip()
                recs.append((rank, rec_title))

        if len(recs) < 50:
            logger.warning(f"User {user_id}: received only {len(recs)} recommendations (expected 50).")
        # Ensure the recommendations are sorted by rank.
        recs = sorted(recs, key=lambda x: x[0])

        # Save the recommendations for this user.
        for rank, rec_title in recs:
            recommendations.append({
                "UserID": user_id,
                "Title": rec_title,
                "Rank": rank
            })

        # ---------------------------------------------------------------------
        # 4c. Compute evaluation metrics for the withheld (test) titles.
        # ---------------------------------------------------------------------
        for k in cutoffs:
            # Select recommendations with rank <= k.
            top_k = [(rank, rec_title) for rank, rec_title in recs if rank <= k]
            hit_flag = 0
            first_relevant_rank = None
            DCG = 0.0
            for rank, rec_title in top_k:
                # Check if rec_title matches any test title (using normalized fuzzy matching).
                if any(is_similar(normalize_title(rec_title), normalize_title(test_title)) for test_title in
                       test_titles):
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    DCG += 1 / math.log2(rank + 1)
                    hit_flag = 1
            hr_scores[k].append(hit_flag)
            ideal_count = min(len(test_titles), k)
            ideal_DCG = sum(1 / math.log2(i + 1) for i in range(1, ideal_count + 1))
            ndcg = DCG / ideal_DCG if ideal_DCG > 0 else 0
            ndcg_scores[k].append(ndcg)
            mrr = 1 / first_relevant_rank if first_relevant_rank is not None else 0
            mrr_scores[k].append(mrr)

        if first_relevant_rank is not None:
            print(f"\nUser {user_id}: Positive match found in recommendations.")
        else:
            print(f"\nUser {user_id}: No match with test set.")

        processed_users.add(user_id)
        save_checkpoint()

    # -------------------------------------------------------------------------
    # 5. Save recommendations to a TSV file.
    # -------------------------------------------------------------------------
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved recommendations to {output_file}.")

    # -------------------------------------------------------------------------
    # 6. Compute and report overall HR@K, nDCG@K, and MRR@K.
    # -------------------------------------------------------------------------
    metrics = {}
    for k in cutoffs:
        avg_hr = np.mean(hr_scores[k]) if hr_scores[k] else 0.0
        avg_ndcg = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0.0
        avg_mrr = np.mean(mrr_scores[k]) if mrr_scores[k] else 0.0
        metrics[k] = {
            f"HR@{k}": avg_hr,
            f"nDCG@{k}": avg_ndcg,
            f"MRR@{k}": avg_mrr
        }
        logger.info(f"Cutoff {k} -> HR@{k}: {avg_hr:.4f}, nDCG@{k}: {avg_ndcg:.4f}, MRR@{k}: {avg_mrr:.4f}")

    # Optionally, you may remove the checkpoint file once the experiment is complete.
    # os.remove(checkpoint_file)

    return recommendations_df, metrics, output_file
