# LLM-MemoryInspector

LLM-MemoryInspector is a research-oriented pipeline designed to probe the 'memory' of Large Language Models (LLMs). It investigates whether specific datasets have been memorized by an LLM. Using carefully engineered prompts and analysis techniques, this tool offers insights into potential data leakage within LLMs.

---

## Table of Contents

- [Reproducing Experimental Results](#reproducing-experimental-results)
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Types](#model-types)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

---

## Reproducing Experimental Results

Follow these steps to set up your environment and reproduce the experimental results with LLM-MemoryInspector.

### 1. Create and Activate a Conda Environment

Create a new Conda environment (named `LLMInspect`) with Python 3.12 and activate it:

```bash
conda create --name LLMInspect python=3.12 -y
conda activate LLMInspect
```

### 2. Install Dependencies

Install all required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Set Up Your Hugging Face Token

If you plan to use Hugging Face models, set your Hugging Face token to allow access to the models and datasets. You can either:

- **Option A:** Update the `hf_key` field in the `config.yaml` file with your token.
- **Option B:** Set it as an environment variable in your shell:

  ```bash
  export HF_TOKEN=your_huggingface_token
  ```

You can obtain your token from your [Hugging Face account](https://huggingface.co/settings/tokens).

### 4. Run the Experiment

Execute the main pipeline script to run the experiments. For example, to run experiments using the `Llama-3.2-1B-Instruct` model, run:

```bash
cd src/
python main_pipeline.py
```

> **Tip:** To test different models, simply modify the `model_name` parameter in the `config.yaml` file and re-run the script.

### 5. Evaluate LLM Recommendations

Execute the evaluate_recommendations script to evaluate the LLM's recommendations.

```bash
cd src/
python evaluate_recommendations.py
```

---
## Overview

The LLM-MemoryInspector pipeline comprises the following main steps:

1. **Data Preparation**: Convert raw data into a standardized CSV format for processing.
2. **Prompt Construction**: Use template-based prompts to instruct the LLM on how to retrieve or reproduce data.
3. **LLM Inference**: Query an LLM—whether it be an open-source Hugging Face model, OpenAI’s API, SGlang, or Azure AI Foundry—to determine if and how the model recalls dataset information.
4. **Analysis & Reporting**: Compare the LLM responses with the original data using similarity metrics and flag potential memorization cases.

---

## Features

- **Multi-Model Support**: Use OpenAI, Hugging Face, SGlang, or Azure AI Foundry models.
- **Few-Shot Prompting**: Leverages example-driven prompts to guide the LLM.
- **Batch Processing**: Processes large datasets in configurable batches.
- **Checkpointing**: Saves intermediate results to resume interrupted experiments.
- **Modular Codebase**: Split across multiple files (data preparation, LLM querying, analysis) for easy customization and extension.

---

## Directory Structure

```plaintext
LLM-MemoryInspector/
├── config.yaml                 # Main configuration file with dataset paths and model settings.
├── LICENSE                     # License file.
├── README.md                   # This file.
├── requirements.txt            # Python dependencies.
├── data/                       # Raw and preprocessed datasets.
│   └── data_preparation.py     # Script to convert raw data into the required CSV format.
├── models/                     # Directory where downloaded Hugging Face models are stored.
├── results/                    # Intermediate outputs, final analysis results, and summary reports.
└── src/                        # Source code for the project.
    ├── analysis.py             # Functions for similarity analysis and summarizing LLM outputs.
    ├── download_hf_model.py    # Utility to download Hugging Face models for offline use.
    ├── llm_requests.py         # Handles interactions with the chosen LLM API or pipeline.
    ├── main_pipeline.py        # Orchestrates data preparation, LLM inference, and analysis.
    └── utils.py                # Utility functions (e.g., similarity computations, prompt handling).
```

---

## Prerequisites

- **Python 3.12+**
- **pip** or **conda** for package management
- **CUDA** (if using GPU-based inference for Hugging Face models)
- For Hugging Face models: the `transformers` and `torch` (or `tensorflow`) libraries.
- For OpenAI models: a valid API key.
- Other Python dependencies are listed in `requirements.txt`.

---

## Installation

1. **Clone the Repository**  
   Open your terminal and run:
   ```bash
   git clone https://github.com/yourusername/LLM-MemoryInspector.git
   cd LLM-MemoryInspector
   ```

2. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, use your preferred package manager (e.g., conda).

3. **Configure Settings**  
   Open and edit the `config.yaml` file to adjust:
   - Dataset file paths
   - Batch size
   - Model type and associated API keys/endpoints  
   (See the [Configuration](#configuration) section for details.)

---

## Configuration

The `config.yaml` file is the central place to set up your environment. It includes:

- **Dataset Configuration**:  
  Specify your dataset name and CSV file paths for movies, ratings, and user profiles.

- **Model Selection**:  
  Choose your LLM provider by setting `model_type` to one of the following: `"openai"`, `"hf"`, `"sglang"`, or `"foundry"`.

- **Hugging Face / SGlang Settings**:  
  Provide the model name, authentication token (`hf_key`), and optional offline directory (`model_dir`).

- **Azure OpenAI Settings**:  
  Set your endpoint URL, API key, API version, and deployment name.

- **Azure AI Foundry Settings**:  
  Provide the foundry model name, endpoint, and API key.

Refer to the commented sample below for guidance:

```yaml
# Dataset Configuration
dataset_name: "MovieLens 1M"
item_data_path: "../data/movielens_1M/movies.csv"
interaction_data_path: "../data/movielens_1M/ratings.csv"
user_data_path: "../data/movielens_1M/users.csv"
batch_size: 50

# Select Model Type: Options are "openai", "hf", "sglang", or "foundry"
model_type: "foundry"

# Hugging Face / SGlang Configuration
model_name: ""     # e.g., "meta-llama/Llama-3.3-70B-Instruct"
hf_key: ""         # Your Hugging Face token, if required.
model_dir: ""      # Local directory for the offline model; leave empty if not used.

# Azure OpenAI Configuration
azure_endpoint: ""
azure_openai_key: ""
api_version: ""
deployment_name: ""

# Azure AI Foundry Configuration
foundry_model_name: ""
foundry_endpoint: ""
foundry_api_key: ""
```

---

## Usage

Follow these steps to run the pipeline:

1. **Prepare the Dataset**  
   Convert your raw data into the required CSV format:
   ```bash
   python data_preparation.py
   ```
   This step processes and formats the data files so they can be queried by the LLM.

2. **Run the Main Pipeline**  
   Execute the primary script to perform LLM querying and analysis:
   ```bash
   python main_pipeline.py
   ```
   The script will:
   - Load the dataset
   - Construct and send prompts to the selected LLM
   - Analyze the responses using similarity metrics
   - Save intermediate and final results in the `results/` directory

3. **Examine the Results**  
   - **Intermediate Outputs**: `results/intermediate_results.csv`
   - **Final Results**: `results/final_results.csv`
   - **Analysis Summary**: `results/analysis_summary.txt`

Review these outputs to see if and how the LLM recalls original training data.

---

## Model Types

- **Hugging Face Models**:  
  When `model_type` is set to `"hf"`, the pipeline loads a local or Hugging Face-hosted model via the `transformers` library.

- **OpenAI Models**:  
  When using `"openai"`, the pipeline connects to the OpenAI ChatCompletion API using your provided API key and deployment settings.

- **SGlang Models**:  
  Similar to Hugging Face models, but using an SGlang-compatible interface.

- **Azure AI Foundry**:  
  When `model_type` is `"foundry"`, the pipeline uses Azure AI Foundry for LLM inference.

---

## Troubleshooting

- **File Paths and Data**:  
  Ensure that paths in `config.yaml` correctly point to your dataset files.

- **Model Loading Issues**:  
  - For Hugging Face, confirm that you have installed `transformers`, `torch`, and any other dependencies.
  - For OpenAI, double-check your API key and endpoint.
  - For Azure AI Foundry, verify that your credentials and endpoint are correct.

- **Rate Limits**:  
  If you encounter rate limits (especially with OpenAI), try reducing `batch_size` or implement further backoff strategies.

- **Environment Compatibility**:  
  Make sure you are running Python 3.12+ and that your GPU (if used) is properly configured.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Contributing

Contributions are welcome! If you have ideas for improvements or bug fixes, please open an issue or submit a pull request. Before contributing, please review the contribution guidelines.