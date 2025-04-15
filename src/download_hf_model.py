from huggingface_hub import snapshot_download, login

# Model repository details
repo_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"

try:
    login(token="hf_token")
    # Download the entire repository
    model_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir="./models",
        ignore_patterns=[".gitattributes", "LICENSE.txt", "README.md", "USE_POLICY.md", ".cache/*", "original/*"],
        resume_download=True,  # Enables resumption of interrupted downloads
    )
    print(f"Model repository downloaded at: {model_dir}")

except Exception as e:
    print(f"An error occurred during the download: {e}")