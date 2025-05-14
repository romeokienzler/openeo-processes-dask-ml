import os

MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./cache/model_cache")

S3_MODEL_REPO_ENDPOINT = os.environ.get("S3_MODEL_REPO_ENDPOINT", None)
S3_MODEL_REPO_ACCESS_KEY_ID = os.environ.get("S3_MODEL_REPO_ACCESS_KEY_ID", None)
S3_MODEL_REPO_SECRET_ACCESS_KEY = os.environ.get("S3_MODEL_REPO_SECRET_ACCESS_KEY", None)

# make sure that both access_key_id and access_key are set
if (S3_MODEL_REPO_ACCESS_KEY_ID is None) != (S3_MODEL_REPO_SECRET_ACCESS_KEY is None):
    raise ValueError(
        "Either S3_MODEL_REPO_ACCESS_KEY_ID or S3_MODEL_REPO_SECRET_ACCESS_KEY is set "
        "not set. You must set either both, or not set any of the two."
    )