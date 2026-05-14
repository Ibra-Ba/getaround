from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

api = HfApi()

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id= os.getenv("REPO_ID"),
    repo_type="space",
    token=os.getenv("HF_TOKEN"),
)

api.upload_file(
    path_or_fileobj="encoders.joblib",
    path_in_repo="encoders.joblib",
    repo_id= os.getenv("REPO_ID"),
    repo_type="space",
    token=os.getenv("HF_TOKEN"),
)

print("✅ Artefacts uploadés")