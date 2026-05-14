from huggingface_hub import HfApi
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path("/home/ibraba/bloc5-deployment/getaround")

DELAY_FILE = PROJECT_ROOT / "data" / "get_around_delay_analysis.xlsx"
api = HfApi()
api.upload_file(
    path_or_fileobj=DELAY_FILE,
    path_in_repo="data/get_around_delay_analysis.xlsx",
    repo_id="VoxUp/getaround-dashboard",
    repo_type="space",
    token=os.getenv("HF_TOKEN"),
)
print("✅ Dataset uploadé")
