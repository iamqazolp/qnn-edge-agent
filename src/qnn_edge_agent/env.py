import os
from pathlib import Path

import qai_hub as hub
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NPU_DEVICE = "Samsung Galaxy S24 (Family)"


def hub_client() -> hub.Client:
    load_dotenv(REPO_ROOT / ".env")
    return hub.Client(config=hub.ClientConfig(api_token=os.environ["QUALCOMM_AIHUB_KEY"]))
