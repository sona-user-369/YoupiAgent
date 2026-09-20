import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

RODIUMAI_APIKEY = os.environ["RODIUMAI_APIKEY"]
RODIUMAI_BASE_URL = os.environ.get("RODIUMAI_BASE_URL", "https://api.rodiumai.io/v1")
RODIUMAI_MODEL = os.environ.get("RODIUMAI_MODEL", "openai/gpt-4o-mini")
RODIUMAI_EMBEDDING_MODEL = os.environ.get("RODIUMAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")
RODIUMAI_EMBEDDING_DIMS = int(os.environ.get("RODIUMAI_EMBEDDING_DIMS", "1536"))

DATA_DIR = Path(os.environ.get("YOUPI_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

EVENT_STORE_FILE = DATA_DIR / "event_store.json"
MEM0_STORAGE_DIR = DATA_DIR / "mem0"
MEM0_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
