import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "12"))
TOTAL_TIMEOUT = float(os.getenv("W6_TOTAL_TIMEOUT", "30"))
TOOL_TIMEOUT = float(os.getenv("W6_TOOL_TIMEOUT", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "4"))
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DB = Path(os.getenv("W6_DB", ROOT / "data" / "runs.sqlite3"))
CORPUS_PATH = ROOT / "data" / "corpus-v2.json"
