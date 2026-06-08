from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(r"C:\Users\fynne\University\Data Science and Marketing Analytics\data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)