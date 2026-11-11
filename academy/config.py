"""Central settings. Edit here, not in the views."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ACADEMY_DATA_DIR", ROOT / "data"))
PROGRESS_FILE = DATA_DIR / "progress.json"

APP_NAME = "Sports Analytics Academy"
SCHEMA_VERSION = 1

# Streaks and timestamps use this zone (Streamlit Cloud servers run in UTC).
TIMEZONE = "America/New_York"

# A lesson is "completed" at this best quiz score, "mastered" at the higher one.
COMPLETE_THRESHOLD = 0.70
MASTERY_THRESHOLD = 0.90

# Sidebar nudges you to export a backup after this many unexported changes.
BACKUP_NUDGE_AFTER = 5

# Streamlit Community Cloud mounts apps under /mount/src.
IS_CLOUD = str(ROOT).startswith("/mount/src")
