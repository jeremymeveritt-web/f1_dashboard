"""Configuration constants and path definitions for the F1 Dashboard."""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [CACHE_DIR, DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API & Data Constants
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON_RANGE = list(range(2018, 2026))
TARGET_CIRCUITS = ["monza", "spa", "silverstone", "monaco", "suzuka"]
SESSION_TYPES = ['Q', 'R', 'FP1', 'FP2', 'FP3', 'S']

# Constructor styling (2025 colors)
CONSTRUCTOR_COLORS = {
    "red_bull": "#3671C6",
    "mercedes": "#27F4D2",
    "ferrari": "#E8002D",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#0093CC",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "sauber": "#52E252",
    "haas": "#B6BABD"
}

# Machine Learning Constants
FEATURE_COLUMNS = [
    "driver_track_affinity",
    "driver_form_3race",
    "constructor_quali_gap",
    "grid_position_normalized",
    "season_progress",
    "is_wet_race"
]
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"