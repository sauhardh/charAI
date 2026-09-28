from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

"""
1) Download metadata
"""
API_URL = "https://xeno-canto.org/api/3/recordings"
OUTPUT_DIR_INITIAL = PROJECT_ROOT / "outputs" / "data" / "initial"
OUTPUT_DIR_INITIAL_CSV = OUTPUT_DIR_INITIAL / "csv"
INITIAL_METADATA_FILE = OUTPUT_DIR_INITIAL_CSV / "initial_metadata.csv"


"""
2) Filter_split_metadata
"""
OUTPUT_DIR_INITIAL_CSV_SPLITS = OUTPUT_DIR_INITIAL_CSV / "splits"

"""
3) Download_audio
"""
OUTPUT_DIR_INITIAL_AUDIO = OUTPUT_DIR_INITIAL / "audio"

"""
4) Normalize_audio 
"""
SAMPLE_RATE = 32000
OUTPUT_DIR_INITIAL_AUDIO_PROCESSED = OUTPUT_DIR_INITIAL / "audio_processed"

"""
5) Extract_clip
"""
CLIP_DURATION = 3
OUTPUT_DIR_FINAL = PROJECT_ROOT / "outputs" / "data" / "final"
OUTPUT_DIR_FINAL_CLIPS = OUTPUT_DIR_FINAL / "clips"
OUTPUT_DIR_FINAL_CSV = OUTPUT_DIR_FINAL / "csv"
FINAL_METADATA_FILE = OUTPUT_DIR_FINAL_CSV / "clips_metadata.csv"
