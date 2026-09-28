from pathlib import Path
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from preprocessing import OUTPUT_DIR_INITIAL_CSV_SPLITS, OUTPUT_DIR_INITIAL_AUDIO


def download_single_file(url: str, output_path: Path) -> bool:
    if output_path.exists() and output_path.stat().st_size > 1000:  # File > 1KB
        return True  # Already exists
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False


def download_dataset_audio(
    splits_dir: Path = OUTPUT_DIR_INITIAL_CSV_SPLITS,
    raw_dir: Path = OUTPUT_DIR_INITIAL_AUDIO,
    max_workers: int = 8,
):
    print("Downloading...")
    for split in ["train", "val", "test"]:
        csv_file = splits_dir / f"{split}.csv"
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        split_dir = raw_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers) as executor:
            futures = {}
            for _, row in df.iterrows():
                url = row["audio_url"]
                rec_id = str(row["recording_id"])
                dest = split_dir / f"XC{rec_id}.mp3"

                futures[executor.submit(download_single_file, str(url), dest)] = rec_id

            success = 0
            for future in as_completed(futures):
                if future.result():
                    success += 1

            print(f"✓ '{split}' download complete: {success}/{len(df)} files available")


if __name__ == "__main__":
    download_dataset_audio()
