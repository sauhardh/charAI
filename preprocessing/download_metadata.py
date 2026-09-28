from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
import os

from preprocessing import API_URL, INITIAL_METADATA_FILE

# API_URL = "https://xeno-canto.org/api/3/recordings"
# OUTPUT_DIR_INITIAL = "../outputs/data/initial"
# OUTPUT_DIR_INITIAL_CSV = OUTPUT_DIR_INITIAL + "csv"
#
# INITIAL_METADATA_FILE = OUTPUT_DIR_INITIAL_CSV + "initial_metadata.csv"

load_dotenv()
API_KEY = os.getenv("XENO_CANTO_API_KEY")


def fetch_xeno_canto_metadata(
    query: str = "cnt:nepal grp:birds",
    output_csv: Path = Path(INITIAL_METADATA_FILE),
):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    page = 1
    first_page = True

    while True:
        response = requests.get(
            API_URL, params={"query": query, "page": page, "key": API_KEY}, timeout=20
        )

        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            break
        response.raise_for_status()

        data = response.json()
        num_pages = int(data.get("numPages", 1))
        recordings = data.get("recordings", [])

        rows = [
            {
                "recording_id": r.get("id"),
                "scientific_name": f"{r.get('gen', '')} {r.get('sp', '')}".strip(),
                "english_name": r.get("en", ""),
                "quality": r.get("q", ""),
                "length_seconds": r.get("length", 0),
                "audio_url": r.get("file", ""),
                "country": r.get("cnt", ""),
                "license": r.get("lic", ""),
            }
            for r in recordings
        ]

        # save
        df = pd.DataFrame(rows)
        df.to_csv(
            output_csv, mode="w" if first_page else "a", header=first_page, index=False
        )
        first_page = False

        print(
            f"  Page [{page:02d}/{num_pages:02d}] | "
            f"Fetched {len(recordings)} recordings"
        )

        if page >= num_pages:
            break
        page += 1

    print(f"✓ Metadata saved to {output_csv}")


if __name__ == "__main__":
    fetch_xeno_canto_metadata()
