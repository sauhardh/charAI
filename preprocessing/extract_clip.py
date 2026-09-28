from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
from typing import cast

from preprocessing import (
    CLIP_DURATION,
    OUTPUT_DIR_FINAL_CLIPS,
    OUTPUT_DIR_INITIAL_AUDIO_PROCESSED,
    OUTPUT_DIR_INITIAL_CSV_SPLITS,
    PROJECT_ROOT,
    SAMPLE_RATE,
    FINAL_METADATA_FILE,
)

CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
STRIDE_SAMPLES = int(2.0 * SAMPLE_RATE)


class BirdActivityDetector:
    def __init__(self, low_hz: float = 300.0, high_hz: float = 15000.0, k: float = 1.1):
        nyquist = SAMPLE_RATE / 2.0
        self.sos = butter(
            3, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos"
        )
        self.k = k

    def detect(self, audio: np.ndarray) -> np.ndarray:
        filtered = cast(np.ndarray, sosfilt(self.sos, audio))
        rms = librosa.feature.rms(y=filtered, frame_length=1024, hop_length=512)[0]
        threshold = np.median(rms) + self.k * np.std(rms)
        return rms > threshold


# CLIP_DURATION = 3
# OUTPUT_DIR_FINAL = PROJECT_ROOT / "outputs" / "data" / "final"
# OUTPUT_DIR_FINAL_CLIPS = OUTPUT_DIR_FINAL / "clips"
# OUTPUT_DIR_FINAL_CSV = OUTPUT_DIR_FINAL / "csv"
# FINAL_METADATA_FILE = OUTPUT_DIR_FINAL_CSV / "clips_metadata.csv"


def extract_clips_from_dataset(
    processed_dir: Path = OUTPUT_DIR_INITIAL_AUDIO_PROCESSED,
    splits_dir: Path = OUTPUT_DIR_INITIAL_CSV_SPLITS,
    output_clips_dir: Path = OUTPUT_DIR_FINAL_CLIPS,
    output_csv: Path = FINAL_METADATA_FILE,
):
    detector = BirdActivityDetector()
    records = []

    for split in ["train", "val", "test"]:
        split_csv = splits_dir / f"{split}.csv"

        if not split_csv.exists():
            continue

        meta_df = pd.read_csv(split_csv)
        rec_to_species = dict(
            zip(meta_df["recording_id"].astype(str), meta_df["scientific_name"])
        )

        split_audio_dir = processed_dir / split
        wav_files = list(split_audio_dir.glob("*.wav"))

        for wav_path in wav_files:
            rec_id = wav_path.stem.replace("XC", "")
            species = rec_to_species.get(rec_id, "unknown")
            species_clean = species.replace(" ", "_")
            print("rec_id", rec_id)

            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if len(audio) < CLIP_SAMPLES:
                    continue

                active_mask = detector.detect(audio)
                n_frames = len(active_mask)

                # Slide a 3.0s window
                start = 0
                clip_idx = 0

                while start + CLIP_SAMPLES <= len(audio):
                    frame_start = int((start / len(audio)) * n_frames)
                    frame_end = int(((start + CLIP_SAMPLES) / len(audio)) * n_frames)

                    # Keep only if at least 15% contains active bird call energy
                    if active_mask[frame_start:frame_end].mean() > 0.15:
                        clip_data = audio[start : start + CLIP_SAMPLES]
                        sec_offset = start / SAMPLE_RATE

                        print("Species clean", species_clean)
                        species_dir = output_clips_dir / split / species_clean
                        species_dir.mkdir(parents=True, exist_ok=True)

                        clip_file = species_dir / f"XC{rec_id}_{sec_offset:.2f}.wav"
                        print("Clip file", clip_file)
                        sf.write(str(clip_file), clip_data, SAMPLE_RATE)

                        records.append(
                            {
                                "recording_id": rec_id,
                                "clip_path": str(clip_file.relative_to(PROJECT_ROOT)),
                                # "clip_path": str(clip_file.resolve()),
                                "species": species,
                                "split": split,
                                "start_sec": sec_offset,
                                "end_sec": sec_offset + CLIP_DURATION,
                            }
                        )
                        clip_idx += 1

                    start += STRIDE_SAMPLES

            except Exception as e:
                print(f"Error extracting {wav_path.name}: {e}")

    df_out = pd.DataFrame(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(
        f"\n✓ Master clips table created: {len(df_out)} clips across {df_out['species'].nunique()} species saved to {output_path}"
    )


if __name__ == "__main__":
    extract_clips_from_dataset()
