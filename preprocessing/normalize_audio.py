from pathlib import Path
import librosa
import soundfile as sf

from preprocessing import (
    SAMPLE_RATE,
    OUTPUT_DIR_INITIAL_AUDIO,
    OUTPUT_DIR_INITIAL_AUDIO_PROCESSED,
)


def normalize_recording(
    input_path: Path, output_path: Path, sample_rate: int = SAMPLE_RATE
) -> bool:
    if output_path.exists():
        print("Looks like it's already done")
        return True

    try:
        waveform, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, sample_rate)
        return True
    except Exception as e:
        print(f"Error normalizing {input_path.name}: {e}")
        return False


def normalize_all_audio(
    raw_dir: Path = OUTPUT_DIR_INITIAL_AUDIO,
    processed_dir: Path = OUTPUT_DIR_INITIAL_AUDIO_PROCESSED,
):
    for split in ["train", "test", "val"]:
        input_split = raw_dir / split
        output_split = processed_dir / split

        if not input_split.exists():
            continue
        files = list(input_split.glob("*.mp3")) + list(input_split.glob("*.wav"))
        print(f"\n⚙️ Normalizing {len(files)} files in '{split}' to 32 kHz mono...")

        success = 0

        for f in files:
            dest = output_split / f"{f.stem}.wav"
            if normalize_recording(f, dest):
                success += 1

        print(f"✓ Normalized {success}/{len(files)} files into {output_split}")


if __name__ == "__main__":
    normalize_all_audio()
