import pathlib
from typing import Dict
import torch
import torchaudio
import config

# audio normalization targets
TARGET_SAMPLE_RATE = config.sample_rate
TARGET_DURATION = config.duration
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)

# dir paths
PROJECT_ROOT = config.project_root
INPUT_PATH = PROJECT_ROOT / "drum_samples"
OUTPUT_PATH = PROJECT_ROOT / f"normalized_{INPUT_PATH.stem}"

AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aif",
    ".aiff",
    ".m4a",
    ".wma",
}

def normalize_dir(*,
                  path: pathlib.Path,
                  output_dir: pathlib.Path,
                  base_dir: pathlib.Path | None = None,
                  ) -> Dict[pathlib.Path, pathlib.Path]:
    """
    Load audio files and normalize them into a uniform shape + format:
    - mono
    - resampled to 32 kHz
    - padded/cropped to 1.5 second
    - amplitude normalized to [-1, 1]

    Returns a dictionary mapping each source file to its normalized output path
    """

    path = pathlib.Path(path)
    if base_dir is None:
        base_dir = path

    if path.is_dir():
        candidates: list[pathlib.Path] = []
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in AUDIO_EXTS:
                continue
            candidates.append(file_path)
        candidates = sorted(candidates)
    elif path.is_file():
        candidates = [path]
    else:
        raise FileNotFoundError(f"{path} does not exist")

    if not candidates:
        raise ValueError(f"No audio files found in {path}")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    normalized: Dict[pathlib.Path, pathlib.Path] = {}
    for file_path in candidates:
        # load audio -> shape [channels, time]
        waveform, sr = torchaudio.load(str(file_path))

        # convert stereo -> mono by averaging channels
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim = 0, keepdim = True)

        # resample if original sample rate doesn't match requirements
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq = sr,
                new_freq = TARGET_SAMPLE_RATE,
            )

        # ensure exact number of samples (fixed length)
        num_samples = waveform.size(1)

        if num_samples < TARGET_SAMPLES:
            # short sounds -> pad zeros at the end
            pad_amount = TARGET_SAMPLES - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > TARGET_SAMPLES:
            # longer sounds -> crop to preserve the transient at the start
            waveform = waveform[:, :TARGET_SAMPLES]

        # peak normalize -> ensures amplitude is consistent across samples
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        relative_path = file_path.relative_to(base_dir)
        out_path = output_dir / relative_path
        out_path = out_path.with_name(f"{out_path.stem}_normalized.wav")
        out_path.parent.mkdir(parents = True, exist_ok = True)
        torchaudio.save(str(out_path), waveform, sample_rate = TARGET_SAMPLE_RATE)
        normalized[file_path] = out_path

    return normalized

def normalize_single(path: str) -> torch.Tensor:
    """
    Load and normalize a single audio file
    """
    
    waveform, sr = torchaudio.load(path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq = sr,
            new_freq = TARGET_SAMPLE_RATE,
        )

    num_samples = waveform.size(1)

    if num_samples < TARGET_SAMPLES:
        pad_amount = TARGET_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform

def main() -> None:
    input_dir = pathlib.Path(INPUT_PATH)
    out_dir = pathlib.Path(OUTPUT_PATH)

    normalize_dir(
        path = input_dir,
        output_dir = out_dir,
        base_dir = input_dir,
    )

if __name__ == "__main__":
    main()
