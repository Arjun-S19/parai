import pathlib
import torch
import torchaudio

# audio normalization targets
TARGET_SAMPLE_RATE = 32000
TARGET_DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)

# dir paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
INPUT_PATH = PROJECT_ROOT / "drum_samples/snare/snare4.mp3"
OUTPUT_PATH = PROJECT_ROOT / f"{INPUT_PATH.stem}_normalized.wav"


def load_and_normalize(path: str) -> torch.Tensor:
    """
    Load audio files and normalize them into a uniform shape + format:
    - mono
    - resampled to 32 kHz
    - padded/cropped to 1.0 second
    - amplitude normalized to [-1, 1]
    """

    # load audio -> shape [channels, time]
    waveform, sr = torchaudio.load(path)

    # convert stereo -> mono by averaging channels
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    # resample if original sample rate doesn't match model requirements
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

    return waveform


def main() -> None:
    input_dir = pathlib.Path(INPUT_PATH)
    out_dir = pathlib.Path(OUTPUT_PATH)

    waveform = load_and_normalize(str(input_dir))
    torchaudio.save(str(out_dir), waveform, sample_rate = TARGET_SAMPLE_RATE)

    print(f"Normalized audio saved to {out_dir}")


if __name__ == "__main__":
    main()
