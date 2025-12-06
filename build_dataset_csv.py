import pathlib
import random
import pandas as pd

# dir paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DRUM_SAMPLES_PATH = PROJECT_ROOT / "drum_samples"
OUTPUT_PATH = PROJECT_ROOT / "datasets"

# dataset ratios
TRAINING_RATIO = 0.7
VALIDATION_RATIO = 0.2
TESTING_RATIO = 0.1

RANDOM_SEED = 42

# accepted audio extensions
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


def infer_label(root: pathlib.Path, file_path: pathlib.Path) -> str | None:
    """
    Extract the audio label from folder name
    Example:
        drum_samples/Kick/sound1.wav -> 'kick'
    """
    rel = file_path.relative_to(root)

    if len(rel.parts) == 0:
        return None

    top = rel.parts[0]
    class_name = top.lower().replace("_", " ").strip()
    return class_name


def collect_samples(root: pathlib.Path) -> list[tuple[str, str]]:
    """
    Go through drum_samples dir and get (filepath, label) pairs for valid files
    """
    samples = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in AUDIO_EXTS:
            continue

        label = infer_label(root, path)
        if label is None:
            continue

        samples.append((str(path.resolve()), label))

    return samples


def split_dataset(samples: list[tuple[str, str]],
                  train_ratio: float = TRAINING_RATIO,
                  validate_ratio: float = VALIDATION_RATIO
                  ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Shuffle and split dataset into train/validate/test according ratios defined by constants
    """
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_validate = int(n * validate_ratio)

    train = samples[:n_train]
    validate = samples[n_train:n_train + n_validate]
    test = samples[n_train + n_validate:]

    return train, validate, test


def save_csv(samples: list[tuple[str, str]], path: pathlib.Path) -> None:
    """
    Save a list of (path, label) tuples into a CSV file
    """
    df = pd.DataFrame(samples, columns = ["path", "label"])
    df.to_csv(path, index = False)


def main() -> None:
    samples_dir = DRUM_SAMPLES_PATH.resolve()
    out_dir = OUTPUT_PATH.resolve()
    out_dir.mkdir(parents = True, exist_ok = True)

    # gather labeled audio files
    samples = collect_samples(samples_dir)
    if not samples:
        raise SystemExit("No labeled audio files found under root")

    # create train/validate/test splits
    train, validate, test = split_dataset(samples)

    # save CSV output
    save_csv(samples, out_dir / "0_dataset.csv")
    save_csv(train, out_dir / "1_train.csv")
    save_csv(validate, out_dir / "2_validate.csv")
    save_csv(test, out_dir / "3_test.csv")

    print(f"Total samples = {len(samples)}")
    print(
        f"Training ({TRAINING_RATIO}) = {len(train)}, "
        f"Validation ({VALIDATION_RATIO}) = {len(validate)}, "
        f"Testing ({TESTING_RATIO}) = {len(test)}"
    )
    print(f"CSV files written to {out_dir}")


if __name__ == "__main__":
    main()