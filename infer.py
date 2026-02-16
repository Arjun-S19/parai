import argparse
import csv
from pathlib import Path
import torch
import config as config
from audio_normalization import normalize_single
from models.encoders import make_panns_encoder
from models.model import DrumClassifier

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

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run inference on a file or folder using a trained checkpoint"
    )
    parser.add_argument(
        "--input",
        required = True,
        help = "Path to an audio file or a folder to recursively scan",
    )
    parser.add_argument(
        "--out",
        default = "predictions.csv",
        help = "Output CSV path",
    )
    parser.add_argument(
        "--checkpoint",
        required = True,
        help = "Path to trained head checkpoint",
    )
    parser.add_argument(
        "--encoder",
        default = "cnn6",
        choices = sorted(config.encoders.keys()),
        help = "Encoder backbone (must match training)",
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 32,
        help = "Batch size for inference",
    )
    parser.add_argument(
        "--device",
        default = None,
        help = "cuda | cpu (default: auto)",
    )
    return parser.parse_args()

def gather_audio_files(path: Path):
    """
    Collect all valid audio files from a file or directory (recursive)
    """

    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTS:
            return [path]
        return []

    if path.is_dir():
        files = []
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
        return sorted(files)

    raise FileNotFoundError(f"{path} does not exist")

def load_model(*, checkpoint_path: str, encoder_name: str, device: str):
    """
    Build frozen encoder + classifier head and load trained weights
    """

    # load pretrained PANNs encoder weights
    encoder_ckpt_path = config.project_root / config.encoders[encoder_name]

    encoder = make_panns_encoder(
        name = encoder_name,
        checkpoint_path = encoder_ckpt_path,
        device = device,
        freeze = True,
    )

    # combine encoder with trained classifier head
    model = DrumClassifier(
        encoder = encoder,
        num_classes = config.num_classes,
    ).to(device)

    # load trained head parameters
    state = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state)

    model.eval()
    return model

def infer_batch(model, paths, device):
    """
    Run normalization + forward pass on a batch of audio files
    """

    # normalize each audio file to the fixed training contract
    xs = []
    for p in paths:
        x = normalize_single(str(p))
        xs.append(x)

    # shape: (batch, 1, samples)
    x = torch.stack(xs, dim = 0).to(device)

    with torch.no_grad():
        logits = model(x)

        # convert logits to probabilities for confidence reporting
        probs = torch.softmax(logits, dim = 1)

        # predicted class and confidence per file
        conf, pred = probs.max(dim = 1)

    return (
        pred.detach().cpu(),
        conf.detach().cpu(),
        probs.detach().cpu(),
    )

def main():
    args = parse_args()

    # auto-select device if not specified
    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    input_path = Path(args.input)
    out_path = Path(args.out)

    # gather files to classify
    files = gather_audio_files(input_path)
    if not files:
        raise ValueError(f"No audio files found under {input_path}")

    # load trained model
    model = load_model(
        checkpoint_path = args.checkpoint,
        encoder_name = args.encoder,
        device = device,
    )

    # map predicted indices back to class names
    idx_to_class = {i: c for i, c in enumerate(config.classes)}

    rows = []

    # batch inference to avoid GPU/memory overload
    for i in range(0, len(files), args.batch_size):
        batch = files[i:i + args.batch_size]

        pred, conf, probs = infer_batch(model, batch, device)

        # store per-file predictions
        for p, y_hat, c_hat, p_vec in zip(
            batch, pred.tolist(), conf.tolist(), probs.tolist()
        ):
            rows.append({
                "path": str(p),
                "pred_label": idx_to_class[y_hat],
                "pred_conf": float(c_hat),
                "probs": str([float(v) for v in p_vec]),
            })

    # write results
    out_path.parent.mkdir(parents = True, exist_ok = True)

    with open(out_path, "w", newline = "") as f:
        writer = csv.DictWriter(
            f,
            fieldnames = ["path", "pred_label", "pred_conf", "probs"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out_path}")

if __name__ == "__main__":
    main()