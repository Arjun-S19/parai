import argparse
import csv
import json
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
    parser.add_argument(
        "--path-class-hint",
        default = False,
        help = "Use file/folder name hints to infer drum class when available",
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

def detect_path_class_hint(path: Path):
    """
    Detect a class name from the filename or its parent folder
    """

    def match_classes(text: str):
        normalized_text = text.lower().replace("_", " ").replace("-", " ")
        matches = []

        for class_name in config.classes:
            variants = config.classname_variants.get(class_name, [class_name])
            for variant in variants:
                normalized_variant = variant.lower().replace("_", " ").replace("-", " ")
                if normalized_variant in normalized_text:
                    matches.append(class_name)
                    break

        return matches

    parent_matches = match_classes(path.parent.name)
    stem_matches = match_classes(path.stem)

    if len(parent_matches) > 1:
        return None

    if len(stem_matches) > 1:
        return None

    if len(parent_matches) == 1 and len(stem_matches) == 0:
        return parent_matches[0]

    if len(stem_matches) == 1 and len(parent_matches) == 0:
        return stem_matches[0]

    if len(parent_matches) == 1 and len(stem_matches) == 1:
        if parent_matches[0] == stem_matches[0]:
            return parent_matches[0]
        return None

    return None

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

def compute_energy(logits: torch.Tensor, temperature: float):
    """
    Compute energy score from logits
    Lower values are treated as more in-distribution
    """

    if temperature <= 0.0:
        raise ValueError("OOD temperature must be > 0")

    scaled = logits / temperature
    return -temperature * torch.logsumexp(scaled, dim = 1)

def load_threshold_overrides(path_value):
    """
    Load optional threshold overrides from JSON
    """

    if path_value is None:
        return {}

    path = Path(path_value)
    if not path.is_absolute():
        path = config.project_root / path

    if not path.exists():
        raise FileNotFoundError(f"OOD threshold file not found: {path}")

    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)

def resolve_conf_threshold(
    *,
    pred_idx: int,
    pred_label: str,
    default_threshold: float,
    per_class_thresholds,
):
    """
    Resolve confidence threshold using class-specific overrides when present
    """

    if not isinstance(per_class_thresholds, dict):
        return float(default_threshold)

    if pred_label in per_class_thresholds:
        return float(per_class_thresholds[pred_label])

    idx_key = str(pred_idx)
    if idx_key in per_class_thresholds:
        return float(per_class_thresholds[idx_key])

    return float(default_threshold)

def infer_batch(model, paths, device, *, ood_temperature: float):
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

        # energy score is used for OOD gating
        energy = compute_energy(logits, ood_temperature)

        # convert logits to probabilities for confidence reporting
        probs = torch.softmax(logits, dim = 1)

        # predicted class and confidence per file
        conf, pred = probs.max(dim = 1)

    return (
        pred.detach().cpu(),
        conf.detach().cpu(),
        energy.detach().cpu(),
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

    # map predicted indices back to class names
    idx_to_class = {i: c for i, c in enumerate(config.classes)}

    # central inference and OOD controls from config
    default_conf_threshold = float(getattr(config, "inference_threshold", 0.0))
    ood_cfg = dict(getattr(config, "ood_detection", {}))

    override_cfg = load_threshold_overrides(ood_cfg.get("thresholds_path"))
    if override_cfg:
        ood_cfg = {**ood_cfg, **override_cfg}

    ood_enabled = bool(ood_cfg.get("enabled", False))
    none_label = str(ood_cfg.get("none_label", "none"))
    ood_temperature = float(ood_cfg.get("temperature", 1.0))
    use_path_class_hint = bool(args.path_class_hint)

    energy_threshold = ood_cfg.get("energy_threshold", None)
    if energy_threshold is not None:
        energy_threshold = float(energy_threshold)

    per_class_conf_thresholds = ood_cfg.get("per_class_conf_thresholds", {})
    if "default_conf_threshold" in ood_cfg:
        default_conf_threshold = float(ood_cfg["default_conf_threshold"])

    rows = []
    path_hint_labels = {}

    for p in files:
        path_hint_labels[p] = detect_path_class_hint(p) if use_path_class_hint else None

    files_to_infer = [p for p in files if path_hint_labels[p] is None]
    inferred_rows = {}

    model = None
    if files_to_infer:
        # load trained model for files that still need inference
        model = load_model(
            checkpoint_path = args.checkpoint,
            encoder_name = args.encoder,
            device = device,
        )

    if files_to_infer:
        # batch inference to avoid GPU/memory overload
        for i in range(0, len(files_to_infer), args.batch_size):
            batch = files_to_infer[i:i + args.batch_size]

            pred, conf, energy, probs = infer_batch(
                model,
                batch,
                device,
                ood_temperature = ood_temperature,
            )

            for p, y_hat, c_hat, e_hat, p_vec in zip(
                batch,
                pred.tolist(),
                conf.tolist(),
                energy.tolist(),
                probs.tolist(),
            ):
                pred_label = idx_to_class[y_hat]
                conf_threshold = resolve_conf_threshold(
                    pred_idx = y_hat,
                    pred_label = pred_label,
                    default_threshold = default_conf_threshold,
                    per_class_thresholds = per_class_conf_thresholds,
                )

                conf_pass = c_hat >= conf_threshold
                ood_pass = True
                if ood_enabled and energy_threshold is not None:
                    ood_pass = e_hat <= energy_threshold

                if not ood_pass:
                    decision_reason = "energy_ood"
                elif not conf_pass:
                    decision_reason = "low_conf"
                else:
                    decision_reason = "accepted"

                inferred_rows[str(p)] = {
                    "path": str(p),
                    "pred_label": pred_label if conf_pass and ood_pass else none_label,
                    "raw_pred_label": pred_label,
                    "pred_conf": float(c_hat),
                    "ood_energy": float(e_hat),
                    "ood_pass": bool(ood_pass),
                    "conf_pass": bool(conf_pass),
                    "conf_threshold": float(conf_threshold),
                    "energy_threshold": energy_threshold,
                    "decision_reason": decision_reason,
                    "path_hint_label": "",
                    "probs": str([float(v) for v in p_vec]),
                }

    # store per-file predictions, using hints when available and model output otherwise
    for p in files:
        path_hint_label = path_hint_labels[p]
        if path_hint_label is not None:
            rows.append({
                "path": str(p),
                "pred_label": path_hint_label,
                "raw_pred_label": path_hint_label,
                "pred_conf": 1.0,
                "ood_energy": "",
                "ood_pass": "",
                "conf_pass": "",
                "conf_threshold": "",
                "energy_threshold": "",
                "decision_reason": "path_hint",
                "path_hint_label": path_hint_label,
                "probs": "",
            })
        else:
            rows.append(inferred_rows[str(p)])

    # write results
    out_path.parent.mkdir(parents = True, exist_ok = True)

    with open(out_path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames = [
                "path",
                "pred_label",
                "raw_pred_label",
                "pred_conf",
                "ood_energy",
                "ood_pass",
                "conf_pass",
                "conf_threshold",
                "energy_threshold",
                "decision_reason",
                "path_hint_label",
                "probs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out_path}")

if __name__ == "__main__":
    main()