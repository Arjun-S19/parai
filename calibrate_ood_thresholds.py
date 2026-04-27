import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import config as config
from data import DrumDataset
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

def compute_energy(logits: torch.Tensor, temperature: float):
    if temperature <= 0.0:
        raise ValueError("Temperature must be > 0")

    scaled = logits / temperature
    return -temperature * torch.logsumexp(scaled, dim = 1)

def quantile(values: torch.Tensor, q: float):
    if values.numel() == 0:
        return 0.0

    q = max(0.0, min(1.0, float(q)))
    return float(torch.quantile(values, q).item())

def gather_audio_files(path: Path):
    if path.is_file():
        return [path] if path.suffix.lower() in AUDIO_EXTS else []

    if path.is_dir():
        files = []
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
        return sorted(files)

    raise FileNotFoundError(f"{path} does not exist")

def infer_audio_paths(model, paths, *, batch_size: int, device: str, temperature: float):
    all_logits = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        xs = [normalize_single(str(p)) for p in batch]
        x = torch.stack(xs, dim = 0).to(device)

        with torch.no_grad():
            logits = model(x)
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim = 0)
    probs = torch.softmax(logits, dim = 1)
    conf, pred = probs.max(dim = 1)
    energy = compute_energy(logits, temperature)
    return logits, conf, pred, energy

def optimize_thresholds(
    *,
    id_conf: torch.Tensor,
    id_pred: torch.Tensor,
    id_energy: torch.Tensor,
    base_per_class_conf_thresholds,
    ood_conf: torch.Tensor,
    ood_pred: torch.Tensor,
    ood_energy: torch.Tensor,
    min_id_retention: float,
    max_conf_multiplier: float,
    conf_steps: int,
):
    """
    Jointly optimize global energy threshold and confidence multiplier
    """

    min_id_retention = float(max(0.0, min(1.0, min_id_retention)))
    max_conf_multiplier = max(1.0, float(max_conf_multiplier))
    conf_steps = max(1, int(conf_steps))

    # energy thresholds are searched on observed values only
    energy_candidates = torch.unique(torch.cat([id_energy, ood_energy], dim = 0)).tolist()
    energy_candidates.sort()

    if conf_steps == 1:
        conf_multipliers = [1.0]
    else:
        conf_multipliers = [
            1.0 + (max_conf_multiplier - 1.0) * (i / (conf_steps - 1))
            for i in range(conf_steps)
        ]

    id_base_th = torch.tensor(
        [base_per_class_conf_thresholds[config.classes[int(i)]] for i in id_pred.tolist()],
        dtype = torch.float32,
    )
    ood_base_th = torch.tensor(
        [base_per_class_conf_thresholds[config.classes[int(i)]] for i in ood_pred.tolist()],
        dtype = torch.float32,
    )

    best = None
    best_score = -1.0

    for conf_mult in conf_multipliers:
        id_conf_pass = id_conf >= (id_base_th * conf_mult)
        ood_conf_pass = ood_conf >= (ood_base_th * conf_mult)

        for energy_th in energy_candidates:
            id_accept = id_conf_pass & (id_energy <= energy_th)
            ood_accept = ood_conf_pass & (ood_energy <= energy_th)

            id_retention = float(id_accept.float().mean().item())
            ood_reject = float((~ood_accept).float().mean().item())

            if id_retention < min_id_retention:
                continue

            # prioritize OOD rejection while preferring higher ID retention on ties
            score = (10.0 * ood_reject) + id_retention
            if score > best_score:
                best_score = score
                best = {
                    "conf_multiplier": float(conf_mult),
                    "energy_threshold": float(energy_th),
                    "id_retention": id_retention,
                    "ood_reject": ood_reject,
                }

    return best

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Calibrate energy and confidence thresholds for OOD gating"
    )
    parser.add_argument("--checkpoint", required = True, help = "Trained checkpoint path")
    parser.add_argument(
        "--split",
        choices = ["train", "validate", "test"],
        default = "validate",
        help = "Dataset split to use for calibration",
    )
    parser.add_argument(
        "--encoder",
        default = "cnn6",
        choices = sorted(config.encoders.keys()),
        help = "Encoder backbone used by checkpoint",
    )
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument(
        "--device",
        default = None,
        help = "cuda | cpu (default: auto)",
    )
    parser.add_argument(
        "--temperature",
        type = float,
        default = 1.0,
        help = "Temperature used in energy computation",
    )
    parser.add_argument(
        "--energy-percentile",
        type = float,
        default = 95.0,
        help = "ID retention percentile for global energy threshold",
    )
    parser.add_argument(
        "--conf-percentile",
        type = float,
        default = 5.0,
        help = "Lower confidence percentile per class",
    )
    parser.add_argument(
        "--output",
        default = "runs/diagnostics/ood_thresholds.json",
        help = "Output JSON path",
    )
    parser.add_argument(
        "--ood-input",
        default = None,
        help = "Optional file/folder of OOD audio to tune thresholds against",
    )
    parser.add_argument(
        "--min-id-retention",
        type = float,
        default = 0.95,
        help = "Minimum accepted in-domain retention during OOD tuning",
    )
    parser.add_argument(
        "--max-conf-multiplier",
        type = float,
        default = 1.5,
        help = "Upper bound when scaling class confidence thresholds in OOD tuning",
    )
    parser.add_argument(
        "--conf-grid-steps",
        type = int,
        default = 51,
        help = "Number of confidence multiplier steps for OOD tuning",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    csv_map = {
        "train": "datasets/1_train.csv",
        "validate": "datasets/2_validate.csv",
        "test": "datasets/3_test.csv",
    }

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ds = DrumDataset(csv_map[args.split])
    loader = DataLoader(ds, batch_size = args.batch_size, shuffle = False)

    encoder_ckpt_path = config.project_root / config.encoders[args.encoder]

    encoder = make_panns_encoder(
        name = args.encoder,
        checkpoint_path = encoder_ckpt_path,
        device = device,
        freeze = True,
    )

    model = DrumClassifier(
        encoder = encoder,
        num_classes = config.num_classes,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location = device))
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim = 0)
    targets = torch.cat(all_targets, dim = 0)

    energy = compute_energy(logits, args.temperature)
    probs = torch.softmax(logits, dim = 1)
    conf, pred = probs.max(dim = 1)

    energy_q = max(0.0, min(100.0, float(args.energy_percentile))) / 100.0
    conf_q = max(0.0, min(100.0, float(args.conf_percentile))) / 100.0

    energy_threshold = quantile(energy, energy_q)

    # default confidence threshold from correctly classified predictions
    is_correct = pred.eq(targets)
    default_conf_threshold = quantile(conf[is_correct], conf_q)

    per_class_conf_thresholds = {}
    counts_per_class = {}

    for class_idx, class_name in enumerate(config.classes):
        class_mask = targets.eq(class_idx)
        class_correct_mask = class_mask & is_correct
        class_conf = conf[class_correct_mask]

        per_class_conf_thresholds[class_name] = quantile(class_conf, conf_q)
        counts_per_class[class_name] = int(class_mask.sum().item())

    tuning_stats = None
    if args.ood_input is not None:
        ood_path = Path(args.ood_input)
        ood_files = gather_audio_files(ood_path)

        if not ood_files:
            raise ValueError(f"No audio files found under OOD input: {ood_path}")

        _, ood_conf, ood_pred, ood_energy = infer_audio_paths(
            model,
            ood_files,
            batch_size = args.batch_size,
            device = device,
            temperature = args.temperature,
        )

        best = optimize_thresholds(
            id_conf = conf,
            id_pred = pred,
            id_energy = energy,
            base_per_class_conf_thresholds = per_class_conf_thresholds,
            ood_conf = ood_conf,
            ood_pred = ood_pred,
            ood_energy = ood_energy,
            min_id_retention = args.min_id_retention,
            max_conf_multiplier = args.max_conf_multiplier,
            conf_steps = args.conf_grid_steps,
        )

        if best is None:
            print("Warning: no threshold pair met min ID retention; keeping percentile thresholds")
            tuning_stats = {
                "ood_input": str(ood_path),
                "ood_sample_count": int(len(ood_files)),
                "tuned": False,
                "reason": "no-feasible-solution",
            }
        else:
            conf_multiplier = best["conf_multiplier"]
            energy_threshold = best["energy_threshold"]

            for class_name in per_class_conf_thresholds:
                per_class_conf_thresholds[class_name] = float(
                    per_class_conf_thresholds[class_name] * conf_multiplier
                )

            default_conf_threshold = float(default_conf_threshold * conf_multiplier)

            tuning_stats = {
                "ood_input": str(ood_path),
                "ood_sample_count": int(len(ood_files)),
                "tuned": True,
                "selected_conf_multiplier": conf_multiplier,
                "selected_energy_threshold": energy_threshold,
                "id_retention": best["id_retention"],
                "ood_reject": best["ood_reject"],
                "min_id_retention": float(args.min_id_retention),
            }

    out = {
        "enabled": True,
        "none_label": "none",
        "temperature": float(args.temperature),
        "energy_threshold": float(energy_threshold),
        "energy_percentile": float(args.energy_percentile),
        "default_conf_threshold": float(default_conf_threshold),
        "conf_percentile": float(args.conf_percentile),
        "per_class_conf_thresholds": per_class_conf_thresholds,
        "encoder": args.encoder,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sample_count": int(targets.numel()),
        "counts_per_class": counts_per_class,
        "tuning": tuning_stats,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = config.project_root / out_path
    out_path.parent.mkdir(parents = True, exist_ok = True)

    out_path.write_text(json.dumps(out, indent = 2), encoding = "utf-8")

    print("Saved OOD thresholds:", out_path)
    print("Energy threshold:", out["energy_threshold"])
    print("Default conf threshold:", out["default_conf_threshold"])
    print("Per-class confidence thresholds:")
    for class_name in config.classes:
        print(" ", class_name, "->", out["per_class_conf_thresholds"][class_name])

if __name__ == "__main__":
    main()