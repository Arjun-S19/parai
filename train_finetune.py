import argparse
import csv
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config as config
from data import DrumDataset
from metrics import accuracy, per_class_accuracy
from models.encoders import make_panns_encoder
from models.model import DrumClassifier

def compute_class_weights(csv_path, num_classes, mode = "inv", max_weight = None):
    """
    Compute class weights to reduce bias from class imbalance
    """

    counts = [0] * num_classes
    label_to_id = {name: i for i, name in enumerate(config.classes)}

    with open(csv_path, "r", newline = "") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise KeyError(
                f"CSV must have a 'label' column; Found: {reader.fieldnames}"
            )

        for row in reader:
            raw = row["label"].strip()

            # treat labels as class names first
            if raw in label_to_id:
                y = label_to_id[raw]
            elif raw.isdigit():
                y = int(raw)
            else:
                raise KeyError(
                    f"Unknown label '{raw}', expected one of: {list(label_to_id.keys())}"
                )

            if y < 0 or y >= num_classes:
                raise ValueError(
                    f"Label '{raw}' mapped to id {y}, but num_classes = {num_classes}"
                    f"config.classes = {config.classes}"
                )

            counts[y] += 1

    total = sum(counts)

    weights = []
    for c in counts:
        if c == 0:
            w = 0.0
        else:
            if mode == "inv":
                w = total / (num_classes * c)
            elif mode == "sqrt_inv":
                w = math.sqrt(total / (num_classes * c))
            else:
                raise ValueError("Mode must be 'inv' or 'sqrt_inv'")

        if max_weight is not None:
            w = min(w, float(max_weight))

        weights.append(w)

    return torch.tensor(weights, dtype = torch.float32)

def set_trainable_last_block(encoder, block_name = "conv_block4"):
    """
    Freeze all encoder params, then unfreeze only the last conv block
    """

    base = encoder.base_model

    for p in base.parameters():
        p.requires_grad = False

    if not hasattr(base, block_name):
        raise AttributeError(
            f"Encoder base model missing '{block_name}'; Available: {list(dict(base.named_children()).keys())}"
        )

    last = getattr(base, block_name)
    for p in last.parameters():
        p.requires_grad = True

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Run one training epoch over the loader
    """

    model.train()

    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)

def eval_epoch(model, loader, criterion, device):
    """
    Evaluate model on a loader and return metrics
    """

    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            total_acc += accuracy(logits, y)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    per_class = per_class_accuracy(
        logits, targets, num_classes = config.num_classes
    )

    return (
        total_loss / len(loader),
        total_acc / len(loader),
        per_class,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Fine-tune last encoder block + classifier head"
    )
    parser.add_argument(
        "--encoder",
        default = "cnn6",
        choices = sorted(config.encoders.keys()),
        help = "Encoder backbone to use (cnn6/cnn10/cnn14)",
    )
    parser.add_argument(
        "--epochs",
        type = int,
        default = 15,
        help = "Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 32,
        help = "Mini-batch size",
    )
    parser.add_argument(
        "--head-lr",
        type = float,
        default = 1e-4,
        help = "Learning rate for the classifier head",
    )
    parser.add_argument(
        "--encoder-lr",
        type = float,
        default = 1e-5,
        help = "Learning rate for the unfrozen encoder block",
    )
    parser.add_argument(
        "--init-checkpoint",
        default = None,
        help = "Optional checkpoint to initialize from (head-only best_val.pt)",
    )
    parser.add_argument(
        "--weighted-loss",
        action = "store_true",
        help = "Use class weights for CrossEntropyLoss",
    )
    parser.add_argument(
        "--weight-mode",
        default = "sqrt_inv",
        choices = ["inv", "sqrt_inv"],
        help = "Weighting strength (sqrt_inv is softer)",
    )
    parser.add_argument(
        "--max-weight",
        type = float,
        default = 2.5,
        help = "Optional cap on class weights",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path("runs") / f"{args.encoder}_finetune_lastblock"
    run_dir.mkdir(parents = True, exist_ok = True)

    train_csv = "datasets/1_train.csv"
    val_csv = "datasets/2_validate.csv"

    train_ds = DrumDataset(train_csv)
    val_ds = DrumDataset(val_csv)

    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False)

    encoder_ckpt_path = config.project_root / config.encoders[args.encoder]

    # build encoder trainable
    encoder = make_panns_encoder(
        name = args.encoder,
        checkpoint_path = encoder_ckpt_path,
        device = device,
        freeze = False,
    )

    model = DrumClassifier(
        encoder = encoder,
        num_classes = config.num_classes,
    ).to(device)

    # optionally start from a head-only checkpoint for faster convergence
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location = device)
        model.load_state_dict(state, strict = True)

    # freeze encoder then unfreeze only last conv block
    set_trainable_last_block(model.encoder, block_name = "conv_block4")

    # two lr groups: small lr for encoder block, larger lr for head
    enc_params = [p for p in model.encoder.base_model.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": enc_params, "lr": args.encoder_lr},
        ]
    )

    if args.weighted_loss:
        class_weights = compute_class_weights(
            train_csv,
            num_classes = config.num_classes,
            mode = args.weight_mode,
            max_weight = args.max_weight,
        ).to(device)

        print("Using weighted loss with class weights:", class_weights.detach().cpu().tolist())
        criterion = nn.CrossEntropyLoss(weight = class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, per_class = eval_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class": per_class,
        })

        print(
            f"epoch {epoch:02d} | "
            f"train acc {train_acc:.3f} | "
            f"val acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best_val.pt")

        torch.save(model.state_dict(), run_dir / "last.pt")

    with open(run_dir / "run_config.json", "w") as f:
        json.dump({
            "encoder": args.encoder,
            "encoder_ckpt": str(encoder_ckpt_path),
            "classes": config.classes,
            "sample_rate": config.sample_rate,
            "duration": config.duration,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "encoder_lr": args.encoder_lr,
            "weighted_loss": args.weighted_loss,
            "weight_mode": args.weight_mode,
            "max_weight": args.max_weight,
            "init_checkpoint": args.init_checkpoint,
            "unfrozen_block": "conv_block4",
        }, f, indent = 2)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent = 2)

if __name__ == "__main__":
    main()