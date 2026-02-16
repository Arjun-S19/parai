import argparse
import json
import torch
from torch.utils.data import DataLoader
import config as config
from data import DrumDataset
from metrics import accuracy, per_class_accuracy, confusion_matrix
from models.encoders import make_panns_encoder
from models.model import DrumClassifier

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Evaluate a checkpoint on train/validate/test"
    )
    parser.add_argument("checkpoint", type = str)
    parser.add_argument("split", choices = ["train", "validate", "test"])
    parser.add_argument(
        "--encoder",
        choices = sorted(config.encoders.keys()),
        default = "cnn6",
        help = "Encoder backbone used by the checkpoint",
    )
    parser.add_argument("--batch-size", type = int, default = 32)
    parser.add_argument(
        "--save-cm",
        default = None,
        help = "Optional path to save confusion matrix JSON",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    csv_map = {
        "train": "datasets/1_train.csv",
        "validate": "datasets/2_validate.csv",
        "test": "datasets/3_test.csv",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DrumDataset(csv_map[args.split])
    loader = DataLoader(ds, batch_size = args.batch_size, shuffle = False)

    # build the same encoder used during training
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

    overall = accuracy(logits, targets)
    per_class = per_class_accuracy(logits, targets, config.num_classes)
    cm = confusion_matrix(logits, targets, config.num_classes)

    print(f"Encoder: {args.encoder}")
    print(f"Encoder ckpt: {encoder_ckpt_path}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Overall acc: {overall}")
    print(f"Per class acc: {per_class}")

    # print a readable confusion matrix with class labels
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c:>9s}" for c in config.classes])
    print(header)

    for i, row in enumerate(cm):
        row_str = " ".join([f"{v:9d}" for v in row])
        print(f"{config.classes[i]:>12s} {row_str}")

    if args.save_cm is not None:
        out = {
            "classes": config.classes,
            "split": args.split,
            "encoder": args.encoder,
            "checkpoint": args.checkpoint,
            "overall_acc": overall,
            "per_class_acc": per_class,
            "confusion_matrix": cm,
        }
        with open(args.save_cm, "w") as f:
            json.dump(out, f, indent = 2)
        print(f"\nSaved confusion matrix to {args.save_cm}")

if __name__ == "__main__":
    main()